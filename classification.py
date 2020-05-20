import os
import pandas as pd
import numpy as np
import random
import itertools
import json
import argparse
import datetime
import time
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
import torch
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import (
    AdamW, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup,
    BertTokenizer, BertForSequenceClassification, 
    XLNetTokenizer, XLNetForSequenceClassification, 
    RobertaTokenizer, RobertaForSequenceClassification, RobertaTokenizer,
    XLMRobertaForSequenceClassification, XLMRobertaTokenizer,
    CamembertForSequenceClassification, CamembertTokenizer,
    DistilBertForSequenceClassification, DistilBertTokenizer,
    AutoTokenizer, AutoModelForSequenceClassification
)
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import confusion_matrix, label_ranking_average_precision_score, precision_recall_fscore_support, accuracy_score, hamming_loss
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument('--dset', type=str, default='data/entire', help='name of dataset')
parser.add_argument('--seed', type=int, default=123, help='seed')
parser.add_argument('--lr', type=float, default=2e-5, help='learning rate')
parser.add_argument('--decay', type=float, default=0.01, help='weight decay ate')
parser.add_argument('--warmups', type=int, default=1000, help='warmups')
parser.add_argument('--eps', type=float, default=1e-8, help='eps for Adam')
parser.add_argument('--n_epochs', type=int, default=300, help='number of epochs')
parser.add_argument('--batch', type=int, default=32, help='batch_size')
parser.add_argument('--model_class', type=str, default='bert', help='model class')
parser.add_argument('--pretrained_model', type=str, help='pretrained model name or model path')
parser.add_argument('--pretrained_tokenizer', type=str, help='pretrained model name or model path for tokenizer')
parser.add_argument('--ensemble', type=str, help='ensemble model class')
parser.add_argument('--cls_type', type=str, choices=['multilabel', 'binary'], help='multilabel classification or binary classification')
args = parser.parse_args() 
dataset = args.dset
ensemble = args.ensemble
cls_type = args.cls_type

MODEL_CLASSES = {
    'bert': (BertForSequenceClassification, BertTokenizer, 'bert-base-uncased'),
    'xlnet': (XLNetForSequenceClassification, XLNetTokenizer, 'xlnet-base-cased'),
    'clinicalxlnet': (XLNetForSequenceClassification, XLNetTokenizer, 'xlnet-base-cased'),
    'roberta': (RobertaForSequenceClassification, RobertaTokenizer, 'roberta-base'),
    'xlm-roberta': (XLMRobertaForSequenceClassification, XLMRobertaTokenizer, 'xlm-roberta-base'),
    'camembert': (CamembertForSequenceClassification, CamembertTokenizer, 'camembert-base'),
    'distilbert': (DistilBertForSequenceClassification, DistilBertTokenizer, 'distilbert-base-uncased'),
}

class TransformerCLS():
    def __init__(self):
        self.cls_type = cls_type
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_grad_norm = 1.0
        self.fine_tuning = True
        self.max_seq_length = 256
        self.set_seed(args.seed)
        self.sentences = {}
        self.idx = {}
        self.filepath = {'train': os.path.join(dataset, 'train.txt'),
                         'valid': os.path.join(dataset, 'valid.txt'),
                         'test':  os.path.join(dataset, 'test.txt')}  
        
    def set_seed(self, seed):
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        
    def make_dir(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
            
    def prepare_data(self):
        """
        Read train/valid/test data from input_dir and 
        convert data to features (input_ids, label_ids, attention_masks)
        """
        self.data = {}
        for mode in ['train', 'valid', 'test']:
            # read data and get sentences and labels
            with open(self.filepath[mode], 'r') as f:
                lines = f.readlines()
                sentences, idx, labels, sent, lab, id = [], [], [], [], [], []
                # label_map
                if mode == 'train':
                    tags = np.array(list(set(line.split()[-1].split('-')[-1] for line in lines if line.split())))
                    mlb = MultiLabelBinarizer(tags)
                    self.label2id = {t: i for i, t in enumerate(list(tags))}
                    self.id2label = {v: k for k, v in self.label2id.items()}
                    self.num_labels = len(self.label2id) if self.cls_type == 'multilabel' else 2
                # read data from CoNLL2003 format
                for line in lines:
                    if '-DOCSTART-' in line or '</s>' in line  or '<s>' in line or line.rstrip()=='':
                        if sent and lab:
                            sentences.append(sent)
                            idx.append(id)
                            lab = set(lab)
                            if self.cls_type == 'multilabel':
                                if len(lab) > 1: lab.remove('O')
                                labels.append(list(mlb.fit_transform([list(lab)]).flatten()))
                            else:
                                true_label = 1 if len(lab) > 1 else 0
                                labels.append([true_label])
                        if '-DOCSTART-' in line:
                            sent, lab, id = [line.split()[-2]], ['O'], ['-DOCSTART-']
                        else:
                            sent, lab, id = [], [], []
                    else:
                        sent.append(line.split()[0])
                        lab.append(line.split()[-1].split('-')[-1]) 
                        id.append(line.split()[-2])
            
            # tokenize the sentences
            input_ids, attention_masks = [], []
            for sent in sentences:
                encode = self.tokenizer.encode_plus(sent, pad_to_max_length=True, max_length=self.max_seq_length, 
                                               return_attention_mask=True)
                input_ids.append(encode['input_ids'])
                attention_masks.append(encode['attention_mask'])
            labels = torch.FloatTensor(labels)) if self.cls_type == 'multilabel' else torch.tensor(labels)
            self.data[mode] = TensorDataset(torch.tensor(input_ids), torch.tensor(attention_masks), lables)
            self.sentences[mode] = sentences
            self.idx[mode] = idx
            
        # Save training parameters
        print('\ndset: %s, batch_size: %d, lr: %4f, weight_decay: %4f, warmups: %d'%(\
                self.dataset, self.batch_size, self.lr, self.weight_decay, self.warmups)) 
        with open(os.path.join(self.output_dir, 'config.json'), 'w') as f:
            json.dump(args.__dict__, f)
    
    def trainer(self, parameterization, weight=None):
        # create output folder and tensorboard
        self.output_dir = 'output/{}/{}/{}'.format(parameterization['model'], dataset.split('/')[-1], 
                                                datetime.datetime.now().strftime('%m%d-%H%M%S'))
        self.model_dir = '{}/model'.format(self.output_dir)
        self.make_dir(self.output_dir)
        self.make_dir(self.model_dir)
        self.tsboard = {'train': SummaryWriter(os.path.join('tensorboard', parameterization['model'], 
                                                dataset.split('/')[-1]+'-train', 
                                                datetime.datetime.now().strftime('%m%d-%H%M%S'))),
                        'valid': SummaryWriter(os.path.join('tensorboard', parameterization['model'], 
                                                dataset.split('/')[-1]+'-valid', 
                                                datetime.datetime.now().strftime('%m%d-%H%M%S'))),
                        'test': SummaryWriter(logdir=os.path.join('tensorboard', parameterization['model'], 
                                                dataset.split('/')[-1]+'-test', 
                                                datetime.datetime.now().strftime('%m%d-%H%M%S')))}
        
        # load pretrained model and tokenizer
        self.model_class, self.tokenizer_class, pretrained_model = MODEL_CLASSES[parameterization['model']]
        self.pretrained_model = args.pretrained_model if args.pretrained_model else pretrained_model
        self.pretrained_tokenizer = args.pretrained_tokenizer if args.pretrained_tokenizer else self.pretrained_model
        self.tokenizer = self.tokenizer_class.from_pretrained(self.pretrained_tokenizer)  
 
        # update parameters from optimization experiemts or arguments
        self.batch_size = parameterization['batch']
        self.n_epochs = parameterization['n_epochs']
        self.lr = parameterization['lr']
        self.weight_decay = parameterization['decay']
        self.warmups = parameterization['warmups']  
        self.eps = parameterization['eps']  
        self.dataset = parameterization['dset'] 
        
        # get datasets
        self.prepare_data()
        train_data, valid_data, test_data = self.data['train'], self.data['valid'], self.data['test']
        train_dataloader = DataLoader(train_data, sampler=RandomSampler(train_data), batch_size=self.batch_size)
        valid_dataloader = DataLoader(valid_data, sampler=SequentialSampler(valid_data), batch_size=self.batch_size)
        test_dataloader = DataLoader(test_data, sampler=SequentialSampler(test_data), batch_size=self.batch_size)

        # load pretrained model and move to GPU
        model = self.model_class.from_pretrained(self.pretrained_model, num_labels=self.num_labels)
        model.to(self.device)

        # ensemble models
        if ensemble:
            ensemble_model_class, ensemble_tokenizer_class, ensemble_pretrained_model = MODEL_CLASSES[ensemble]  
            self.tokenizer = ensemble_tokenizer_class.from_pretrained(ensemble_pretrained_model)  
            train_data, valid_data, test_data = self.data['train'], self.data['valid'], self.data['test']
            train_dataloader_e = DataLoader(train_data, sampler=RandomSampler(train_data), batch_size=self.batch_size)
            valid_dataloader_e = DataLoader(valid_data, sampler=SequentialSampler(valid_data), batch_size=self.batch_size)
            test_dataloader_e = DataLoader(test_data, sampler=SequentialSampler(test_data), batch_size=self.batch_size)

            model_e = ensemble_model_class.from_pretrained(pretrained_model, num_labels=self.num_labels)
            model_e.to(self.device) 
        else:
            train_dataloader_e = train_dataloader
            valid_dataloader_e = valid_dataloader
            test_dataloader_e = test_dataloader
        
        # optimizer
        if self.fine_tuning:
            param_optimizer = list(model.named_parameters())
            no_decay = ['bias', 'gamma', 'beta']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                 'weight_decay_rate': self.weight_decay},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                 'weight_decay_rate': 0.0}]
        else:
            param_optimizer = list(model.classifier.named_parameters())
            optimizer_grouped_parameters = [{'params': [p for n, p in param_optimizer]}]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.lr, eps=self.eps)

        # learning rate scheduler
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=self.warmups, num_training_steps=len(train_dataloader) * self.n_epochs)

        global_steps, valid_steps, test_steps = 0, 0, 0
        best_valid_f1, best_valid_epoch, best_test_f1, best_test_epoch = 0, 0, 0, 0
        for epoch in range(self.n_epochs):
            # --------------- Training --------------- 
            model.train()
            train_loss = 0
            predictions , true_labels, pr_pred, pr_label = [], [], [], []
            start = time.time()
            for batch, batch_e in tqdm(zip(train_dataloader, train_dataloader_e), total=len(train_dataloader), desc='train'):
                # move batch to gpu
                b_input_ids, b_input_mask, b_labels = tuple(b.to(self.device) for b in batch)

                model.zero_grad()

                # forward pass
                if self.cls_type=='multilabel':
                    outputs = model(b_input_ids, attention_mask=b_input_mask)
                    loss = self.criterion(outputs[0], b_labels)
                    logits = outputs[1]
                else:
                    outputs = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
                    loss, logits = outputs

                # average two model's logits and calculate the loss for ensemble model
                if ensemble:
                    e_input_ids, e_input_mask, e_labels = tuple(b.to(self.device) for b in batch_e)
                    model_e.zero_grad()
                    if self.cls_type=='multilabel':
                        outputs_e = model_e(e_input_ids, attention_mask=e_input_mask)
                    else:
                        outputs_e = model_e(e_input_ids, attention_mask=e_input_mask, labels=e_labels)
                    logits = (outputs[1] + outputs_e[1])/2
                    loss = self.criterion(logits, b_labels)

                if self.cls_type=='multilabel':
                    logits = torch.sigmoid(logits).detach().cpu().numpy()
                    predictions.extend((logits>0.5)*1.)
                else:
                    logits = logits.detach().cpu().numpy()
                    predictions.extend(np.argmax(logits.tolist(),axis=1).tolist())
                label_ids = b_labels.detach().cpu().numpy()
                true_labels.extend(label_ids)
                
                # backward pass
                loss.backward()
                
                # train loss
                train_loss += loss
                global_steps += 1
                self.tsboard['train'].add_scalar('loss/loss', loss, global_steps)
                
                # avoid exploding gradients problem
                torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=self.max_grad_norm)
                
                # update parameters and learning rate
                optimizer.step()
                scheduler.step()
                self.tsboard['train'].add_scalar('loss/learning_rate', optimizer.param_groups[0]['lr'], global_steps)
                
            # train time
            train_time = time.time() - start
            # average train loss
            train_loss /= len(train_dataloader)
            
            # calculate metrics on each epoch
            train_metrics = self.metrics(predictions, true_labels, 'train', epoch)

            # --------------- Validation --------------- 
            model.eval()
            if ensemble: model_e.eval()
            valid_loss = 0
            predictions , true_labels, pr_pred, pr_label = [], [], [], []
            factor = len(train_dataloader)/len(valid_dataloader)
            start = time.time()
            for batch, batch_e in tqdm(zip(valid_dataloader, valid_dataloader_e), total=len(valid_dataloader), desc='valid'):
                # move batch to gpu
                b_input_ids, b_input_mask, b_labels = tuple(b.to(self.device) for b in batch)
                
                with torch.no_grad():
                    # Forward pass
                    if self.cls_type=='multilabel':
                        outputs = model(b_input_ids, attention_mask=b_input_mask)
                        loss = self.criterion(outputs[0], b_labels)
                        logits = outputs[1]
                    else:
                        outputs = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
                        loss, logits = outputs

                    if ensemble:
                        e_input_ids, e_input_mask, e_labels = tuple(b.to(self.device) for b in batch_e)
                        if self.cls_type=='multilabel':
                            outputs_e = model_e(e_input_ids, attention_mask=e_input_mask)
                        else:
                            outputs_e = model_e(e_input_ids, attention_mask=e_input_mask, labels=e_labels)
                        logits = (outputs[1] + outputs_e[1])/2
                        loss = self.criterion(logits, b_labels)

                if self.cls_type=='multilabel':
                    logits = torch.sigmoid(logits).detach().cpu().numpy()
                    predictions.extend((logits>0.5)*1.)
                else:
                    logits = logits.detach().cpu().numpy()
                    predictions.extend(np.argmax(logits.tolist(),axis=1).tolist())                       
                label_ids = b_labels.detach().cpu().numpy()
                true_labels.extend(label_ids)
                valid_loss += loss 
                self.tsboard['valid'].add_scalar('loss/loss', outputs[0].mean().item(), valid_steps*factor)
                valid_steps += 1                
            valid_loss /= valid_steps
            valid_time = time.time() - start
            
            # calculate metrics
            valid_metrics = self.metrics(predictions, true_labels, 'valid', epoch)
            out = self.process_output(predictions, true_labels, 'valid')
            
            # save best result
            if valid_metrics['all']['f1_macro'] > best_valid_f1:
                best_valid_epoch = epoch + 1
                best_valid_f1 = valid_metrics['all']['f1_macro']
                # save best prediction output
                with open(os.path.join(self.output_dir, 'prediction_valid.json'), 'w') as f:
                    json.dump(out, f)
                # save best model
                model.save_pretrained(self.model_dir)
                self.tokenizer.save_pretrained(self.model_dir)
                # save best result
                with open(os.path.join(self.output_dir, 'result_valid.json'), 'w') as f:
                    valid_metrics['time'] = valid_time
                    valid_metrics['best_epoch'] = best_valid_epoch  
                    json.dump(valid_metrics, f)

            print('[Epoch %d] train_loss: %.4f, val_loss: %.4f' % (
                      epoch+1, train_loss, valid_loss))
            print('Train - time:%.2f, acc: %.2f%%, Precision: %.2f%%, Recall: %.2f%%, F1: %.2f%%' % (
                      train_time, train_metrics['all']['accuracy'], train_metrics['all']['precision_macro'],
                      train_metrics['all']['recall_macro'], train_metrics['all']['f1_macro']))            
            print('Valid - time: %.2f, acc: %.2f%%, Precision: %.2f%%, Recall: %.2f%%, F1: %.2f%% (best epoch: %d)' % (
                      valid_time, valid_metrics['all']['accuracy'], valid_metrics['all']['precision_macro'],
                      valid_metrics['all']['recall_macro'], valid_metrics['all']['f1_macro'], best_valid_epoch))

            # --------------- Test --------------- 
            model.eval()
            if ensemble:
                model_e.eval()
            test_loss, test_steps = 0, 0
            predictions , true_labels, pr_pred, pr_label = [], [], [], []
            start = time.time()
            for batch, batch_e in zip(test_dataloader, test_dataloader_e):
                # move batch to gpu
                b_input_ids, b_input_mask, b_labels = tuple(t.to(self.device) for t in batch)

                with torch.no_grad():
                    # Forward pass
                    if self.cls_type=='multilabel':
                        outputs = model(b_input_ids, attention_mask=b_input_mask)
                        loss = self.criterion(outputs[0], b_labels)
                        logits = outputs[1]
                    else:
                        outputs = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
                        loss, logits = outputs

                    if ensemble:
                        e_input_ids, e_input_mask, e_labels = tuple(b.to(self.device) for b in batch_e)
                        if self.cls_type=='multilabel':
                            outputs_e = model_e(e_input_ids, attention_mask=e_input_mask)
                        else:
                            outputs_e = model_e(e_input_ids, attention_mask=e_input_mask, labels=e_labels)
                        logits = (outputs[1] + outputs_e[1])/2
                        loss = self.criterion(logits, b_labels)

                if self.cls_type=='multilabel':
                    logits = torch.sigmoid(logits).detach().cpu().numpy()
                    predictions.extend((logits>0.5)*1.)
                else:
                    logits = logits.detach().cpu().numpy()
                    predictions.extend(np.argmax(logits.tolist(),axis=1).tolist())  
                label_ids = b_labels.detach().cpu().numpy()
                true_labels.extend(label_ids)
                test_loss += outputs[0].mean().item()
                test_steps += 1
            test_loss /= test_steps
            test_time = time.time() - start
            
            # calculate metrics
            test_metrics = self.metrics(predictions, true_labels, 'test', epoch)
            out = self.process_output(predictions, true_labels, 'test')
            
            if test_metrics['all']['f1_macro'] > best_test_f1:
                best_test_epoch = epoch + 1
                best_test_f1 = test_metrics['all']['f1_macro']
                # save best prediction output
                try:
                    with open(os.path.join(self.output_dir, 'prediction_test.json'), 'w') as f:
                        json.dump(out, f)
                except:
                    pass
                # save best test result
                with open(os.path.join(self.output_dir, 'result_test.json'), 'w') as f:
                    test_metrics['time'] = test_time
                    test_metrics['best_epoch'] = best_test_epoch  
                    json.dump(test_metrics, f)
                    
            print('Test  - time: %.2f, acc: %.2f%%, Precision: %.2f%%, Recall: %.2f%%, F1: %.2f%% (best epoch: %d)' % (
                      test_time, test_metrics['all']['accuracy'], test_metrics['all']['precision_macro'], 
                      test_metrics['all']['recall_macro'], test_metrics['all']['f1_macro'],  best_test_epoch))
            
        for m in ['train', 'valid', 'test']:
            self.tsboard[mode].close()
        
        # return metrics for optimization experiments
        return {
            'f1': (valid_metrics['all']['f1_macro'], 0.0), 
            'precision': (valid_metrics['all']['precision_macro'], 0.0), 
            'recall': (valid_metrics['all']['recall_macro'], 0.0),   
            'accuracy': (valid_metrics['all']['accuracy_macro'], 0.0),
        }
    
    def criterion(self, prediction, true_labels):
        if self.cls_type=='multilabel':
            loss_fn = BCEWithLogitsLoss()
        else:
            loss_fn = CrossEntropyLoss()
        return loss_fn(input=prediction, target=true_labels)
   
    def metrics(self, predictions, true_labels, mode, epoch):
        """
        calculate metrics and save to tensorboard
        """
        def calculate_metrics(predictions, true_labels):
            ppv_macro, sen_macro, f1_macro, _ = precision_recall_fscore_support(true_labels, predictions, 
                                                                                average='macro', warn_for=tuple())
            ppv_micro, sen_micro, f1_micro, _ = precision_recall_fscore_support(true_labels, predictions, 
                                                                                average='macro', warn_for=tuple())
            return {'f1_macro':f1_macro*100, 'precision_macro':ppv_macro*100, 'recall_macro':sen_macro*100, 
                    'f1_micro':f1_micro*100, 'precision_micro':ppv_micro*100, 'recall_micro':sen_micro*100, 
                    'accuracy': accuracy_score(true_labels, predictions)*100, 
                    'hamming_loss': hamming_loss(true_labels, predictions),
                    'LRAP': label_ranking_average_precision_score(true_labels, predictions)*100}
        metric = {}
        predictions, true_labels = np.array(predictions), np.array(true_labels)
        
        if self.cls_type == 'multilabel':
            # get metrics on all labels
            metric['all'] = calculate_metrics(predictions, true_labels)
            for m in ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro', 'hamming_loss', 'LRAP']:
                self.tsboard[mode].add_scalar('metrics/{}'.format(m), metric['all'][m], epoch)

            # get metrics on single label
            metric['individual'] = {}
            for id in self.id2label.keys():
                if any(t>0 for t in true_labels[:, id]):
                    label = self.id2label[id]
                    if all(p==0 for p in predictions[:, id]):
                        ppv, sen, f1, acc = 0, 0, 0, 0
                    else:
                        ppv, sen, f1, _ = precision_recall_fscore_support(true_labels[:, id], predictions[:, id], 
                                                                          average='binary', warn_for=tuple())
                        acc = accuracy_score(true_labels[:, id], predictions[:, id])
                    metric['individual'][label] = {'f1': f1*100, 'precision':ppv*100, 'recall':sen*100, 'accuracy':  acc*100}
        else:
            ppv, sen, f1, _ = precision_recall_fscore_support(true_labels, predictions, average='binary', warn_for=tuple())
            metric['all'] = {'f1':f1*100, 'precision':ppv*100, 'recall':sen*100, 
                             'accuracy': accuracy_score(true_labels, predictions)*100}
            for m in ['accuracy', 'f1', 'precision', 'recall']:
                self.tsboard[mode].add_scalar('metrics/{}'.format(m), metric['all'][m], epoch)
            
        return metric
   
    def process_output(self, predictions, true_labels, mode):
        """
        convert ids to original labels and create formated output prediction
        """
        output, out = {}, {}
        for prediction, true_label, sent, id in zip(predictions, true_labels, self.sentences[mode], self.idx[mode]):
            if '-DOCSTART-' == id[0]:
                doc_id = sent[0]
                if out:
                    output[doc_id] = out
                out = {}
            else:
                if self.cls_type == 'multilabel':
                    pred = [self.id2label[i] for i, pred in enumerate(prediction) if pred==1]
                    gt = [self.id2label[i] for i, label in enumerate(true_label) if label==1]
                    out[id[0]] = {'sentence':' '.join(sent), 'pred':pred, 'true':gt}
                else:
                    out[id[0]] = {'sentence':' '.join(sent), 'pred':prediction, 'true':true_label[0]}
        else:

        return output

if __name__ == "__main__":
    # if not running optimization experiments, get the parameters from arguments
    parameterization = {'lr': args.lr, 'decay': args.decay, 'warmups': args.warmups, 'eps': args.eps,
                        'batch': args.batch, 'n_epochs': args.n_epochs, 
                        'dset':args.dset, 'model':args.model_class}
    ner = TransformerNER()
    ner.trainer(parameterization)
    