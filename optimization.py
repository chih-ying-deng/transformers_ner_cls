import os
import json
import numpy as np
import argparse
import datetime
from ax import ParameterType, ChoiceParameter, RangeParameter, FixedParameter, SearchSpace, SimpleExperiment, modelbridge, models
from ax.plot.contour import interact_contour, plot_contour
from ax.plot.diagnostic import interact_cross_validation
from ax.plot.scatter import interact_fitted, plot_objective_vs_constraints
from ax.plot.slice import plot_slice
from ax.modelbridge.cross_validation import cross_validate
from ax.plot.trace import optimization_trace_single_method
from plotly.offline import plot
from ner import TransformerNER
from classification import TransformerCLS

parser = argparse.ArgumentParser()
parser.add_argument('--dset', type=str, default='data/symptom', help='name of dataset')
parser.add_argument('--lr', type=float, nargs=2, default=[1e-6,1e-4], help='search space of learning rate')
parser.add_argument('--decay', type=float, nargs=2, default=[0,0.05], help='search space of weight decay rate')
parser.add_argument('--warmups', type=int, nargs=2, default=[0,3000], help='search space of warmups')
parser.add_argument('--eps', type=float, nargs=2, default=[1e-9,1e-7], help='search space of eps for Adam')
parser.add_argument('--init_trials', type=int, default=5, help='initialization trials')
parser.add_argument('--opt_trials', type=int, default=25, help='optimization trials')
parser.add_argument('--n_epochs', type=int, default=30, help='number of epochs')
parser.add_argument('--model', type=str, default='bert', choices=['bert', 'xlnet', 'roberta', 'xlm-roberta', 
                             'camembert', 'distilbert', 'electra'], help='model type')
parser.add_argument('--type', type=str, default='ner', choices=['ner', 'cls'], help='NER or classification')

args = parser.parse_args()
dset = args.dset
lr = args.lr
decay = args.decay
warmups = args.warmups
eps = args.eps
init_trials = args.init_trials
opt_trials = args.opt_trials
n_epochs = args.n_epochs
type = args.type

# Search space
transformer_search_space = SearchSpace(parameters=[
    RangeParameter(
        name='lr', parameter_type=ParameterType.FLOAT, 
        lower=min(lr), upper=max(lr), log_scale=False),
    RangeParameter(
        name='decay', parameter_type=ParameterType.FLOAT, 
        lower=min(decay), upper=max(decay), log_scale=False),
    RangeParameter(
        name='warmups', parameter_type=ParameterType.FLOAT, 
        lower=min(warmups), upper=max(warmups), log_scale=False), 
    RangeParameter(
        name='eps', parameter_type=ParameterType.FLOAT, 
        lower=min(eps), upper=max(eps), log_scale=False), 
    ChoiceParameter(
        name='batch', parameter_type=ParameterType.INT, 
        values=[16, 32, 64]), 
    FixedParameter(name="n_epochs", parameter_type=ParameterType.INT, value=n_epochs),
    FixedParameter(name="dset", parameter_type=ParameterType.STRING, value=dset),
])
dset = dset.split('/')[-1]
if type == 'ner':
    transformer = TransformerNER()
else:
    transformer = TransformerCLS()

# Create Experiment
exp = SimpleExperiment(
    name = 'transformer',
    search_space = transformer_search_space,
    evaluation_function = transformer.trainer,
    objective_name = 'f1',
)

# Run the optimization and fit a GP on all data
sobol = modelbridge.get_sobol(search_space=exp.search_space)
print(f"\nRunning Sobol initialization trials...\n{'='*40}\n")
for _ in range(init_trials):
    exp.new_trial(generator_run=sobol.gen(1))

for i in range(opt_trials):
    print(f"\nRunning GP+EI optimization trial {i+1}/{opt_trials}...\n{'='*40}\n")
    gpei = modelbridge.get_GPEI(experiment=exp, data=exp.eval())
    exp.new_trial(generator_run=gpei.gen(1))
    
    # save every 5 trials
    if (i+1)%5==0:
        output_dir = os.path.join('Ax_output', dset, datetime.datetime.now().strftime('%m%d-%H%M%S'))
        os.makedirs(output_dir)
        
        # Save all experiment parameters 
        df = exp.eval().df
        df.to_csv(os.path.join(output_dir, 'exp_eval.csv'), index=False)
        
        # Save best parameter
        best_arm_name = df.arm_name[df['mean'] == df['mean'].max()].values[0]
        exp_arm = {k:v.parameters for k, v in exp.arms_by_name.items()}
        exp_arm['best'] = best_arm_name
        print('Best arm:\n', str(exp.arms_by_name[best_arm_name]))
        with open(os.path.join(output_dir, 'exp_arm.json'), 'w') as f: 
            json.dump(exp_arm, f)

        # Contour Plot
        os.makedirs(os.path.join(output_dir, 'contour_plot'))
        for metric in ['f1', 'precision', 'recall', 'accuracy']:
            contour_plot = interact_contour(model=gpei, metric_name=metric)
            plot(contour_plot.data, filename=os.path.join(output_dir, 'contour_plot', '{}.html'.format(metric)))

        # Tradeoff Plot
        tradeoff_plot = plot_objective_vs_constraints(gpei, 'f1', rel=False)
        plot(tradeoff_plot.data, filename=os.path.join(output_dir, 'tradeoff_plot.html'))

        # Slice Plot
        # show the metric outcome as a function of one parameter while fixing the others
        os.makedirs(os.path.join(output_dir, 'slice_plot'))
        for param in ["lr", "decay",  "warmups", "eps"]:
            slice_plot = plot_slice(gpei, param, "f1")
            plot(slice_plot.data, filename=os.path.join(output_dir, 'slice_plot', '{}.html'.format(param)))

        # Tile Plot
        # the effect of each arm
        tile_plot = interact_fitted(gpei, rel=False)
        plot(tile_plot.data, filename=os.path.join(output_dir, 'tile_plot.html'))

        # Cross Validation plot
        # splits the model's train data into train/test folds and makes out-of-sample predictions on the test folds.
        cv_results = cross_validate(gpei)
        cv_plot = interact_cross_validation(cv_results)
        plot(cv_plot.data, filename=os.path.join(output_dir, 'cv_plot.html'))
