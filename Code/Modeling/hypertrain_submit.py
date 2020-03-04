from azureml.train.hyperdrive import (
    RandomParameterSampling,
    HyperDriveConfig, PrimaryMetricGoal)
from azureml.core import Workspace, Experiment
from azureml.train.estimator import Estimator
import pandas as pd
from azureml.core.compute import ComputeTarget, AmlCompute
import os
from random import choice

ws = Workspace(
    "e0eeddf8-2d02-4a01-9786-92bb0e0cb692", "azure-ml-rg",
    "azure-machine-learning-ws",
    auth=None, _location=None, _disable_service_check=False,
    _workspace_id=None, sku='basic'
)

cluster_name = 'hypetuning'
provisioning_config = AmlCompute.provisioning_configuration(
        vm_size='Standard_D4_v2',
        # vm_priority = 'lowpriority', # optional
        max_nodes=16)

if cluster_name in ws.compute_targets:
    compute_target = ws.compute_targets[cluster_name]
    if type(compute_target) is not AmlCompute:
        raise Exception('Compute target {} is not an AML cluster.'
                        .format(cluster_name))
    print('Using pre-existing AML cluster {}'.format(cluster_name))
else:
    # Create the cluster
    compute_target = ComputeTarget.create(ws, cluster_name,
                                          provisioning_config)

    compute_target.wait_for_completion(show_output=True,
                                       min_node_count=None,
                                       timeout_in_minutes=20)


estimator = Estimator(
    source_directory=os.path.dirname(os.path.realpath(__file__)),
    compute_target=compute_target,
    entry_script='hypertrain.py',
    pip_packages=[
        'numpy==1.15.4',
        'pandas==0.23.4',
        'scikit-learn==0.20.1',
        'scipy==1.0.0',
        'matplotlib==3.0.2',
        'utils==0.9.0'
    ]
)

param_sampling = RandomParameterSampling({
    "max_depth": choice([100, 50, 20, 10]),
    "n_estimators": choice([50, 150, 200, 250]),
    "criterion": choice(['gini', 'entropy']),
    "min_samples_split": choice([2, 3, 4, 5])
    }
)

hyperdrive_run_config = HyperDriveConfig(
    estimator=estimator,
    hyperparameter_sampling=param_sampling,
    policy=None,
    primary_metric_name="accuracy",
    primary_metric_goal=PrimaryMetricGoal.MAXIMIZE,
    max_total_runs=2,
    max_concurrent_runs=None
)

experiment = Experiment(ws, "testhypertuning")

hyperdrive_run = experiment.submit(hyperdrive_run_config)
hyperdrive_run.wait_for_completion()
# run = experiment.submit(estimator)

best_run = hyperdrive_run.get_best_run_by_primary_metric()
best_run_metrics = best_run.get_metrics()
hyperdrive_run.log("Accuracy", best_run_metrics['accuracy'])


parameter_values = best_run.get_details()['runDefinition']['arguments']
best_parameters = dict(zip(parameter_values[::2], parameter_values[1::2]))
pd.Series(best_parameters, name='Value').to_frame()
model_parameters = best_parameters.copy()
pd.Series(model_parameters, name='Value').to_frame()
print(model_parameters)


# model_parameters['--data-folder'] = ds.as_mount()
exp = Experiment(ws, "finalmodel")

model_est = Estimator(source_directory=os.path.dirname(os.path.realpath(__file__)),
                      entry_script='hypertrain.py',
                      script_params=model_parameters,
                      compute_target=compute_target,
                      pip_packages=[
                      'numpy==1.15.4',
                      'pandas==0.23.4',
                      'scikit-learn==0.20.1',
                      'scipy==1.0.0',
                      'matplotlib==3.0.2',
                      'utils==0.9.0'])


model_run = exp.submit(model_est)
model_run_status = model_run.wait_for_completion(wait_post_processing=True)
model = model_run.register_model(model_name='model',
                                 model_path=os.path.join('outputs', 'model.pkl'))
