"""
Training submitter

Facilitates remote training execution through the Azure ML service.
Compute and data references defined according to run configuration.
"""
import os
import shutil
from azureml.core import Workspace, Experiment, Run
from azureml.core import RunConfiguration, ScriptRunConfig
from azureml.core.authentication import AzureCliAuthentication
from azureml.train.estimator import Estimator

# Define run configuration (compute/environment/data references/..)
#run_config_name = 'dsvmcluster'  # local/docker/dsvmcluster
#exp_name = "TemplateExperiment"
#curr_dir = os.path.dirname(os.path.realpath(__file__))
#output_dir = 'outputs'

# load workspace config, load default datastore.
ws = Workspace(
    "e0eeddf8-2d02-4a01-9786-92bb0e0cb692", "azure-ml-rg",
    "azure-machine-learning-ws",
    auth=None, _location=None, _disable_service_check=False,
    _workspace_id=None, sku='basic'
)

experiment = Experiment(ws, "testexperimentsubmittrain")

# load run config
#run_config = RunConfiguration.load(
#    path=os.path.join(curr_dir, '../../', 'aml_config'),
#    name=run_config_name
#)
cluster_name = 'text-cluster'
compute_target = ws.compute_targets[cluster_name]
# Submit experiment run
# datastores will be attached as specified in the run configuration
est = Estimator(
    entry_script='train.py',
    source_directory=os.path.dirname(os.path.realpath(__file__)),
    compute_target=compute_target,
    pip_packages=[
        'numpy==1.15.4',
        'pandas==0.23.4',
        'scikit-learn==0.20.1',
        'scipy==1.0.0',
        'matplotlib==3.0.2',
        'utils==0.9.0'
    ]
)

print('Submitting experiment.. if compute is idle, this may take some time')
run = experiment.submit(est)
run.submit_child(est)

# wait for run completion to show logs
run.wait_for_completion(show_output=True)
print('View run results in portal:', run.get_portal_url(), '\n')

# ignore Azure ML service bug: delete redundant folders/files
#shutil.rmtree(os.path.join(curr_dir, 'assets'))
#shutil.rmtree(os.path.join(curr_dir, 'aml_config'))
#os.remove(os.path.join(curr_dir, '.amlignore'))


minimum_accuracy_runid = None
minimum_accuracy = None

for run in experiment.get_runs():
    run_metrics = run.get_metrics()
    run_details = run.get_details()
    # each logged metric becomes a key in this returned dict
    run_accuracy = run_metrics["accuracy"]
    run_id = run_details["runId"]

    if minimum_accuracy is None:
        minimum_accuracy = run_accuracy
        minimum_accuracy_runid = run_id
    else:
        if run_accuracy > minimum_accuracy:

            minimum_accuracy = run_accuracy
            minimum_accuracy_runid = run_id

print("Best run_id: " + minimum_accuracy_runid)
print("Best run_id accuracy: " + str(minimum_accuracy))

best_run = Run(experiment=experiment, run_id=minimum_accuracy_runid)
print(best_run.get_file_names())

# register model
model = best_run.register_model(
    model_name='bestalgorithm',
    model_path=os.path.join('outputs', str(best_run.get_file_names())+'.pkl')
)
print(model.name, model.id, model.version, sep='\t')

