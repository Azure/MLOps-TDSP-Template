"""
Training submitter

Facilitates remote training execution through the Azure ML service.
Compute and data references defined according to run configuration.
"""
import os
import shutil
from azureml.core import Workspace, Experiment
from azureml.core import RunConfiguration, ScriptRunConfig
from azureml.core.authentication import AzureCliAuthentication

# Define run configuration (compute/environment/data references/..)
run_config_name = 'dsvmcluster'  # local/docker/dsvmcluster
exp_name = "TemplateExperiment"
curr_dir = os.path.dirname(os.path.realpath(__file__))
output_dir = 'outputs'

# load workspace config, load default datastore.
ws = Workspace.from_config(auth=AzureCliAuthentication())
experiment = Experiment(workspace=ws, name=exp_name)
default_ds = ws.get_default_datastore()

# load run config
run_config = RunConfiguration.load(
    path=os.path.join(curr_dir, '../../', 'aml_config'),
    name=run_config_name
)

# Submit experiment run
# datastores will be attached as specified in the run configuration
est = ScriptRunConfig(
    script='train.py',
    source_directory=curr_dir,
    run_config=run_config,
    arguments=[
        '--data-dir', str(default_ds.as_mount()),
        '--output-dir', output_dir
    ]
)

print('Submitting experiment.. if compute is idle, this may take some time')
run = experiment.submit(est)

# wait for run completion to show logs
run.wait_for_completion(show_output=True)
print('View run results in portal:', run.get_portal_url(), '\n')

# ignore Azure ML service bug: delete redundant folders/files
shutil.rmtree(os.path.join(curr_dir, 'assets'))
shutil.rmtree(os.path.join(curr_dir, 'aml_config'))
os.remove(os.path.join(curr_dir, '.amlignore'))
