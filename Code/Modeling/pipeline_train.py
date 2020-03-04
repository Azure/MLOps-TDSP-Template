"""
Model Training Pipeline

Note: ML Pipelines are executed on registered compute resources.
Run configurations hence cannot reference local compute.
"""
import os
from azureml.core import Experiment, Workspace
from azureml.pipeline.core import Pipeline, PipelineData
from azureml.pipeline.steps import PythonScriptStep
from azureml.core import RunConfiguration
from azureml.core.authentication import AzureCliAuthentication
from azureml.data.data_reference import DataReference

# Define run configuration (compute/environment/data references/..)
run_config_name = 'dsvmcluster'
exp_name = "Training_Pipeline"
curr_dir = os.path.dirname(os.path.realpath(__file__))
output_dir = 'outputs'
output_dir_local = os.path.join(curr_dir, '../../../', 'outputs')

# Pipeline parameters
run_experiment = True
register_model = False
publish_pipeline = False

# load workspace config, load default datastore.
ws = Workspace.from_config(auth=AzureCliAuthentication())
default_ds = ws.get_default_datastore()

# load run config
run_config = RunConfiguration.load(
    path=os.path.join(curr_dir, '../../../', 'aml_config'),
    name=run_config_name
)

# define training pipeline with one AMLCompute step
trainStep = PythonScriptStep(
    script_name="train.py",
    name="Model Training",
    arguments=[
        '--data-dir', str(default_ds.as_mount()),
        '--output-dir', output_dir
    ],
    inputs=[
        DataReference(
            datastore=default_ds,
            mode="mount"
        )
    ],
    outputs=[
        PipelineData(
            name="model",
            datastore=default_ds,
            output_path_on_compute="training"
        )
    ],
    compute_target=run_config.target,
    runconfig=run_config,
    source_directory=os.path.join(curr_dir, '../')
)

training_pipeline = Pipeline(workspace=ws, steps=[trainStep])
training_pipeline.validate()
print("Pipeline validation complete")

# Submit pipeline run
pipeline_run = Experiment(ws, exp_name).submit(training_pipeline)
pipeline_run.wait_for_completion()
