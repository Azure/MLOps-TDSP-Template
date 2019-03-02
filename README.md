# AI DevOps Quickstart Template #

This repo provides a quickstarter template as a fork on TDSP (https://github.com/Azure/Azure-TDSP-ProjectTemplate), extending the template with a suggested structure for operationalization using Azure. The current code base includes ARM templates as IaC for resource deployment, template build and release pipelines to enable ML model CI/CD, template code for working with Azure ML.

## How to get started ##

* Clone this repo
* Make sure you have an Azure Subscription set up.
* Make sure you have an Azure DevOps instance set up.
* Import the build and release definitions ('Code'>'Operationalization'>'build_and_release') into Azure DevOps pipelines.
* Update the build and release definitions to use your credentials i.e. Azure subscription.
* Create an initial commit.
* If everything is set up correctly, Azure DevOps will provision your Azure Resources as triggered by the CI.
* Use the Azure CLI ML Extension (`az ml project attach` command) or Azure ML SDK to configure your local workspace to use the created Azure ML workspace.
* Run `Code/Modeling/train_submit` to run your first AzureML experiment on remote compute.
