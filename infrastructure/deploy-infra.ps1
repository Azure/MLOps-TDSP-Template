# Deployment script for machine learning resources
# Run locally to debug changes in the resource configuration
# Use `deploy-infrastructure.yml` for deployment automation.

# Prompt users for resource group and location
$resourceGroupName = Read-Host -Prompt "Provide a resource group name"
$location = Read-Host -Prompt "Provide a DC location"

# Create a Resource Group
New-AzResourceGroup -Name $resourceGroupName -Location $location

# Deploy Storage Account
New-AzResourceGroupDeployment -ResourceGroupName $resourceGroupName `
  -TemplateFile arm-templates/storage/template.json `
  -TemplateParameterFile arm-templates/storage/parameters.json

# Deploy Container Registry
New-AzResourceGroupDeployment -ResourceGroupName $resourceGroupName `
  -TemplateFile arm-templates/containerregistry/template.json `
  -TemplateParameterFile arm-templates/containerregistry/parameters.json

# Deploy Application Insights
New-AzResourceGroupDeployment -ResourceGroupName $resourceGroupName `
  -TemplateFile arm-templates/appinsights/template.json `
  -TemplateParameterFile arm-templates/appinsights/parameters.json

# Deploy Key Vault
New-AzResourceGroupDeployment -ResourceGroupName $resourceGroupName `
  -TemplateFile arm-templates/keyvault/template.json `
  -TemplateParameterFile arm-templates/keyvault/parameters.json

# Deploy Workspace
New-AzResourceGroupDeployment -ResourceGroupName $resourceGroupName `
  -TemplateFile arm-templates/mlworkspace/template.json `
  -TemplateParameterFile arm-templates/mlworkspace/parameters.json

# Deploy Compute
New-AzResourceGroupDeployment -ResourceGroupName $resourceGroupName `
-TemplateFile arm-templates/mlcompute/template.json `
-TemplateParameterFile arm-templates/mlcompute/parameters.json
