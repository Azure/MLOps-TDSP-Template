import os
from azureml.core import Workspace
from azureml.core.authentication import AzureCliAuthentication

# Create the workspace using the specified parameters
ws = Workspace.create(
    name='azuremlworkshopws',
    subscription_id='cf4e1704-b4bc-4554-bcd7-309394f2ee56',
    resource_group='azuremlworkshoprgp',
    location='westeurope',
    create_resource_group=True,
    sku='basic',
    exist_ok=True,
    auth=AzureCliAuthentication()
)

print(ws.get_details())

# write the details of the workspace to a configuration file in the project root
ws.write_config(
    path=os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        '../'
    )
)
