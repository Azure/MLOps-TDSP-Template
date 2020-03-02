# Pre-processes SKLearn sample data 
# Ingest the data into an Azure ML Datastore for training
import pandas as pd
import time
import os
from sklearn.datasets import fetch_20newsgroups
from azureml.core import Workspace, Datastore
from azureml.core.authentication import AzureCliAuthentication

# Define newsgroup categories to be downloaded to generate sample dataset
# @TODO add additional newsgroups
categories = [
    'alt.atheism',
    'talk.religion.misc',
    'comp.graphics',
    'sci.space',
]

print("Loading 20 newsgroups dataset for categories:")
print(categories if categories else "all")

for data_split in ['train', 'test']:
    # retrieve newsgroup data
    newsgroupdata = fetch_20newsgroups(
        subset=data_split,
        categories=categories,
        shuffle=True,
        random_state=42
    )

    # construct pandas data frame from loaded sklearn newsgroup data
    df = pd.DataFrame({
        'text': newsgroupdata.data,
        'target': newsgroupdata.target
    })

    print('data loaded')

    # pre-process:
    # remove line breaks
    # replace target index by newsgroup name
    target_names = newsgroupdata.target_names
    df.target = df.target.apply(lambda x: target_names[x])
    df.text = df.text.replace('\n', ' ', regex=True)

    print(df.head(5))

    # write to csv
    df.to_csv(os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        'tmp',
        data_split,
        '{}.csv'.format(int(time.time()))  # unique file name
    ), index=False, encoding="utf-8", line_terminator='\n')


datastore_name = 'workspaceblobstore'

# get existing ML workspace
workspace = Workspace.from_config(auth=AzureCliAuthentication())

# retrieve an existing datastore in the workspace by name
datastore = Datastore.get(workspace, datastore_name)

# upload files
datastore.upload(
    src_dir=os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        'tmp'
    ),
    target_path=None,
    overwrite=True,
    show_progress=True
)
