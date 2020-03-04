## Lab 2: running experiments ##

# Understand the non-azure / open source ml model code #
We first start with understanding the training script. The training script is an open source ML model code from https://scikit-learn.org/stable/auto_examples/text/plot_document_classification_20newsgroups.html. This is an example showing how scikit-learn can be used to classify documents by topics using a bag-of-words approach. This example uses a scipy.sparse matrix to store the features and demonstrates various classifiers that can efficiently handle sparse matrices. The dataset used in this example is the 20 newsgroups dataset. It will be automatically downloaded, then cached. The newsgroup datasets contains text documents that are classified into 20 categories.

1. Open the train.py document to inspect the code.
The first step in the code is to load the dataset from the 20 newsgroup dataset. In this example we are only going to use a subset of the categories. Please state the catogories we are going to use:

...

The second step is to extract the features from the text. We do this with a sparse vecorizer. We also clean the data a bit. What is the operation that we do on the data to clean the text?

...

After we have reshaped our data and made sure the feature names are in the right place, we are going to define the algorithm to fit the model. This step is defining the benchmark. We fit the data and make predictions on the test set. To validate out model we need a metric to score the model. There are many metrics we can use. Define in the code the metric that you want to use to validate your model and make sure the print statement will output your metric. (Note: you can define multiple scores if you want. If so, make sure to return these scores.)

...


The last step is to define tha algoritms that we want to fit over our data. In this example we are using 15 classification algoritms to fit the data. We keep track of the metrics of all olgoritms, so we can compare the performance and pick the model. Look at the code and whrite down the different algoritms that we are going to test.

...

# Run the training locally #
We are now going to train the scripts locally. The script will return the diffetent metrics for all algoritms. Inspect the metrics that you specified. Wich algoritms performs best?

...

#  Run the code via Azure ML #
We are now going to run our code via Azure ML. 
We are going to make use of child runs. The expiriment will perform a parent run that is going to execute train.py. Within train.py we are going to create child runs. For every of the 15 algoritms that we have we want to create a sub run and log the metrics seprately. Whihin the child run we are going to log the performane and the model .pkl files. This way we can easily track and compare our experiment in Azure ML.

1. Read Experiment Tracking documentation

2. Read How to Mange a Run documentation

3. Refactor the code to capture run metrics in train.py
    1. Get the run context
    2. Create a child run
    3. Log the metric in the child trun
    4. upload the .pkl file to the output folder of child run
    5. close the child run

4. ALter the train_submit.py file

    1. Load Azure ML workspace form config file
    2. Create an extimator to define the run configuration
    3. Define the ML experiment
    4. Submit the experiment

5. Go to the portal to inspect the run history

