## Lab 5: hypertune capabilities ##
 
# Understand goal #
In this lab we are going to tune the hyperparameters of a random forest classifier. We do this in order to find the best model to fit our data and that will give the highest proobabailities. In Azure ML we can run a special run that is optimized for hyperparamter tuning. 

1. Read the documentation Hyperparameter tuning

# Define the Hyper paramters
Before we start creating the hyper parameter run, we need to know and understand the parameters that we can tune for the random forest classifier

2. Search for sklearn randomclassifier and identify the parameters that we can tune. Write them down below

...

# Alter the hypertrain scipt #
The hypertrain script is similar to the train script, but instead of running 15 different algortims, we are only going to run the RandomForestClassifier. As we have seen in the previous step, the RandomForestClassifier has a lot of parameters that we can tune. In this example, we will only tune max_depth, n_estimators, criterion, min_samples_split. (Note: if you want to add more more hyperparameters you can do that in the similair way as we are adding these paramters.)

3. Define the parameters as input arguments in the OptionParser(), define the input type and set the default value to the default provided in the documentation. 

op.add_option("--max_depth",
              type=int, default=10)
op.add_option("--n_estimators",
              type=int, default=100)
op.add_option("--criterion",
              type=str,
              default='gini')
op.add_option("--min_samples_split",
              type=int,
              default=2)

4. Create a dict of hyperparameters from the input flags.

hyperparameters = {
    "max_depth": opts.max_depth,
    "n_estimators": opts.n_estimators,
    "criterion": opts.criterion,
    "min_samples_split": opts.min_samples_split
}

5. Select the training hyperparameters as imput variables

max_depth = hyperparameters["max_depth"]
n_estimators = hyperparameters["n_estimators"]
criterion = hyperparameters["criterion"]
min_samples_split = hyperparameters["min_samples_split"]

6. Add the hyperparameters as imput options to RandomForestClassifier()

7. Add the log metrics to the script

8. Save the .pkl file 

# Understand differences in run configuration
The run configuration for the hypertuning is slightly different from the standard run configuration. Azure ML has a special package azureml.train.hyperdrive for creating a hyperparamter tuning run. From this package, we are going to make use of the HyperDriveConfig to create the config file. 

9. Create the estimator. (Note: the estimator for the hypertrain run is the same as for a normal run, but we are now running the script hypertrain.py)

10. Define the parameter sampling space and the search algoritm
There are primaly 3 different ways to perform parameter searching:  Random, Sweeping and....
In this example we will make use of the RandomParameterSampling.
For every hyperparamter we can tune, we need to specify the search space. This search space can be conitious and defined by a uniform or normal distribution, or can be dircrete and defined by a choice function.

11. De fine the hyperdrive run configuration
Make sure to use the paramter sampling as an imput of the config file and 

# Submit run on AML compute

# View results in the portal
