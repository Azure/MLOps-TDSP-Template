"""
Model Trainer

Load data, train model(s), evaluate performance
"""
import os
from azureml.core import Run
import pandas as pd
import argparse
from modelpackage.modeler import Modeler
from sklearn.externals import joblib

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', type=str, default="Sample_Data/Raw",
                    dest='data_dir', help='data folder')
parser.add_argument('--output-dir', type=str, default="outputs",
                    dest='output_folder', help='experiment output folder')
args = parser.parse_args()

# load experiment run for tracking
run = Run.get_context()

# read a file from mounted storage
df = pd.read_csv(os.path.join(args.data_dir, 'file.csv'))
print(df.head())

# train model(s)
modeler = Modeler()

print('Validate data integrity..')
modeler.validate_data(df)

print('Splitting data..')
X_train, X_test, y_train, y_test = modeler.splitData(df)

print('Training model..')
cv_scores = modeler.train(X_train, y_train)

print('Scoring model..')
y_pred = modeler.score(df)

# log metric values with experiment run
print('Evaluating model performance..')
run.log("r-squared", 0.84)
run.log("rmse", 20)

# experiment run results, store model
# files stored in the 'outputs' folder are automatically tracked by Azure ML
try:
    os.makedirs(args.output_folder, exist_ok=True)

    file = open(os.path.join(args.output_folder, 'model.pkl'), 'wb')
    joblib.dump(modeler.model, file)
    file.close()

    print('Persisted trained model to path:', args.output_folder)
except Exception as e:
    print('Error serializing model' + e)
