"""
Real Time Scoring Service
@TODO
"""
import json
import time
import numpy as np
from azureml.core.model import Model
from sklearn.externals import joblib


def init():
    """
    Load model and other dependencies for inferencing
    """
    global model
    # Print statement for appinsights custom traces:
    print("model initialized" + time.strftime("%H:%M:%S"))

    # note here "sklearn_regression_model.pkl" is the name of the
    # model registered under the workspace this call should return
    # the path to the model.pkl file on the local disk.
    model_path = Model.get_model_path(model_name='model.pkl')

    # deserialize the model file back into a sklearn model
    model = joblib.load(model_path)


def run(raw_data):
    """
    Score new data against model
    """
    try:
        data = json.loads(raw_data)['data']
        data = np.array(data)
        result = model.predict(data)

        # you can return any datatype as long as it is JSON-serializable
        return result.tolist()
    except Exception as e:
        error = str(e)
        print(error + time.strftime("%H:%M:%S"))
        return error
