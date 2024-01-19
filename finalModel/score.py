import json
import numpy as np
import os
import pickle

def init():
    global model
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'best_rf_model.pkl')
    with open(model_path, 'rb') as file:
        model = pickle.load(file)

def run(raw_data):
    try:
        data = json.loads(raw_data)
        if isinstance(data, list):
            # If a list of user data is provided
            data = np.array([list(user.values()) for user in data])
        else:
            # If a single user's data is provided
            data = np.array([list(data.values())])

        result = model.predict(data)
        return result.tolist()
    except Exception as e:
        return str(e)
