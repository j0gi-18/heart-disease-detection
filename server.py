from fastapi import FastAPI
import joblib
import numpy as np

model = joblib.load("app/knn.joblib")

class_names = np.array(['Healthy', 'Unhealthy'])

# FastAPI Instance
app = FastAPI()

# create a get method
@app.get('/')
def reed_root():
    return {'message': 'Welcome to Heart disease predictor'}

# create a post method
@app.post('/predict')
def predict(data: dict):
    features = np.array(data['features']).reshape(1, -1)
    prediction = model.predict(features)
    class_name = class_names[prediction][0]
    return {'predicted_class': class_name}