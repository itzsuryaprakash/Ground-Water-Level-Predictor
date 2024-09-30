from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import joblib
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"], 
)


model = tf.keras.models.load_model('lstm_model.h5')
scaler_X = joblib.load('scaler_X.pkl')
scaler_y = joblib.load('scaler_y.pkl')


class PredictionRequest(BaseModel):
    latitude: float
    longitude: float

def create_dataset(X, time_step=10):
    if len(X) < time_step:
        X_padded = np.pad(X, ((time_step - len(X), 0), (0, 0)), mode='constant', constant_values=0)
    else:
        X_padded = X
    Xs = []
    for i in range(len(X_padded) - time_step + 1):
        Xs.append(X_padded[i:(i + time_step)])
    return np.array(Xs)

@app.post("/predict/")
async def predict(request: PredictionRequest):
    try:
        input_data = np.array([[request.latitude, request.longitude]])
        input_data_scaled = scaler_X.transform(input_data)

        time_step = 10
        X = create_dataset(input_data_scaled, time_step)

        if X.shape[0] == 0:
            raise HTTPException(status_code=400, detail="Insufficient data length for prediction")

        prediction_scaled = model.predict(X)
        prediction = scaler_y.inverse_transform(prediction_scaled)

        return {"predicted_water_level": prediction.flatten().tolist()}

    except Exception as e:
        return {"error": str(e)}

