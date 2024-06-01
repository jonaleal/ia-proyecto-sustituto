from fastapi import FastAPI
from patient import Patient

import train as tr
import predict as pr


app = FastAPI()


@app.post("/train")
def train():
    """
    Expone un endpoint para entrenar un modelo de aprendizaje automático.
    """
    return tr.train()


@app.post("/predict")
def predict(patient: Patient):
    """
    Expone un endpoint para hacer una predicción basada en la información del paciente.

    Parámetros:
    - patient: Un objeto Patient que contiene la información del paciente de entrada.

    Devuelve:
    - Un diccionario con un mensaje de predicción.
    """
    return pr.predict(patient)
