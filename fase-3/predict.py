from patient import Patient
from sklearn.preprocessing import StandardScaler
from imblearn.combine import SMOTEENN
from loguru import logger
import os
import pandas as pd
import pickle


model_file = "model.pkl"  # Ruta del archivo de modelo

def predict(patient: Patient):
    """
    Hace una predicción basada en la información del paciente de entrada.

    Parámetros:
    - patient: Un objeto Patient que contiene la información del paciente de entrada.

    Devuelve:
    - Un diccionario con un mensaje de predicción.
    """

    try:
        # Cargar los datos de entrada
        Xts = pd.DataFrame([patient.model_dump()])

        # Verificar si el archivo del modelo existe
        if not os.path.isfile(model_file):
            logger.error(f"model file {model_file} does not exist")
            exit(-1)

        # Codificar los datos de entrada
        logger.info("encoding data")
        gender_dict = {"Female": 0, "Male": 1, "Other": 2}
        smoking_history_dict = {
            "No Info": 0,
            "current": 1,
            "ever": 2,
            "former": 3,
            "never": 4,
            "not current": 5,
        }
        Xts = Xts.replace({"gender": gender_dict, "smoking_history": smoking_history_dict})

        # Escalar los datos de entrada
        logger.info("scaling data")
        scaler = StandardScaler()
        Xts = scaler.fit_transform(Xts)

        # Cargar el modelo
        logger.info("loading model")
        with open(model_file, "rb") as f:
            m = pickle.load(f)

        # Hacer predicciones
        logger.info("making predictions")
        preds = m.predict(Xts)

        return {"message": "Tiene diabetes" if preds[0] == 1 else "No tiene diabetes"}
    
    except:
        return {"message": "Something went wrong"}
