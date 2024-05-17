from fastapi import FastAPI
from patient import Patient
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.combine import SMOTEENN
from loguru import logger
import os
import pandas as pd
import pickle

app = FastAPI()


model_file = "model.pkl"  # Ruta del archivo de modelo
data_file = "train.csv"  # Ruta del archivo de datos de entrenamiento
overwrite = True  # Sobrescribir el modelo existente


@app.post("/train")
def train():
    """
    Una función que entrena un modelo de aprendizaje automático utilizando los datos proporcionados en 'train.csv'.
    Carga los datos de entrenamiento, los codifica, los escala, aplica el remuestreo SMOTEENN, ajusta un modelo RandomForestClassifier y guarda el modelo entrenado en 'model.pkl'.
    """

    try:
        # Verificar si el archivo de modelo ya existe
        if os.path.isfile(model_file):
            if overwrite:
                logger.info(f"overwriting existing model file {model_file}")
            else:
                logger.info(
                    f"model file {model_file} exists. exitting. use --overwrite_model option"
                )
                exit(-1)

        # Cargar los datos de entrenamiento
        logger.info("loading train data")
        z = pd.read_csv(data_file)

        # Codificar los datos de entrenamiento
        logger.info("encoding train data")
        gender_dict = {"Female": 0, "Male": 1, "Other": 2}
        smoking_history_dict = {
            "No Info": 0,
            "current": 1,
            "ever": 2,
            "former": 3,
            "never": 4,
            "not current": 5,
        }
        z = z.replace({"gender": gender_dict, "smoking_history": smoking_history_dict})

        # Separar las características (Xtr) y las etiquetas (ytr)
        Xtr = z.drop("diabetes", axis=1)
        ytr = z[["diabetes"]]

        # Escalar los datos de entrenamiento
        logger.info("scaling train data")
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(Xtr)

        # Aplicar sobre y submuestreo con SMOTEENN
        smote_enn = SMOTEENN(random_state=42)
        Xtr, ytr = smote_enn.fit_resample(Xtr, ytr)

        # Entrenar el modelo
        logger.info("fitting model")
        m = RandomForestClassifier()
        m.fit(Xtr, ytr)

        # Guardar el modelo en un archivo
        logger.info(f"saving model to {model_file}")
        with open(model_file, "wb") as f:
            pickle.dump(m, f)

        return {"message": "Model successfully trained"}

    except:
        return {"message": "Something went wrong"}


@app.post("/predict")
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
