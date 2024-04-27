import argparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.combine import SMOTEENN
from loguru import logger
import os
import pandas as pd
import pickle

# Parsear los argumentos de la línea de comandos
parser = argparse.ArgumentParser()
parser.add_argument('--data_file', required=True, type=str, help='a csv file with train data')
parser.add_argument('--model_file', required=True, type=str, help='where the trained model will be stored')
parser.add_argument('--overwrite_model', default=False, action='store_true', help='if sets overwrites the model file if it exists')

args = parser.parse_args()

# Obtener los argumentos de la línea de comandos
model_file = args.model_file
data_file = args.data_file
overwrite = args.overwrite_model

# Verificar si el archivo de modelo ya existe
if os.path.isfile(model_file):
    if overwrite:
        logger.info(f"overwriting existing model file {model_file}")
    else:
        logger.info(f"model file {model_file} exists. exitting. use --overwrite_model option")
        exit(-1)

# Cargar los datos de entrenamiento
logger.info("loading train data")
z = pd.read_csv(data_file)

# Codificar los datos de entrenamiento
logger.info("encoding train data")
gender_dict = {'Female': 0, 'Male': 1, 'Other': 2}
smoking_history_dict = {'No Info': 0, 'current': 1, 'ever': 2, 'former': 3, 'never': 4, 'not current': 5}
z = z.replace({'gender': gender_dict, 'smoking_history': smoking_history_dict})

# Separar las características (Xtr) y las etiquetas (ytr)
Xtr = z.drop('diabetes', axis=1)
ytr = z[['diabetes']]

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
