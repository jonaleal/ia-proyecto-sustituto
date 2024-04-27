import argparse
import numpy as np
from loguru import logger
from sklearn.preprocessing import StandardScaler
import os
import pandas as pd
import pickle

# Parsear los argumentos de la línea de comandos
parser = argparse.ArgumentParser()
parser.add_argument('--input_file', required=True, type=str, help='a csv file with input data (no targets)')
parser.add_argument('--predictions_file', required=True, type=str, help='a csv file where predictions will be saved to')
parser.add_argument('--model_file', required=True, type=str, help='a pkl file with a model already stored (see train.py)')

args = parser.parse_args()

# Obtener los argumentos de la línea de comandos
model_file = args.model_file
input_file = args.input_file
predictions_file = args.predictions_file

# Verificar si el archivo del modelo existe
if not os.path.isfile(model_file):
    logger.error(f"model file {model_file} does not exist")
    exit(-1)

# Verificar si el archivo de entrada existe
if not os.path.isfile(input_file):
    logger.error(f"input file {input_file} does not exist")
    exit(-1)
    
# Cargar los datos de entrada
logger.info("loading input data")
Xts = pd.read_csv(input_file)

# Codificar los datos de entrada
logger.info("encoding data")
gender_dict = {'Female': 0, 'Male': 1, 'Other': 2}
smoking_history_dict = {'No Info': 0, 'current': 1, 'ever': 2, 'former': 3, 'never': 4, 'not current': 5}
Xts = Xts.replace({'gender': gender_dict, 'smoking_history': smoking_history_dict})

# Escalar los datos de entrada
logger.info("scaling data")
scaler = StandardScaler()
Xts = scaler.fit_transform(Xts)

# Cargar el modelo
logger.info("loading model")
with open(model_file, 'rb') as f:
    m = pickle.load(f)

# Hacer predicciones
logger.info("making predictions")
preds = m.predict(Xts)

# Guardar las predicciones en un archivo
logger.info(f"saving predictions to {predictions_file}")
pd.DataFrame(preds.reshape(-1,1), columns=['preds']).to_csv(predictions_file, index=False)
