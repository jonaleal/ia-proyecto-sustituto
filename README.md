## Datos estudiante

**Nombre**: Jonatan Jair Leal González 
**C.C**: 1128482518
**Programa**: Ingeniería de sistemas
# Predicción de diabetes
**kaggle**: https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset

# Fase-1 
1. Abre el notebook ````Diabetes_Prediction```` del directorio ```fase-1``` en google colab.
2. Sube las credenciales de kaggle a colab (kaggle.json).
3. Ejecuta el notebook para ver cómo se entrena y predice con el modelo.

# Fase-2
**Sobre el directorio fase-2 abre una terminal y ejecuta:**
- ``docker build -t ai-proyecto-sustituto .``
- ``docker run -it --name ai-container ai-proyecto-sustituto /bin/bash``

**Desde una nueva terminal sobre el directorio fase-2 ejecuta:**
- ``docker cp train.csv ai-container:/app`` 
- ``docker cp test.csv ai-container:/app ``

**Vuelve a la primera terminal y ejecuta:**
- ``python train.py --model_file model.pkl --data_file train.csv  --overwrite_model``
- ``python predict.py --model_file model.pkl --input_file test.csv  --predictions_file predictions.csv``

**Finalmente para ver las predicciones ejecuta:**
- ``cat predictions.csv``

