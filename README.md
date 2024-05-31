## Datos estudiante

**Nombre**: Jonatan Jair Leal González 
**C.C**: 1128482518
**Programa**: Ingeniería de sistemas
# Predicción de diabetes
**kaggle**: https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset

# Fase-1 
- Abre el notebook ````Diabetes_Prediction```` del directorio ```fase-1``` en google colab.
- Sube las credenciales de kaggle a colab (kaggle.json).
- Ejecuta el notebook para ver cómo se entrena y predice con el modelo.

# Fase-2
**Sobre el directorio ``fase-2`` abre una terminal y ejecuta:**
```shell
docker build -t ai-proyecto-sustituto .
docker run -it --name ai-container ai-proyecto-sustituto /bin/bash
```

**Desde una nueva terminal sobre el directorio ``resources`` ejecuta:**
```shell
docker cp train.csv ai-container:/app
docker cp test.csv ai-container:/app
```

**Vuelve a la primera terminal y ejecuta:**
```shell
python train.py --model_file model.pkl --data_file train.csv  --overwrite_model
python predict.py --model_file model.pkl --input_file test.csv  --predictions_file predictions.csv
```

**Finalmente para ver las predicciones ejecuta:**
```shell
cat predictions.csv
```

# Fase-3
**Sobre el directorio ``fase-3`` abre una terminal y ejecuta:**
```shell
docker build -t apirest .
docker run -d --name apirest-container -p 80:80 apirest
```

**Desde una nueva terminal sobre el directorio ``resources`` ejecuta:**
```shell
docker cp train.csv apirest-container:/app
```

**Prueba la API-REST**

- En el navegador ve a ```localhost/docs``` donde verás la interfaz de swagger y podrás probar cada endpoint.

- Primero entrena el modelo. Para ello ve al endpoint ``/train`` haz clic sobre ``Try it out`` y luego sobre ``Execute``.

- Ahora haz una predicción. Para ello ve al endpoint ``/predict`` haz clic sobre ``Try it out`` y en el ``Request Body`` envía los datos necesarios para realizar la predicción, por ejemplo:
    ```json
    {
    "gender": "Female",
    "age": 36,
    "hypertension": 0,
    "heart_disease": 0,
    "smoking_history": "current",
    "bmi": 32.27,
    "HbA1c_level": 6.2,
    "blood_glucose_level": 220
    }
    ```
    luego haz clic sobre ``Execute``.
