# Seleccionar una imagen baseada en Python
FROM python:3.12

# Establecer el directorio de trabajo
WORKDIR /app

# Copiar los archivos necesarios al directorio de la aplicación
ADD train.py /app
ADD predict.py /app
ADD requirements.txt /app

# Instalar las dependencias
RUN pip install --no-cache-dir -r requirements.txt