# Seleccionar una imagen baseada en Python
FROM python:3.12

# Establecer el directorio de trabajo
WORKDIR /app

# Copiar los archivos necesarios al directorio de la aplicación
ADD requirements.txt /app
ADD apirest.py /app
ADD patient.py /app

# Instalar las dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Ejecutar la aplicación
CMD ["fastapi", "run", "apirest.py", "--port", "80"]