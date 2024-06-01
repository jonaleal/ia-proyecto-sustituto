# Seleccionar una imagen baseada en Python
FROM python:3.12

# Establecer el directorio de trabajo
WORKDIR /app

# Copiar los archivos necesarios al directorio de la aplicación
ADD .. /app

# Instalar las dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Ejecutar la aplicación
CMD ["fastapi", "run", "apirest.py", "--port", "80"]