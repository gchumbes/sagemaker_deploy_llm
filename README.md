# Llama-3.2-1B-Instruct - SageMaker Deployment

Este repositorio contiene el modelo Llama 3.2 1B Instruct listo para ser desplegado en AWS SageMaker. Incluye scripts de inferencia, configuración y despliegue.

## 📦 Estructura del Proyecto

```
Llama-3.2-1B-Instruct/
├── code/                     # Carpeta obligatoria para SageMaker, contiene lógica de inferencia
│   ├── inference.py          # Script que carga el modelo y define cómo manejar las peticiones
│   └── requirements.txt      # Librerías necesarias para el entorno de inferencia
├── deploy.py                 # Script para desplegar el modelo en SageMaker (opcional, fuera del tar)
├── config.json
├── generation_config.json
├── LICENSE.txt
├── model.safetensors
├── original/
│   ├── consolidated.00.pth
│   ├── params.json
│   └── tokenizer.model
├── special_tokens_map.json
├── tokenizer_config.json
├── tokenizer.json
├── USE_POLICY.md
```

## 🧾 Instrucciones de Uso

1. **Añadir la carpeta `code/` al directorio del modelo**, junto a los archivos de pesos (`model.safetensors`, `config.json`, etc.).
2. Comprimir el contenido del modelo (incluyendo `code/`) en un archivo `.tar.gz`.
3. Subir este archivo a un bucket de Amazon S3:
   ```bash
   aws s3 cp modelV4_Base.tar.gz s3://your-bucket-name/
   ```
4. Ejecutar el script `deploy.py` (opcional) desde tu entorno local o un notebook para desplegar el modelo en SageMaker. Este script **no debe incluirse en el `.tar.gz`**.

## 📌 Requisitos

- AWS CLI configurado
- Permisos adecuados en IAM (ej. rol SageMaker)
- SageMaker Studio o Jupyter con soporte para PyTorch y Hugging Face

## 📫 Contacto

Para dudas o soporte técnico, contactar al equipo de ML o al responsable del despliegue.

