import sagemaker
import boto3

iam_client = boto3.client('iam')
role = iam_client.get_role(RoleName="SageMakerTestRole")['Role']['Arn']
sess = sagemaker.Session()
from sagemaker.huggingface.model import HuggingFaceModel

# Definir variables de entorno
environment_variables = {
    "MMS_DEFAULT_WORKERS_PER_MODEL": "1",
    "MMS_NETTY_CLIENT_THREADS": "1",
    "MMS_NUMBER_OF_NETTY_THREADS": "1",
    "MMS_DEFAULT_RESPONSE_TIMEOUT": "360",  # 6 minutos
    "MMS_MAX_REQUEST_SIZE": "52428800",
    "MMS_MAX_RESPONSE_SIZE": "52428800"
}

# Crear Hugging Face Model Class
huggingface_model = HuggingFaceModel(
   model_data="s3://test-bucket-llm-local/modelV4_Base.tar.gz",  # Ruta a tu modelo entrenado en SageMaker
   role="arn:aws:iam::183631337267:role/SageMakerTestRole",  # IAM Role con permisos para crear un endpoint
   transformers_version="4.48.0",  # Versión de Transformers usada
   pytorch_version="2.3.0",       # Versión de PyTorch usada
   py_version='py311',            # Versión de Python usada
   model_server_workers=1,        # Un trabajador por modelo
   env=environment_variables  # Pasar las variables de entorno
)

# Desplegar el modelo en SageMaker Inference
predictor = huggingface_model.deploy(
   initial_instance_count=1,
   #instance_type="ml.t2.2xlarge",
   instance_type="ml.g5.xlarge", # modelo cpu  16 GB gpu 24GB
   endpoint_name="model-endpoint-syncV4-Base"
   # Se puede ajustar el tipo de instancia si es necesario, dependiendo del uso de CPU/RAM
)

# # Ejemplo de solicitud de inferencia:
# data = {
#    "inputs": "Camera - You are awarded a SiPix Digital Camera! Call 09061221066 from the landline."
# }

# # Solicitar inferencia
# response = predictor.predict(data)
# print(response)
