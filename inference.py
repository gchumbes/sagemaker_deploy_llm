from sagemaker_huggingface_inference_toolkit import decoder_encoder
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import torch

def model_fn(model_dir):
    """Carga el modelo Llama automáticamente en GPU si está disponible, sino en CPU."""
    print("[model_fn] Cargando el modelo desde:", model_dir)

    # Detectar dispositivo
    device = 0 if torch.cuda.is_available() else -1
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    print(f"[model_fn] Dispositivo detectado: {'GPU' if device == 0 else 'CPU'}")
    print(f"[model_fn] Tipo de precisión: {torch_dtype}")

    # Cargar tokenizador
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    print("[model_fn] Tokenizador cargado correctamente.")

    # Cargar modelo con el tipo y dispositivo adecuados
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        device_map="auto" if torch.cuda.is_available() else None  # solo en GPU
    )
    print("[model_fn] Modelo cargado correctamente.")

    # Crear pipeline
    text_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        #device=device #transformers detecta que existe gpu y si se le fuerza con device entra en conflicto
    )
    print("[model_fn] Pipeline de inferencia creado correctamente.")
    
    return text_pipeline

def input_fn(input_data, content_type):
    """Decodifica la entrada JSON de SageMaker."""
    print("[input_fn] Recibiendo input con content_type:", content_type)
    data = decoder_encoder.decode(input_data, content_type)
    print("[input_fn] Datos decodificados:", data)
    return {
        "prompt": data.get("prompt", ""),
        "max_length": data.get("max_length", 350),
        "temperature": data.get("temperature", 0.5),
        "do_sample":data.get("do_sample",True),
        "top_p": data.get("top_p", 0.9),
        "repetition_penalty": data.get("repetition_penalty", 1.1),
        "top_k": data.get("top_k", 50),
        "num_return_sequences": data.get("num_return_sequences", 1),

    }

def predict_fn(data, model):
    """Genera texto usando el modelo."""
    print("[predict_fn] Procesando predicción con datos:", data)
    response = model(
        data["prompt"],
        max_length=data["max_length"],
        temperature=data["temperature"],
        do_sample=data["do_sample"],
        top_p=data["top_p"],
        repetition_penalty=data["repetition_penalty"],
        top_k=data["top_k"],
        num_return_sequences=data["num_return_sequences"],
    )
    print("[predict_fn] Respuesta cruda del modelo:", response)

    if isinstance(response, list) and len(response) > 0 and "generated_text" in response[0]:
        response_text = response[0]["generated_text"].replace(data["prompt"], "").strip()
    else:
        response_text = str(response)

    print("[predict_fn] Respuesta procesada:", response_text)
    return {"response": response_text}

def output_fn(prediction, accept):
    """Codifica la salida en formato JSON."""
    print("[output_fn] Formateando salida para accept:", accept)
    encoded_output = decoder_encoder.encode(prediction, accept)
    print("[output_fn] Respuesta codificada correctamente.")
    print("respuesta a enviar:", encoded_output)
    return encoded_output
