import tiktoken

# Límites estrictos para llamadas a herramientas e iteraciones para evitar bucles infinitos locales
MAX_TOOL_CALLS = 5       
MAX_ITERATIONS = 5       

# Configuración para la compresión de contexto
BASE_TOKEN_THRESHOLD = 2000     
TOKEN_GROWTH_FACTOR = 0.9       

def estimate_context_tokens(messages: list) -> int:
    """
    Estima la cantidad de tokens en la memoria del agente.
    Usamos 'cl100k_base' (el estándar de OpenAI) como una buena aproximación 
    para modelos locales y evitar que la memoria colapse.
    """
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
    except Exception:
        # Fallback de seguridad por si hay algún problema con tiktoken
        return sum(len(str(getattr(msg, 'content', ''))) // 4 for msg in messages)
        
    return sum(len(encoding.encode(str(getattr(msg, 'content', '')))) for msg in messages)

if __name__ == "__main__":
    print("✅ Configuración del agente (Paso 8) definida correctamente.")
