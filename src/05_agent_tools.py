from langchain_core.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

# Rutas locales
RUTA_QDRANT = "qdrant_db"
COLECCION_NOMBRE = "demo_rag"

# Inicializamos el motor de búsqueda fuera de la función 
# para no recargar el modelo de embeddings en cada consulta
print("🧠 Preparando motor de búsqueda...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
cliente_qdrant = QdrantClient(path=RUTA_QDRANT)

vector_store = QdrantVectorStore(
    client=cliente_qdrant,
    collection_name=COLECCION_NOMBRE,
    embedding=embeddings,
)

@tool
def buscar_en_documentos(pregunta: str) -> str:
    """
    Busca información en la base de datos de documentos locales.
    Úsala SIEMPRE que necesites responder preguntas sobre el contenido del PDF o temas específicos del usuario.
    """
    print(f"\n[🔍 TOOL] Ejecutando búsqueda semántica para: '{pregunta}'")
    
    # Extraemos los 3 fragmentos más similares matemáticamente a la pregunta
    resultados = vector_store.similarity_search(pregunta, k=10)
    
    if not resultados:
        return "No se encontró información relevante en la base de datos."
        
    # Formateamos el resultado para entregárselo en texto plano al LLM
    textos_encontrados = []
    for i, doc in enumerate(resultados, 1):
        metadatos = doc.metadata
        texto = doc.page_content
        # Incluimos los metadatos para que el LLM sepa de qué sección viene
        textos_encontrados.append(f"--- Fragmento {i} ---\nSección: {metadatos}\nContenido: {texto}\n")
        
    return "\n".join(textos_encontrados)

if not cliente_qdrant.collection_exists(COLECCION_NOMBRE):
    print(f"⚠️ Alerta: La colección '{COLECCION_NOMBRE}' no existe. Ejecuta primero 04_hierarchical_indexing.py")

if __name__ == "__main__":
    # Prueba manual directa de la herramienta sin usar el Agente aún
    print("\n--- INICIANDO PRUEBA DE HERRAMIENTA ---")
    
    # Cambia esta pregunta por algo que sepas que está en tu documento.pdf
    pregunta_prueba = "dime lo mas importante del documento" 
    resultado = buscar_en_documentos.invoke(pregunta_prueba)
    
    print("\n[📦 RAW TO LLM] -> Esto es lo que leerá el modelo:")
    print(resultado)