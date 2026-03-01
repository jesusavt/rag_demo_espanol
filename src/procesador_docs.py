import os
import pymupdf4llm
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

# Configuraciones básicas
RUTA_PDF = "../data/documento.pdf"  # Asegúrate de poner un PDF de prueba en la carpeta data
COLECCION_NOMBRE = "demo_rag"
RUTA_QDRANT = "../qdrant_db" # Carpeta local donde se guardará la BD por ahora

def procesar_y_guardar():
    # 1. Extraer texto del PDF (usando PyMuPDF que es muy rápido)
    print("📄 Leyendo PDF...")
    documentos = pymupdf4llm.LlamaMarkdownReader().load_data(RUTA_PDF)
    
    # 2. Dividir el texto en fragmentos (chunks)
    print("✂️ Dividiendo texto...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    fragmentos = text_splitter.split_documents(documentos)
    
    # 3. Cargar el modelo de Embeddings (Local via HuggingFace/SentenceTransformers)
    print("🧠 Cargando modelo de embeddings...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # 4. Inicializar cliente Qdrant (Modo local en disco por ahora)
    print("🗄️ Conectando a Qdrant...")
    cliente_qdrant = QdrantClient(path=RUTA_QDRANT)
    
    # Crear la colección si no existe
    if not cliente_qdrant.collection_exists(COLECCION_NOMBRE):
        cliente_qdrant.create_collection(
            collection_name=COLECCION_NOMBRE,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )

    # 5. Guardar los vectores en Qdrant
    print("💾 Guardando vectores...")
    QdrantVectorStore.from_documents(
        fragmentos,
        embeddings,
        client=cliente_qdrant,
        collection_name=COLECCION_NOMBRE,
    )
    
    print("✅ Proceso completado con éxito.")

if __name__ == "__main__":
    # Crear carpeta data si no existe
    os.makedirs("../data", exist_ok=True)
    procesar_y_guardar()