import os
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

# Rutas locales
RUTA_MD = "data/data_md/documento.md"
RUTA_QDRANT = "qdrant_db" # Carpeta donde se creará la BD
COLECCION_NOMBRE = "demo_rag"

def indexar_documentos():
    print("📖 Leyendo archivo Markdown...")
    if not os.path.exists(RUTA_MD):
        print(f"❌ Error: No se encontró {RUTA_MD}")
        return
        
    with open(RUTA_MD, "r", encoding="utf-8") as f:
        texto_md = f.read()

    # 1. División Jerárquica por Markdown
    print("✂️ Dividiendo por jerarquía de títulos...")
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    fragmentos_md = markdown_splitter.split_text(texto_md)

    # 2. Sub-división por tamaño (por si un bloque bajo un título es muy largo)
    print("📏 Ajustando tamaño de fragmentos...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    fragmentos_finales = text_splitter.split_documents(fragmentos_md)

    print(f"Total de fragmentos generados: {len(fragmentos_finales)}")

    # 3. Cargar Embeddings locales
    print("🧠 Cargando modelo de embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 4. Conectar a Qdrant (Modo Local en disco)
    print("🗄️ Inicializando Qdrant...")
    cliente_qdrant = QdrantClient(path=RUTA_QDRANT)

    if not cliente_qdrant.collection_exists(COLECCION_NOMBRE):
        cliente_qdrant.create_collection(
            collection_name=COLECCION_NOMBRE,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )

    # 5. Inicializar el VectorStore de LangChain
    print("🔗 Vinculando LangChain con Qdrant...")
    vector_store = QdrantVectorStore(
        client=cliente_qdrant,
        collection_name=COLECCION_NOMBRE,
        embedding=embeddings,
    )

    # 6. Guardar los vectores
    print("💾 Insertando vectores...")
    vector_store.add_documents(documents=fragmentos_finales)
    
    print("✅ Indexación completada con éxito.")
    
    print("✅ Indexación completada con éxito.")

if __name__ == "__main__":
    indexar_documentos()