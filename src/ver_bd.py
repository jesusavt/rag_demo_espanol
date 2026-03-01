from qdrant_client import QdrantClient

RUTA_QDRANT = "qdrant_db"
COLECCION_NOMBRE = "demo_rag"

def ver_contenido_bd():
    print("🗄️ Conectando a la base de datos Qdrant...\n")
    cliente = QdrantClient(path=RUTA_QDRANT)
    
    # Validar que la colección exista
    if not cliente.collection_exists(COLECCION_NOMBRE):
        print(f"La colección '{COLECCION_NOMBRE}' no existe.")
        return

    # Extraer información de la colección
    info = cliente.get_collection(COLECCION_NOMBRE)
    print(f"📊 Total de vectores (fragmentos) guardados: {info.points_count}\n")
    print("-" * 50)

    # Hacer un "scroll" para traer los primeros 3 registros
    # with_payload=True trae el texto y metadatos
    # with_vectors=True trae los números matemáticos (lo limitaremos visualmente)
    registros, _ = cliente.scroll(
        collection_name=COLECCION_NOMBRE,
        limit=3,
        with_payload=True,
        with_vectors=True
    )

    for i, registro in enumerate(registros, 1):
        print(f"🔍 FRAGMENTO {i}")
        print(f"ID del punto: {registro.id}")
        
        # El payload contiene el texto original y los metadatos de LangChain
        payload = registro.payload
        print("\n📝 METADATOS (Jerarquía):")
        print(payload.get('metadata', 'No hay metadatos'))
        
        print("\n📄 TEXTO (page_content):")
        print(payload.get('page_content', 'No hay texto'))
        
        # El vector es una lista de 384 números, solo mostramos un resumen
        vector = registro.vector
        print(f"\n🧠 VECTOR (Embeddings):")
        print(f"Tamaño: {len(vector)} dimensiones")
        print(f"Muestra de los primeros 5 números: {vector[:5]}...")
        print("-" * 50)

if __name__ == "__main__":
    ver_contenido_bd()