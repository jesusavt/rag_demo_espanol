import pymupdf4llm
import os

# Definir las nuevas rutas apuntando a agentic-rag-for-dummies
DIRECTORIO_RAW = "data/raw"
DIRECTORIO_MD = "data/data_md"
RUTA_PDF = f"{DIRECTORIO_RAW}/documento.pdf"
RUTA_MD = f"{DIRECTORIO_MD}/documento.md"

def convertir_pdf_a_md():
    print("📄 Ejecutando Paso 3: Convirtiendo PDF a Markdown...")
    
    # Crear las carpetas si no existen (por seguridad)
    os.makedirs(DIRECTORIO_RAW, exist_ok=True)
    os.makedirs(DIRECTORIO_MD, exist_ok=True)
    
    # Verificar que el PDF esté en la carpeta raw
    if not os.path.exists(RUTA_PDF):
        print(f"❌ Error: No se encontró el archivo en {RUTA_PDF}")
        print("Asegúrate de colocar 'documento.pdf' dentro de 'data/raw/'")
        return
        
    # Extraer texto del PDF directamente a formato Markdown
    texto_md = pymupdf4llm.to_markdown(RUTA_PDF)
    
    # Guardar el resultado en la carpeta data_md
    with open(RUTA_MD, "w", encoding="utf-8") as archivo_md:
        archivo_md.write(texto_md)
        
    print(f"✅ Éxito: Archivo Markdown generado y guardado en {RUTA_MD}")

if __name__ == "__main__":
    convertir_pdf_a_md()