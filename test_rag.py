import sys
import os
import importlib.util

# Ruta absoluta a la carpeta src
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "src")
sys.path.append(SRC_DIR)

def cargar_app():
    path_11 = os.path.join(SRC_DIR, "11_main_graph.py")
    spec = importlib.util.spec_from_file_location("main_graph", path_11)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo.app

if __name__ == "__main__":
    from langchain_core.messages import HumanMessage

    try:
        app = cargar_app()
        print("🚀 Iniciando RAG...")
        app = cargar_app()
        
        pregunta = "¿Cuántos micronegocios estaban afiliados a Yape en 2024 y cuántos incluidos financieramente?" # Prueba con algo de tu PDF
        inputs = {"messages": [HumanMessage(content=pregunta)]}

        for output in app.stream(inputs):
            for node, values in output.items():
                print(f"📍 Nodo finalizado: {node}")
        
            # NUEVO: Imprimir el contenido final cuando llegue al agregador
                if node == "aggregate_answers":
                    mensaje_final = values["messages"][-1].content
                    print(f"\n================ FINAL ANSWER ================\n")
                    print(mensaje_final)
                    print(f"\n==============================================\n")
         
    except Exception as e:
        print(f"Error: {e}")
    finally:
        print("👋 Cerrando recursos...")

