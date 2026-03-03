import os
import sys
import importlib.util
from langgraph.graph import StateGraph, END

# Configuración de rutas
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_PATH)

def import_numeric_file(module_name, file_name):
    file_path = os.path.join(BASE_PATH, file_name)
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Importaciones dinámicas de los pasos anteriores
nodos_9 = import_numeric_file("nodos_9", "09_nodos_principales.py")
subgrafo_10 = import_numeric_file("subgrafo_10", "10_agent_subgraph.py")

State = nodos_9.State
agent_subgraph = subgrafo_10.agent_subgraph

# --- NUEVO NODO DE ACLARACIÓN ---
def request_clarification_node(state: State):
    """Devuelve el mensaje de aclaración generado por el maestro."""
    return {"messages": state["messages"]}

# --- CONSTRUCCIÓN DEL GRAFO MAESTRO ---
workflow = StateGraph(State)

# 1. Agregar todos los nodos
workflow.add_node("summarize_history", nodos_9.summarize_history)
workflow.add_node("rewrite_query", nodos_9.rewrite_query)
workflow.add_node("agent", agent_subgraph)
workflow.add_node("aggregate_answers", nodos_9.aggregate_answers)
workflow.add_node("request_clarification", request_clarification_node) # Nodo nuevo

# 2. Definir el flujo (Aristas)
workflow.set_entry_point("summarize_history")
workflow.add_edge("summarize_history", "rewrite_query")

# Arista condicional: decide si va a los agentes o pide aclaración
workflow.add_conditional_edges(
    "rewrite_query", 
    nodos_9.route_after_rewrite,
    {
        "agent": "agent", 
        "request_clarification": "request_clarification"
    }
)

# 3. Cerrar el flujo
workflow.add_edge("agent", "aggregate_answers")
workflow.add_edge("aggregate_answers", END)
workflow.add_edge("request_clarification", END) # La aclaración termina el flujo

# 4. COMPILAR (Solo una vez al final)
app = workflow.compile()