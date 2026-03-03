import os
import sys
import importlib
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage

# Importaciones dinámicas
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
state_models = importlib.import_module("07_state_models")
agent_config = importlib.import_module("08_agent_config")
tools_module = importlib.import_module("05_agent_tools")
prompts_module = importlib.import_module("06_system_prompts")

AgentState = state_models.AgentState
buscar_en_documentos = tools_module.buscar_en_documentos
llm = importlib.import_module("09_nodos_principales").llm
MAX_TOOL_CALLS = agent_config.MAX_TOOL_CALLS

# --- NODOS DEL SUBGRAFO ---

def agent_orchestrator(state: AgentState):
    agent_id = state.get("question_index", 0)
    query = state["question"]
    
    if state["tool_call_count"] >= MAX_TOOL_CALLS:
        return {"next_step": "responder", "agent_reasoning": "Límite alcanzado."}

    if not state.get("context_summary"):
        return {"next_step": "buscar"}
    
    # Prompt más estricto: pedimos que verifique si el dato ESPECÍFICO está presente
    prompt = f"""Pregunta: {query}
Contexto recuperado: {state['context_summary'][:1000]}

¿El contexto contiene la respuesta EXACTA a la pregunta?
- Si el contexto menciona la respuesta directamente, responde 'SI'.
- Si el contexto habla del tema pero NO responde la duda específica, responde 'NO'.
Responde solo SI o NO."""

    decision = llm.invoke([HumanMessage(content=prompt)]).content.upper()
    
    # Si el modelo dice SI, avanzamos. Si dice NO, forzamos otra búsqueda.
    if "SI" in decision:
        print(f"✅ [AGENTE {agent_id}] Datos encontrados en el contexto.")
        return {"next_step": "responder"}
    else:
        # Aquí es donde el agente evoluciona la búsqueda en lugar de rendirse
        intento = state["tool_call_count"]
        palabras_clave = ["calificación", "puntaje", "indicador", "tabla", "2024", "Sustainalytics", "MSCI"]
        nueva_busqueda = f"{state['question']} {palabras_clave[intento % len(palabras_clave)]}"
    
        print(f"🔄 [AGENTE {agent_id}] Refinando búsqueda a: '{nueva_busqueda}'")
        return {
            "next_step": "buscar", 
            "next_search_query": nueva_busqueda,
            "agent_reasoning": "Buscando términos específicos de la tabla financiera."
        }
    
def retrieve_tool_node(state: AgentState):
    agent_id = state.get("question_index", 0)
    # Usa la consulta evolucionada en lugar de la original siempre
    query_to_use = state.get("next_search_query") or state["question"]
    
    res = buscar_en_documentos.invoke(query_to_use)
    return {
        "context_summary": (state.get("context_summary", "") + f"\n[Busqueda: {query_to_use}]: {res}").strip(),
        "tool_call_count": 1 
    }

def summarize_context_node(state: AgentState):
    """Compresión de contexto si supera el límite de tokens (Paso 8)."""
    prompt = f"Resume este contexto técnico para responder '{state['question']}':\n{state['context_summary']}"
    resumen = llm.invoke([SystemMessage(content="Sintetiza."), HumanMessage(content=prompt)])
    return {"context_summary": resumen.content}

def generate_agent_answer(state: AgentState):
    """Genera la respuesta parcial del sub-agente."""
    system_msg = prompts_module.AGENTE_RAG_PROMPT
    user_msg = f"Contexto:\n{state.get('context_summary')}\nPregunta: {state['question']}"
    res = llm.invoke([SystemMessage(content=system_msg), HumanMessage(content=user_msg)])
    return {"agent_answers": [{"index": state["question_index"], "answer": res.content}]}

# --- CONSTRUCCIÓN Y ARISTAS ---

def decide_compression(state: AgentState):
    tokens = agent_config.estimate_context_tokens([HumanMessage(content=state.get("context_summary", ""))])
    return "comprimir" if tokens > agent_config.BASE_TOKEN_THRESHOLD else "evaluar"

agent_builder = StateGraph(AgentState)
agent_builder.add_node("orchestrator", agent_orchestrator)
agent_builder.add_node("retrieve", retrieve_tool_node)
agent_builder.add_node("summarize", summarize_context_node)
agent_builder.add_node("generate", generate_agent_answer)

agent_builder.set_entry_point("orchestrator")
agent_builder.add_conditional_edges("orchestrator", lambda x: x["next_step"], {"buscar": "retrieve", "responder": "generate"})
agent_builder.add_conditional_edges("retrieve", decide_compression, {"comprimir": "summarize", "evaluar": "orchestrator"})
agent_builder.add_edge("summarize", "orchestrator")
agent_builder.add_edge("generate", END)

agent_subgraph = agent_builder.compile()