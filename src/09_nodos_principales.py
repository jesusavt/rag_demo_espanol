import os
import sys
import importlib
from typing import List, Dict, Any, Literal
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.types import Send
from langchain_ollama import ChatOllama

# Configuración de rutas para evitar errores de importación
DIRECTORIO_ACTUAL = os.path.dirname(os.path.abspath(__file__))
if DIRECTORIO_ACTUAL not in sys.path:
    sys.path.append(DIRECTORIO_ACTUAL)

# Importaciones dinámicas seguras
state_models = importlib.import_module("07_state_models")
State = state_models.State
QueryAnalysis = state_models.QueryAnalysis

# Inicialización del LLM local (Ollama)
llm = ChatOllama(model="qwen2.5:7b", temperature=0.1)

# --- PROMPT DE REESCRITURA OPTIMIZADO ---
PROMPT_REESCRITURA = """Analiza la consulta del usuario.
Si la pregunta pide datos numéricos múltiples (ej: cuántos préstamos Y qué monto), 
OBLIGATORIAMENTE divídela en sub-preguntas separadas para asegurar que buscamos en diferentes partes del PDF.
"""

# --- NODOS DEL GRAFO PRINCIPAL ---

def summarize_history(state: State):
    """Resume la charla si hay demasiados mensajes."""
    if len(state["messages"]) < 4:
        return {"conversation_summary": ""}
    msgs = [m.content for m in state["messages"][-6:]]
    conversation = "\n".join(msgs)
    response = llm.invoke([SystemMessage(content="Resume la charla de forma breve."), HumanMessage(content=conversation)])
    return {"conversation_summary": response.content}

def rewrite_query(state: State):
    """Analiza la pregunta y decide cuántos agentes disparar."""
    last_message = state["messages"][-1].content
    llm_estructurado = llm.with_structured_output(QueryAnalysis)
    
    # El LLM usa el PROMPT_REESCRITURA restrictivo que definimos
    response = llm_estructurado.invoke([
        SystemMessage(content=PROMPT_REESCRITURA),
        HumanMessage(content=last_message)
    ])
    
    # CASO 1: La pregunta es clara y generó sub-preguntas de búsqueda
    if response.questions and response.is_clear:
        print(f"🧠 [MAESTRO] Razonamiento: {response.reasoning}")
        print(f"✅ [GRAFO MAESTRO] Consulta: '{last_message}' -> {len(response.questions)} sub-preguntas.")
        return {
            "questionIsClear": True, 
            "originalQuery": last_message, 
            "rewrittenQuestions": response.questions,
            "main_reasoning": response.reasoning # Guardamos el porqué de la decisión
        }
    
    # CASO 2: La pregunta es ambigua o irrelevante para el PDF
    print(f"⚠️ [GRAFO MAESTRO] Pregunta no clara. Solicitando aclaración.")
    return {
        "questionIsClear": False, 
        "main_reasoning": response.reasoning,
        "messages": [AIMessage(content=response.clarification_needed or "No entiendo la consulta. ¿Podrías ser más específico sobre qué buscas en el documento?")]
    }

def route_after_rewrite(state: State):
    """Lógica de ruteo paralelo (Fan-out) con identificación de Agentes."""
    if not state.get("questionIsClear"):
        return "request_clarification"
    
    preguntas = state.get("rewrittenQuestions", [])
    # Enviamos el índice y reseteamos el contador de herramientas para cada sub-agente
    return [Send("agent", {
        "question": q, 
        "question_index": i,
        "tool_call_count": 0 
    }) for i, q in enumerate(preguntas)]

def aggregate_answers(state: State):
    """Une las respuestas de los sub-agentes en una sola respuesta natural."""
    # Ordenar por el índice original para mantener coherencia
    respuestas_ordenadas = sorted(state["agent_answers"], key=lambda x: x['index'])
    formatted = "\n".join([f"- {a['answer']}" for a in respuestas_ordenadas])
    
    prompt = f"Sintetiza esta información para responder a: {state['originalQuery']}\n\nDatos:\n{formatted}"
    res = llm.invoke([HumanMessage(content=prompt)])
    return {"messages": [AIMessage(content=res.content)]}