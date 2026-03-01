from typing import Literal
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, RemoveMessage
from langgraph.types import Send
from langchain_ollama import ChatOllama
import os
import sys
import importlib

# Importar dependencias locales (Pasos 7 y 8)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
state_models = importlib.import_module("07_state_models")
agent_config = importlib.import_module("08_agent_config")

State = state_models.State
QueryAnalysis = state_models.QueryAnalysis

# --- LLM LOCAL ---
llm = ChatOllama(model="qwen2.5", temperature=0.1)

# --- PROMPTS INTEGRADOS ---
PROMPT_RESUMEN = "Resume la siguiente conversación de forma concisa. Mantén los detalles clave para dar contexto a futuras preguntas."
PROMPT_REESCRITURA = """Analiza la consulta del usuario basándote en el contexto de la conversación.
Determina si la pregunta es clara por sí sola o si le falta contexto.
Si es clara, reescríbela (pueden ser varias preguntas si pide varias cosas) para que sea perfecta para una búsqueda en base de datos vectorial.
Si no es clara (ej: "explícame más de eso"), explica qué falta en el campo 'clarification_needed'."""
PROMPT_SINTESIS = "Sintetiza las siguientes respuestas en una sola respuesta final y natural en español para el usuario."

# --- NODOS DEL GRAFO PRINCIPAL ---

def summarize_history(state: State):
    if len(state["messages"]) < 4:
        return {"conversation_summary": ""}

    relevant_msgs = [msg for msg in state["messages"][:-1] if isinstance(msg, (HumanMessage, AIMessage)) and not getattr(msg, "tool_calls", None)]
    if not relevant_msgs:
        return {"conversation_summary": ""}

    conversation = "Historial:\n" + "\n".join([f"{'Usuario' if isinstance(m, HumanMessage) else 'Asistente'}: {m.content}" for m in relevant_msgs[-6:]])
    response = llm.invoke([SystemMessage(content=PROMPT_RESUMEN), HumanMessage(content=conversation)])
    return {"conversation_summary": response.content, "agent_answers": [{"__reset__": True}]}

def rewrite_query(state: State):
    last_message = state["messages"][-1]
    conversation_summary = state.get("conversation_summary", "")
    context_section = f"Contexto:\n{conversation_summary}\n\nConsulta Usuario:\n{last_message.content}"

    llm_estructurado = llm.with_structured_output(QueryAnalysis)
    response = llm_estructurado.invoke([SystemMessage(content=PROMPT_REESCRITURA), HumanMessage(content=context_section)])

    if response.questions and response.is_clear:
        delete_all = [RemoveMessage(id=m.id) for m in state["messages"] if not isinstance(m, SystemMessage)]
        return {"questionIsClear": True, "messages": delete_all, "originalQuery": last_message.content, "rewrittenQuestions": response.questions}

    clarification = response.clarification_needed if response.clarification_needed else "Necesito más información para entender tu pregunta."
    return {"questionIsClear": False, "messages": [AIMessage(content=clarification)]}

def request_clarification(state: State):
    return {}

def route_after_rewrite(state: State) -> Literal["request_clarification", "agent"]:
    if not state.get("questionIsClear", False):
        return "request_clarification"
    else:
        return [Send("agent", {"question": query, "question_index": idx, "messages": []}) for idx, query in enumerate(state["rewrittenQuestions"])]

def aggregate_answers(state: State):
    if not state.get("agent_answers"):
        return {"messages": [AIMessage(content="No se encontraron respuestas.")]}

    sorted_answers = sorted(state["agent_answers"], key=lambda x: x.get("index", 0))
    formatted_answers = "\n".join([f"Respuesta {i+1}:\n{ans['answer']}" for i, ans in enumerate(sorted_answers)])
    
    user_message = HumanMessage(content=f"Pregunta original: {state['originalQuery']}\nRespuestas recuperadas:\n{formatted_answers}")
    synthesis_response = llm.invoke([SystemMessage(content=PROMPT_SINTESIS), user_message])
    return {"messages": [AIMessage(content=synthesis_response.content)]}

if __name__ == "__main__":
    print("✅ Nodos del grafo principal configurados correctamente (con dependencias modulares).")
