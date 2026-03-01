from langgraph.graph import MessagesState
from pydantic import BaseModel, Field
from typing import List, Annotated, Set
import operator

# Funciones reductoras para manejar cómo se actualiza la memoria
def accumulate_or_reset(existing: List[dict], new: List[dict]) -> List[dict]:
    if new and any(item.get('__reset__') for item in new):
        return []
    return existing + new

def set_union(a: Set[str], b: Set[str]) -> Set[str]:
    return a | b

# --- 1. ESTADO DEL GRAFO PRINCIPAL ---
class State(MessagesState):
    questionIsClear: bool = False
    conversation_summary: str = ""
    originalQuery: str = ""
    rewrittenQuestions: List[str] = []
    agent_answers: Annotated[List[dict], accumulate_or_reset] = []

# --- 2. ESTADO DEL SUBGRAFO (AGENTE) ---
class AgentState(MessagesState):
    tool_call_count: Annotated[int, operator.add] = 0
    iteration_count: Annotated[int, operator.add] = 0
    question: str = ""
    question_index: int = 0
    context_summary: str = ""
    retrieval_keys: Annotated[Set[str], set_union] = set()
    final_answer: str = ""
    agent_answers: List[dict] = []

# --- 3. MODELO DE DATOS ESTRUCTURADO ---
class QueryAnalysis(BaseModel):
    is_clear: bool = Field(description="Indica si la pregunta del usuario es clara y se puede responder.")
    questions: List[str] = Field(description="Lista de preguntas reescritas y auto-contenidas basadas en el historial.")
    clarification_needed: str = Field(description="Explicación de qué información falta si la pregunta no es clara.")

if __name__ == "__main__":
    print("✅ Modelos de estado avanzados definidos correctamente.")