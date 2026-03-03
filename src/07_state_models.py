from langgraph.graph import MessagesState
from pydantic import BaseModel, Field
from typing import List, Annotated, Set, Optional
import operator

# Funciones reductoras
def accumulate_or_reset(existing: List[dict], new: List[dict]) -> List[dict]:
    if new and any(item.get('__reset__') for item in new): return []
    return existing + new

def set_union(a: Set[str], b: Set[str]) -> Set[str]:
    return a | b

# --- 1. ESTADO DEL GRAFO PRINCIPAL ---
class State(MessagesState):
    questionIsClear: bool = False
    originalQuery: str = ""
    rewrittenQuestions: List[str] = []
    main_reasoning: str = "" # RAZONAMIENTO DEL MAESTRO
    agent_answers: Annotated[List[dict], accumulate_or_reset] = []

# --- 2. ESTADO DEL SUBGRAFO (AGENTE) ---
class AgentState(MessagesState):
    tool_call_count: Annotated[int, operator.add] = 0
    question: str = ""
    next_search_query: str = "" # NUEVA PREGUNTA EVOLUCIONADA
    question_index: int = 0
    context_summary: str = ""
    agent_reasoning: str = "" # RAZONAMIENTO DEL AGENTE
    agent_answers: List[dict] = []

# --- 3. MODELO DE DATOS ESTRUCTURADO (Para rewrite_query) ---
class QueryAnalysis(BaseModel):
    is_clear: bool = Field(description="¿Es clara la pregunta?")
    questions: List[str] = Field(description="Sub-preguntas reescritas.")
    reasoning: str = Field(description="Por qué se decidió esta división.")
    clarification_needed: Optional[str] = Field(description="Si no es clara, ¿qué falta?")

if __name__ == "__main__":
    print("✅ Modelos de estado actualizados con campos de razonamiento.")