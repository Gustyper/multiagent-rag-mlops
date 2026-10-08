from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from agents import exect_multiagent_search 

app = FastAPI(
    title="MultiAgent RAG System API",
    description="Corporate docs consulting endpoint with CrewAI and FAISS.",
    version="1.0.0"
)

# Input Schema
class ChatRequest(BaseModel):
    query: str

# Response Schema
class ChatResponse(BaseModel):
    resposta: str
    status: str

# Endpoint HTTP POST
@app.post("/api/v1/chat", response_model=ChatResponse)
def endpoint_chat(requisicao: ChatRequest):
    try:
        # Gets the query and sends to crewAI
        resultado_texto = exect_multiagent_search(requisicao.query)
        
        # Returns response in JSON
        return ChatResponse(
            resposta=resultado_texto,
            status="sucesso"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))