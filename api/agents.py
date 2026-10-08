import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import BaseTool
from pydantic import BaseModel, Field # Data validation in Python type hints
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# MLOps
import time
import mlflow

load_dotenv() # loads environment variables (gemini api key)

# ============== Tools definition
class FaissSearchInput(BaseModel):
    query: str = Field(description="A pergunta exata a ser pesquisada no banco de dados.")

class FaissSearchTool(BaseTool):
    """
    A BaseTool from crewai is used to define a function that an agent can use. 
    It receives the name and the description of the tool inside the prompt, 
    and decides if it needs it or not.
    """
    name: str = "Buscador_Documental"
    description: str = "Busca contexto no banco de dados interno. Recebe uma string e retorna trechos de texto."
    args_schema: type[BaseModel] = FaissSearchInput

    def _run(self, query: str) -> str:
        # Same embedding model used in Databricks
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        caminho_indice = os.path.join(os.path.dirname(__file__), "../data/faiss_index")
        
        vectorstore = FAISS.load_local(
            folder_path=caminho_indice, 
            embeddings=embeddings, 
            allow_dangerous_deserialization=True
        )
        
        # The similarity is measured by L2 distance
        resultados = vectorstore.similarity_search(query, k=3)
        contexto = "\n---\n".join([doc.page_content for doc in resultados])
        return contexto if contexto else "Nenhum contexto encontrado."

ferramenta_busca = FaissSearchTool()

# ============== Agents definition

gemini_llm = LLM(
    model=os.getenv("LLM_MODEL_NAME", "gemini/gemini-1.5-flash-latest"), # fast, light and free
    temperature=0.1 

    # Temperature is the name of a term used Softmax in token scores.
    # Each logit is divised by the temperature. 
    # T < 1 increases logit differences, so the most problable token becomes much more probable.
    # T = 1 is the default model setting. 
    # T > 1 decreases the difference, so the model can choose less probable tokens. 

    # For a RAG, we don't want it to create content, so we set T = 0.1
)

agente_pesquisador = Agent(
    role="Engenheiro de Recuperação de Dados",
    goal="Buscar com precisão as informações solicitadas nos documentos internos.",
    backstory="Você é um especialista em busca vetorial. Sempre usa a ferramenta Buscador_Documental para encontrar o contexto antes de responder.",
    tools=[ferramenta_busca], # Tells him that he can use this function
    llm=gemini_llm,
    verbose=True, # True = More exec logs 
    allow_delegation=False # True = Can delegate its tasks to other agents
)

agente_revisor = Agent(
    role="Analista de Síntese",
    goal="Sintetizar dados brutos em respostas claras, estruturadas e sem alucinações.",
    backstory="Você é um auditor rigoroso. Você NUNCA utiliza conhecimento prévio, memória ou dados externos. Se o contexto recebido da busca for vazio ou insuficiente para responder à pergunta, você DEVE retornar a string exata: 'A informação não consta na base de dados interna.' e encerrar.",
    llm=gemini_llm,
    verbose=True,
    allow_delegation=False
)

# ================== Task definition
def exect_multiagent_search(pergunta_usuario: str) -> str:
    """
    Receives a question (string) and controls de agents. 
    Add logs to MLFlow and returns answer.
    """

    tarefa_pesquisa = Task(
        description=f"O usuário perguntou: '{pergunta_usuario}'. Extraia o contexto necessário dos documentos.",
        expected_output="Um compilado dos trechos mais relevantes.",
        agent=agente_pesquisador
    )

    tarefa_sintese = Task(
        description="Analise o texto recuperado e responda à pergunta original do usuário.",
        expected_output="Resposta em Markdown. Se não achar, diga explicitamente.",
        agent=agente_revisor
    )

    equipe = Crew(
        agents=[agente_pesquisador, agente_revisor],
        tasks=[tarefa_pesquisa, tarefa_sintese],
        process=Process.sequential
    )

    # MLflow Tracking
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("Sistema_Multiagente_RAG")

    with mlflow.start_run() as run:
        mlflow.log_param("modelo_llm", os.getenv("LLM_MODEL_NAME", "gemini/gemini-1.5-flash-latest"))
        mlflow.log_param("temperatura", 0.1)
        mlflow.log_param("pergunta_usuario", pergunta_usuario)
        
        start_time = time.time()
        resultado = equipe.kickoff()
        latencia = time.time() - start_time
        
        mlflow.log_metric("latencia_segundos", latencia)
        mlflow.log_text(resultado.raw, "resposta_final.txt")
        
        return resultado.raw