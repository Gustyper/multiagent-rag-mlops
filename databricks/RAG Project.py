# Databricks notebook source
# DBTITLE 1,Installing Dependencies
# MAGIC %pip install langchain-community langchain-huggingface pypdf faiss-cpu sentence-transformers
# MAGIC dbutils.library.restartPython() # Isso garante que o cluster carregue as bibliotecas instaladas sem precisar reiniciar

# COMMAND ----------

# DBTITLE 1,Processing text embeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

pdf_path = "/Volumes/workspace/default/raw_documents/constituicao.pdf" 
db_persist_directory = "/Volumes/workspace/default/raw_documents/faiss_index"

loader = PyPDFLoader(pdf_path)
docs = loader.load()
print(f"Loading: {len(docs)} pages.")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
print(f"Doc split in {len(splits)} chunks.")

# Starting model
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Saving the vectors in a base
vectorstore = FAISS.from_documents(
    documents=splits,
    embedding=embeddings
)

vectorstore.save_local(db_persist_directory)

print(f"Embeddings base created at: {db_persist_directory}")

# COMMAND ----------

# DBTITLE 1,Testing
loaded_vectorstore = FAISS.load_local(
    folder_path=db_persist_directory, 
    embeddings=embeddings,
    allow_dangerous_deserialization=True 
)

query = "Todos os cidadãos tem direito ao voto?"
resultados = loaded_vectorstore.similarity_search(query, k=3)

for i, doc in enumerate(resultados):
    print(f"\n--- Resultado {i+1} ---")
    print(doc.page_content)