import streamlit as st
import requests

st.set_page_config(page_title="MultiAgent RAG", page_icon="🤖")
st.title("Sistema de Análise Documental")
st.markdown("Interface conectada à API FastAPI (CrewAI + FAISS + MLflow)")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Faça uma pergunta técnica sobre os documentos..."):
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # FastAPI communication
    with st.chat_message("assistant"):
        with st.spinner("A equipe de agentes está analisando os documentos..."):
            try:
                # Sends the payload to the endpoint 
                url = "http://127.0.0.1:8000/api/v1/chat"
                payload = {"query": prompt}
                
                response = requests.post(url, json=payload)
                response.raise_for_status() # Check HTTP = 200 (OK)
                
                # Extracts the structured response
                dados_api = response.json()
                resposta_texto = dados_api["resposta"]
                
                # Show the LLM answer
                st.markdown(resposta_texto)
                st.session_state.messages.append({"role": "assistant", "content": resposta_texto})
                
            except requests.exceptions.RequestException as e:
                st.error(f"Falha na comunicação com o backend (FastAPI): {e}")