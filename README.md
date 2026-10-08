# Multi-Agent RAG with MLOps Tracking (Brazilian Constitution CHAT)

This repository contains a study project designed to explore the integration of Multi-Agent AI systems, Retrieval-Augmented Generation (RAG), and MLOps practices. It simulates juridical document analysis tool, through RESTful API and an interactive web interface.

## Interface
<img width="478" height="315" alt="image" src="https://github.com/user-attachments/assets/206d632d-8def-4b9e-a26d-4340c5995084" />

## Overview

The current system ingests the embeddings of an already processed documentation, and delegates user queries to a team of autonomous AI agents. The agents are strictly constrained to answer based **only** on the retrieved context, mitigating hallucinations. Every execution is tracked via an MLOps server to monitor latency, parameters, and outputs.

**Knowledge Base:** The static dataset currently loaded into the vector database is the **Brazilian Constitution of 1988** (Constituição da República Federativa do Brasil de 1988).

## Architecture & Tech Stack

The architecture is decoupled into three main layers: Presentation, Engine/API, and Observability.

*   **Data Engineering & Retrieval:** `FAISS`, `LangChain`, `HuggingFace Embeddings` (`all-MiniLM-L6-v2`)
*   **AI Orchestration:** `CrewAI`, `Google Gemini 1.5 Flash`
*   **Backend API:** `FastAPI`, `Uvicorn`, `Pydantic`
*   **Frontend UI:** `Streamlit`
*   **MLOps & Telemetry:** `MLflow`

## Features

*   **Multi-Agent Workflow:** Utilizes a *Researcher Agent* equipped with a custom FAISS search tool to extract context, and an *Analyst Agent* to synthesize the final markdown response.
*   **Strict Grounding:** Agents are prompted with strict rules to refuse answers if the context is missing, preventing reliance on parametric memory.
*   **RESTful Backend:** Asynchronous API built with FastAPI, enforcing input/output schemas via Pydantic.
*   **Session-aware UI:** Interactive chat interface built with Streamlit that communicates directly with the FastAPI backend.
*   **Run Tracking:** MLflow context managers wrap the AI execution to automatically log the LLM model version, temperature, input queries, system latency, and final outputs.

## 📂 Project Structure

```text
multiagent-rag-mlops/
├── api/
│   ├── main.py             # FastAPI server and endpoints
│   └── agents.py           # CrewAI logic, Tools, and MLflow tracking
├── app/
│   └── frontend.py         # Streamlit chat interface
├── data/
│   └── faiss_index/        # Serialized FAISS vector database and embeddings
├── requirements.txt        # Project dependencies
└── README.md
