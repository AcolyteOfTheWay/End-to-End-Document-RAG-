# End-to-End Document RAG

This project implements a full **Document Retrieval-Augmented Generation (RAG)** pipeline. Given a user’s query and a set of documents (PDFs, text, etc.), the system retrieves relevant contexts and then uses a generative model to produce a response grounded in those documents. It includes data ingestion, preprocessing, vector embeddings, retrieval, generation, user interface, and evaluation.

---

## Table of Contents

* [Features](#features)
* [Project Structure](#project-structure)
* [Installation](#installation)
* [Usage](#usage)

  * [Preprocessing Documents](#preprocessing-documents)
  * [Indexing & Embeddings](#indexing--embeddings)
  * [Running the Streamlit App](#running-the-streamlit-app)
  * [Querying & Generation](#querying--generation)
* [Configuration](#configuration)
* [Dependencies](#dependencies)
* [Data](#data)
* [How it Works (Architecture)](#how-it-works-architecture)
* [Limitations & Possible Improvements](#limitations--possible-improvements)
* [License](#license)

---

## Features

* Accepts PDFs (and potentially other document formats) as input.
* Document parsing / text extraction.
* Splitting documents into “chunks” / passages to facilitate retrieval.
* Embedding generation for document chunks with a HuggingFaceEmbedding vector embedding model.
* Vector store / similarity search over chunk embeddings with FAISS.
* LLM for producing answers from retrieved contexts.
* A user-facing interface (Streamlit) to input queries and see responses.
* Configuration to allow easy adjustment of model, chunk size, etc.

---

## Project Structure

Here’s a breakdown of the major files and directories:

```
/ (root)
│
├─ Data/                  
│   ├─ raw/               ← Original documents (PDFs etc.)
│   └─ urls/              ← raw urls
│
├─ src/
│   ├─ document_ingestion.py  ← Code to load, parse, and clean documents
│   ├─ graph_builder.py    ← Definition of StateGraph with LangGraph
│   ├─ nodes.py        ← States the nodes for regular RAG and ReAct tools RAG
│   ├─ state.py        ← Fetches current and/or updated RAG state
│   ├─ vectorstore.py        ← FAISS vector stores and embeddings
│   └─ config.py            ← API, model, hyperparameters
│
├─ streamlit_app.py        ← Entry point for the Streamlit-based UI
├─ main.py                 ← Script for running pipeline end-to-end (preprocess, index, query etc.)
├─ requirements.txt        ← Python package dependencies
├─ config/                 ← Configuration files (settings, model choices, hyperparameters)
├─ README.md               
└─ .gitignore, etc.
```

---

## Installation

These steps assume you have **Python 3.12+** installed.

1. Clone the repository

   ```bash
   git clone https://github.com/AcolyteOfTheWay/End-to-End-Document-RAG.git
   cd End-to-End-Document-RAG
   ```

2. Create and activate a virtual environment

   ```bash
   python -m venv venv
   source venv/bin/activate      # On Windows: venv\Scripts\activate
   ```

3. Install dependencies

   ```bash
   pip install -r requirements.txt
   ```

4. (Optional) If you have GPU and want to use it with certain embedding / large model libraries, ensure relevant toolkits (like CUDA, PyTorch with GPU support) are set up.

---

## Usage

Below are typical workflow.

### Preprocessing Documents

* Place raw documents (PDFs, text files) in `Data`.

### Indexing & Embeddings

* HuggingFaceEmbeddings create the embeddings and vectors.
* The embeddings get stored
* Vector store / index (FAISS)
  
### Running the Streamlit App

* Run the UI:

  ```bash
  streamlit run streamlit_app.py
  ```
* In the UI, you can enter a query and view the generated answer along with retrieved document contexts.

### Querying & Generation

* The system retrieves top-k relevant chunks for the user’s query using the embedding + vector store.
* These contexts are passed to a generative model (specified via config) which produces a response.
* Options for the generative model (temperature, max tokens etc.) are adjustable in config.

---

## Configuration

Key configurable components:

* **Embedding model**: Which model is used to embed text (e.g. sentence-transformers, OpenAI embeddings etc.).
* **Chunk size / overlap**: How big each document chunk is, and how much overlap between chunks.
* **Top-k retrieval**: How many context chunks to fetch for each query.
* **Generation model & parameters**: Which LLM is used, its prompt template, temperature, max tokens, etc.
* **Paths / directories**: Where raw data, processed data, embeddings, and vector store are stored.

Configuration is done via files in `config/` directory and/or environment variables.

---

## Dependencies

Some of the main Python packages used:

* `streamlit` — for UI
* Embedding library (e.g. `sentence-transformers` or `openai-embeddings`)
* Model (e.g. `huggingface`, `openai`, `gemini` etc.)
* Vector store / similarity search tool (e.g. `faiss`)
* PDF/text parsing tools (`pypdf')
* Tools: `Retriever`, `Wikipedia`.

All required versions are listed in `requirements.txt`.

---

## Data

* The directory `Data/` contains the original documents to be used.
* Document ingestion and processing is done by `document_ingestion`.
* Embeddings + vector store data are stored in `vectorstore`.

If you want to use your own documents, follow the same folder conventions:

1. Put raw documents into `Data/`.
2. Run preprocessing + embedding pipeline to populate `vectorstore` then build `graph_builder` which uses `nodes`.

---

## How It Works (Architecture)

Here’s the high-level flow of what the system does:

1. **Document Ingestion**: Read documents from disk, parse PDF or text files, extract text.
2. **Text Cleaning & Splitting**: Normalize text (remove boilerplate, non-text stuff), then split long documents into chunks (based on a configurable size and overlap).
3. **Embedding Generation**: For each chunk, generate a vector embedding.
4. **Vector Store / Indexing**: Store chunk embeddings with metadata in a vector store for similarity search.
5. **Query Handling**: When user submits a query:
   a. Embed the query using same embedding model.
   b. Search vector store for top-k similar chunks.
6. **Answer Generation**: Provide the query + retrieved contexts to a generative model to produce an answer.
7. **UI + Interaction**: Through Streamlit app, let user ask queries and display both the answer and sources.

---
## Improvements (Most Relevant)

1. Hybrid retrieval: combine semantic embeddings + keyword search, Web search.
2. Query reformulation / expansion for better recall
3. Better ranking / re-ranking of retrieved chunks to reduce noise
4. Source attribution & more robust verification to reduce hallucinations
5. Support for multimodal content (images, tables, diagrams)



