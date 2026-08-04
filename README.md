# RAG with LangChain + IBM Granite

A minimal, end-to-end example of **Retrieval Augmented Generation (RAG)** built with LangChain and IBM's Granite models. It shows how to ground an LLM's answers in real documents instead of relying purely on what the model memorized during training.

## What is RAG?

Retrieval Augmented Generation improves the accuracy and relevance of language model output by retrieving factual information from a knowledge base and feeding it into the model alongside the user's question — instead of asking the model to answer from memory alone.

Typical use cases:

- **Customer support** — answer product questions using the actual product docs.
- **Domain Q&A** — explore a specialized field (finance, law, medicine) using source papers or articles.
- **Current events** — chat about recent news by retrieving relevant articles at query time.

The pipeline has three stages:

1. **Index** — split source documents into chunks, embed them, and store the vectors in a vector database.
2. **Retrieve** — embed the incoming query and pull the most semantically similar chunks from the database.
3. **Generate** — pass the retrieved chunks and the original question to an LLM to produce a grounded answer.

## Stack

| Component | Choice used in this notebook |
|---|---|
| Embeddings | `ibm-granite/granite-embedding-small-english-r2` (Hugging Face) |
| Vector store | [Chroma](https://www.trychroma.com/) (in-memory) |
| LLM | `ibm-granite/granite-4.1-8b` via [Replicate](https://replicate.com/ibm-granite) |
| Orchestration | [LangChain](https://python.langchain.com/) |

Each component is swappable — see the linked recipes in the notebook for alternative embedding models, vector stores, and LLM providers.

## Getting started

### 1. Clone the repo

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
```

### 2. Set up your environment

This notebook is designed to run in Google Colab or a local Jupyter environment with Python 3.10+.

### 3. Get a Replicate API token

The LLM in this example is served via [Replicate](https://replicate.com/). Create an account, generate an API token, and set it as an environment variable (or Colab secret) named:

```
REPLICATE_API_TOKEN
```

### 4. Run the notebook

Open `RAG_with_Langchain.ipynb` and run the cells top to bottom. The notebook will:

- Install dependencies
- Load an embedding model and vector database
- Download a sample document (the 2022 State of the Union address)
- Chunk and embed the document into the vector store
- Retrieve relevant passages for a sample query
- Generate a grounded answer using the Granite LLM

## Example

```python
query = "What did the president say about Ketanji Brown Jackson?"
output = rag_chain.invoke({"input": query})
print(output["answer"])
```

The model answers using only the passages retrieved from the source document, rather than relying on unverified prior knowledge.

## Swapping in your own data

Replace the document download step with your own source files (PDF, text, HTML, etc.) and adjust the `TextLoader` accordingly. Any LangChain-supported document loader will work here — the rest of the pipeline (chunking, embedding, retrieval, generation) stays the same.

## Acknowledgments

Built on the [IBM Granite Community](https://github.com/ibm-granite-community) recipes and the [LangChain](https://python.langchain.com/) framework.

## License

Add a license of your choice (MIT is a common default for sample projects like this).
