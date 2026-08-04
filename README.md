# RAG with LangChain + IBM Granite

An end-to-end Retrieval Augmented Generation (RAG) pipeline built with LangChain and IBM's Granite models — extended with a RAG-vs-no-RAG comparison, a retrieval confidence check, and a reusable query interface.

## Why this exists

Most RAG tutorials stop at "retrieve, stuff into prompt, generate" and call it done. That leaves two open questions I wanted answered before trusting a pipeline like this:

1. **Is retrieval actually changing the answer?** If the model already knows the answer, RAG isn't adding anything — and without checking, you can't tell "grounded" apart from "lucky."
2. **How do I know retrieval found the right context?** A RAG pipeline can silently retrieve weakly-related chunks and still produce a confident-sounding (wrong) answer.

This project builds on the base [IBM Granite Community](https://github.com/ibm-granite-community) RAG recipe and adds three things to answer those questions directly.

## What's in here

| Addition | What it does | Why it matters |
|---|---|---|
| **RAG vs. no-RAG comparison** | Asks the same question with and without retrieved context | Shows the actual delta retrieval provides, instead of assuming it helps |
| **Retrieval confidence check** | Prints similarity scores for each retrieved chunk | Lets you catch weak/irrelevant retrieval before trusting the answer |
| **`ask()` helper function** | Wraps the pipeline for reuse across questions or documents | Turns a one-shot notebook demo into something you can actually query interactively |

## How the pipeline works

1. **Index** — split source documents into chunks, embed them with `ibm-granite/granite-embedding-small-english-r2`, and store the vectors in a [Chroma](https://www.trychroma.com/) vector database.
2. **Retrieve** — embed the incoming query and pull the most semantically similar chunks.
3. **Generate** — pass the retrieved chunks and the question to `ibm-granite/granite-4.1-8b` (via [Replicate](https://replicate.com/ibm-granite)) to produce a grounded answer.
4. **Verify** — check retrieval scores and compare against the ungrounded baseline before trusting the output.

## Getting started

### 1. Clone the repo

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
```

### 2. Get a Replicate API token

The LLM is served via [Replicate](https://replicate.com/). Create an account, generate an API token, and set it as an environment variable (or Colab secret) named:

```
REPLICATE_API_TOKEN
```

### 3. Run the notebook

Open `RAG_with_Langchain.ipynb` in Colab or Jupyter and run top to bottom. It will install dependencies, build the vector index over a sample document (the 2022 State of the Union address), and walk through retrieval, generation, and the verification steps above.

## Example

```python
print(ask("What did the president say about inflation?", show_sources=True))
```

Output includes both the generated answer and the source chunks it was grounded in, so you can audit the answer instead of taking it on faith.

## Using your own data

Swap the document-download cell for your own source (PDF, text, HTML, etc.) — any LangChain document loader works. The rest of the pipeline, including the verification steps, works unchanged.

## Possible next steps

- Swap Chroma for a persistent/managed vector store for production use.
- Add automated retrieval-quality regression tests against a labeled question set.
- Extend `ask()` with conversation memory for multi-turn Q&A.

## Acknowledgments

Built on the [IBM Granite Community](https://github.com/ibm-granite-community) RAG recipe and the [LangChain](https://python.langchain.com/) framework. Extended with the comparison, scoring, and reusability additions described above.

## License

MIT (or a license of your choice).
