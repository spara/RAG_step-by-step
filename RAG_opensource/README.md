# RAG Step-by-Step with Open Source

This is a working example of [Frank Denneman's](https://www.linkedin.com/in/frankdenneman/) article [RAG Architecture Deep Dive](https://www.linkedin.com/pulse/rag-architecture-deep-dive-frank-denneman-4lple/) which defines the Load-Transform-Embed-Store workflow. For building RAG applications.

Examples for [RAG Step-by-Step](https://dev.to/spara_50/rag-step-by-step-3fof).

- create_embeddings.py: splits the transcripts in chunks and creates vectors from the data
- insert_embeddings.py: creates a Pincone index and upserts the embeddins to a serverless vector database
- app.py: a Streamlit client for querying the Pinecone database and prompting OpenAI

## Requirements

This example uses [Ollama running the llama2 LLM model](https://ollama.com/) and PostgreSQL with the [pgvector extension](https://github.com/pgvector/pgvector). These can be installed locally.

[localstack_ai] is a containerized environment with both Ollama and PostgreSQL and pgvector. To use localstack_ai, [Docker](https://www.docker.com/) and [docker compose](https://docs.docker.com/compose/install/) are required.

To start localstack_ai:

```bash
git clone https://github.com/spara/localstack_ai.git
cd ./localstack_ai
docker compose -f ollama_stack.yml up
cd ./ollama
ollama pull llama2
```

## Running the examples

1. Clone the repository and install the packages:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. Get data. A [script](./get_transcripts.py) for downloading YouTube video transcripts is available, but any set of plain text documents will also work with this example.

3. Run `python3 create_embeddings.py` to parse the text documents and create embeddings. Change the directory to the location of the text files and the name of the output file.

4. Create a PostgreSQL database. Use either [pgsql](https://docs.risingwave.com/docs/current/install-psql-without-postgresql/) or a client such as [pgAdmin](https://www.pgadmin.org/) or [DBeaver](https://dbeaver.io/). The example uses a database called `items`.

5. Run `python3 insert_embeddings.py` to insert the records into PostgreSQL.

6. Run `streamllit run app.py` to run the client. Note that the `database_search` function in the application includes queries for cosine similarity, L2 distance, and inner product [metrics](https://www.imaurer.com/which-vector-similarity-metric-should-i-use/). To experiment with similarity metrics, uncomment the chosen metric and comment the other metrics.


