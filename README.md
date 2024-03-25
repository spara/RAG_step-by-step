# RAG Step-by-Step

This is a working example of [Frank Denneman's](https://www.linkedin.com/in/frankdenneman/) article [RAG Architecture Deep Dive](https://www.linkedin.com/pulse/rag-architecture-deep-dive-frank-denneman-4lple/) which defines the Load-Transform-Embed-Store workflow. For building RAG applications.

Examples for [RAG Step-by-Step](https://dev.to/spara_50/rag-step-by-step-3fof).

- get_transcript.py: retrieves transcripts from Youtube videos
- create_embeddings.py: splits the transcripts in chunks and creates vectors from the data
- upsert-serverless.py: creates a Pincone index and upserts the embeddins to a serverless vector database

To run the exanples:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```