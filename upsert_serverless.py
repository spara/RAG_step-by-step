import json
from pinecone import Pinecone
from pinecone import ServerlessSpec, PodSpec

f = open('vector_store.json', "r")
embeds = json.load(f)

pc = Pinecone(api_key="b2effeff-6298-4e40-afe5-e1c5feaf86ce")


index_name = "serverless-future-ai"

if index_name not in pc.list_indexes().names():
    pc.create_index(
    name=index_name,
    dimension=1536,
    metric="cosine",
    spec=ServerlessSpec(
        cloud="aws",
        region="us-west-2"
    )
)
    
index = pc.Index(index_name)
index.describe_index_stats()

index.upsert(
    vectors=  embeds
)

index.describe_index_stats()