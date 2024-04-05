from pgvector.psycopg import register_vector
import psycopg
import streamlit as st
from sentence_transformers import SentenceTransformer
import ollama

# embedding model for generaing vectors
embedding_model = SentenceTransformer("thenlper/gte-large")

# database connection
conn = psycopg.connect(dbname='ai_usecases', autocommit=True)

def database_search(query):
    vector = create_vector(query)
    embedding_query = "[" + ",".join(map(str, vector)) + "]"

    # cosine similarity
    # query_sql = f"""
    # SELECT chunk, embedding <=>'{embedding_query}' AS similarity
    # FROM items
    # ORDER BY embedding <=> '{embedding_query}'
    # LIMIT 20;
    # """

    # L2 distance
    # query_sql = f"""
    # SELECT chunk, embedding <-> '{embedding_query}' AS distance
    # FROM items
    # ORDER BY embedding <-> '{embedding_query}'
    # LIMIT 20;
    # """

    # Negative inner product
    query_sql = f"""
    SELECT chunk, embedding <#> '{embedding_query}' AS distance
    FROM items
    ORDER BY embedding <#> '{embedding_query}'
    LIMIT 20;
    """


    data = conn.execute(query_sql).fetchall()
    result=[]
    for row in data:
        result.append(row[0])

    return result

def create_vector(prompt):
    result = embedding_model.encode(prompt)
    return result

def create_prompt(llm_query, database_results):
    content_start = (
        "Answer the question based on the context below.\n\n"+
        "Context:\n"
    )

    content_end = (
        f"\n\nQuestion: {llm_query}\nAnswer:"
    )

    content = (
        content_start + "\n\n---\n\n".join(database_results) + 
        content_end
    )

    prompt = [{'role': 'user', 'content': content }]

    return prompt

def create_completion(prompt):
    completion = ollama.chat(model='llama2', messages=prompt)

    return completion['message']['content']

# ui
with st.form("prompt_form"):
    result =""
    prompt = ""
    semantic_query = st.text_area("Database prompt:", None)
    llm_query = st.text_area("LLM prompt:", None)
    submitted = st.form_submit_button("Send")
    if submitted:
        vector_results = database_search(semantic_query)
        prompt = create_prompt(llm_query,vector_results)
        print(prompt)
        result = create_completion(prompt)
    
    e = st.expander("LLM prompt created:")
    e.write(prompt)
    st.write(result)



