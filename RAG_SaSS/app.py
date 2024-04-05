import json, os
from pinecone import Pinecone
from openai import OpenAI
import streamlit as st

pc = Pinecone(api_key="my_pinecone_api_key")
index = pc.Index("rag_example")

client = OpenAI(
    api_key="my_openai_api_key"
)
MODEL = "text-embedding-3-small"

# creates embeddings
def get_embeddings(text,model="text-embedding-3-small"):
   response = client.embeddings.create(input=text, model=model)
   
   return response.data[0].embedding

# queries the vector database
def semantic_search(query):
    # create embedding from query
    embeddings = get_embeddings(query)

    # search the database
    response = index.query(
        vector=embeddings,
        top_k=5,
        include_metadata=True
    )

    # extract text from results
    results_text = [r['metadata']['text'] for r in response['matches']]

    return results_text

# create a prompt for OpenAI
def create_prompt(llm_request, vector_results):
    prompt_start = (
        "Answer the question based on the context below.\n\n"+
        "Context:\n"
    )

    prompt_end = (
        f"\n\nQuestion: {llm_request}\nAnswer:"
    )

    prompt = (
        prompt_start + "\n\n---\n\n".join(vector_results) + 
        prompt_end
    )
    return prompt

# create a completion
def create_completion(prompt):
    res = client.completions.create(
        model="gpt-3.5-turbo-instruct",
        prompt=prompt,
        temperature=0,
        max_tokens=1000,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None
    )

    return res.choices[0].text

# user interface
with st.form("prompt_form"):
    result =""
    prompt = ""
    semantic_query = st.text_area("Database prompt:", None)
    llm_query = st.text_area("LLM prompt:", None)
    submitted = st.form_submit_button("Send")
    if submitted:
        vector_results = semantic_search(semantic_query)
        prompt = create_prompt(llm_query,vector_results)
        result = create_completion(prompt)
    
    e = st.expander("LLM prompt created:")
    e.write(prompt)
    st.write(result)