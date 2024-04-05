import numpy as np
from pgvector.psycopg import register_vector
import psycopg
import os, json 

dimensions = 1024

# enable extension
conn = psycopg.connect(dbname='ai_usecases', autocommit=True)
conn.execute('CREATE EXTENSION IF NOT EXISTS vector')
register_vector(conn)

# create table
conn.execute('DROP TABLE IF EXISTS items')
conn.execute(f'CREATE TABLE items (id bigserial, filename varchar(64), chunk text, embedding vector({dimensions}))')

# get data
f = open('embeddings.json', "r")
records = json.load(f)

# insert records to database
for record in records:
    filename = record[0]
    chunk = record[1]
    embedding = record[2]
    conn.execute('INSERT INTO items (filename, chunk, embedding) VALUES (%s, %s, %s)', (filename, chunk, embedding))

# create index
conn.execute('CREATE INDEX ON items USING hnsw (embedding vector_cosine_ops)')
