import os, textwrap, json
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import NLTKTextSplitter
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import SpacyTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
import nltk
import tiktoken
from pgvector.psycopg import register_vector

MODEL = "text-embedding-3-small"

# https://huggingface.co/thenlper/gte-large
embedding_model = SentenceTransformer("thenlper/gte-large")

def create_vectors(prompt):
    result = embedding_model.encode(prompt)

    return result.tolist()

def tiktoken_length(text):
    encoding = tiktoken.encoding_for_model(MODEL)
    num_tokens = len(encoding.encode(text))

    return num_tokens

# split on character, e.g. '.'
def naive_chunking(text):
    doc = text.split(".")

    return doc

# split by fixed number of characters
def textwrap_chunking(text):
    doc = textwrap.wrap(
        text, 
        2500, 
        replace_whitespace=False
        )

    return doc

# langchain character text splitter
def fixed_sized_chunking(text):

    # get number of tokens
    tiktoken_len = tiktoken_length(text)

    text_splitter = CharacterTextSplitter(
        separator = "\n",
        chunk_size = 1024,
        chunk_overlap  = 20,    
        length_function=tiktoken_length,
        is_separator_regex=False
        )
    data = text_splitter.create_documents([text])

    # text splitter returns langchain Document class
    # reformat into an array of strings
    doc =[]
    for d in data:
        d = d.page_content
        doc.append(d)

    return doc

# langchain text splitter
def NLTK_chunking(text):
    nltk.download('punkt')
    text_splitter = NLTKTextSplitter()
    doc = text_splitter.split_text(text)

    return doc

# langchain semantic text splitter
# https://python.langchain.com/docs/modules/data_connection/document_transformers/split_by_token#spacy
def spaCy_chunking(text):
    text_splitter = SpacyTextSplitter(chunk_size=2000)
    doc = text_splitter.split_text(text)

    return doc

# langchaind recursive character text splitter
# https://python.langchain.com/docs/modules/data_connection/document_transformers/recursive_text_splitter
def recursive_chracter_splitter_chunking(text):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=20,
        length_function=tiktoken_length,
        is_separator_regex=False,
    )

    data = text_splitter.create_documents([text])
    
    # text splitter returns langchain Document class
    # reformat into an array of strings
    doc =[]
    for d in data:
        d = d.page_content
        doc.append(d)

    return doc

def create_embbeddings(directory, files, chunking_method):
    data = []
    for filename in files:
        file_path = os.path.join(directory,filename)
        with open(file_path, 'r', errors="replace") as file:
            document_text = file.read()
            
            # choose the chunking method
            match chunking_method:
                case "naive_chunking":
                    chunks = naive_chunking(document_text)
                case "textwrap_chunking":
                    chunks = textwrap_chunking(document_text)    
                case "fix_sized_chunking":
                    chunks = fixed_sized_chunking(document_text)
                case "NLTK_chunking":
                    chunks = NLTK_chunking(document_text)
                case "spaCY_chunking":
                    chunks = spaCy_chunking(document_text)
                case "recursive_character_splitter_chunking":
                    chunks = recursive_chracter_splitter_chunking(document_text)
                case default:
                    chunks = naive_chunking(document_text)        
            
            id = 0
            for chunk in chunks:
                chunk = chunk.replace('\n',' ')
                chunk = chunk.replace("\x00", "")
                id = id + 1
                embedding = create_vectors(chunk)
                pc_list = [filename, chunk, embedding]
                data.append(pc_list)

    return data


input_directory = "./data"
file_list = sorted(os.listdir(input_directory))
# file_list = file_list[:1]


chunking_method = "recursive_character_splitter_chunking"

embeddings = create_embbeddings(input_directory, file_list, chunking_method)

with open("embeddings.json", "w") as outfile:
    json.dump(embeddings, outfile)
