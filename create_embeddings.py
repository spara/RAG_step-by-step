from openai import OpenAI
import os, textwrap, json
import string, random
from langchain.text_splitter import NLTKTextSplitter
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import SpacyTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
import nltk
import tiktoken

client = OpenAI(
    api_key="my_openai_key"
)

MODEL = "text-embedding-3-small"

def tiktoken_length(text):
    encoding = tiktoken.encoding_for_model(MODEL)
    num_tokens = len(encoding.encode(text))

    return num_tokens

# create random  10 character alpha ids for pinecone upsert
def random_id():
    N = 10
    res = ''.join(random.choices(string.ascii_uppercase +
                             string.digits, k=N))
    return res

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
        chunk_overlap  = 200,    
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
        with open(file_path, 'r') as file:
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
            
            for chunk in chunks:
                # chunk = chunk.replace('\n',' ')
                id = random_id()
                embedding = client.embeddings.create(
                    model = MODEL, 
                    input = chunk
                ).data[0].embedding
                pc_dict = {"id" : id , "values" : embedding, "metadata": { "source" : filename, "text" : chunk}}
                data.append(pc_dict)

    return data

input_directory = "./vector_store"
file_list = sorted(os.listdir(input_directory))
file_list = file_list[:1]
MODEL = "text-embedding-3-small"

chunking_method = "recursive_character_splitter_chunking"

embeddings = create_embbeddings(input_directory, file_list, chunking_method)

with open("fixed_size_vector_store.json", "w") as outfile:
    json.dump(embeddings, outfile)