import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from src.helpers import get_pdf, clean_docs, split_docs, get_embeddings
from langchain_pinecone import PineconeVectorStore

# Load environment variables from a .env file
load_dotenv()

PINECONE_API_KEY= os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY= os.getenv("OPENAI_API_KEY")

# Load and process documents
docs= get_pdf("data/")
cleaned_docs= clean_docs(docs)
splitted_docs= split_docs(cleaned_docs)
embedding_model= get_embeddings()

# create pinecone index and initialize pinecone client
# define index name
index_name= "medical-assistant"

# initialize pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

# define serverless specification
spec = ServerlessSpec(
    cloud="aws", region="us-east-1"
    )

# create index if it does not exist
if not pc.has_index(index_name):
    pc.create_index(
        name=index_name, 
        dimension=384, 
        metric="cosine", 
        spec=spec
        )

# connect to the index  
index= pc.Index(index_name)

# create pinecone vector store from splitted documents
doc_vector_store= PineconeVectorStore.from_documents(
    index_name= index_name,
    embedding= embedding_model,
    documents= splitted_docs
    )