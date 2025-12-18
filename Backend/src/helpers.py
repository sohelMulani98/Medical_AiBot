from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.schema import Document

"""
Helper functions for document processing and embedding generation.

This module provides functions to load PDF documents from a directory,
clean their metadata, split them into smaller chunks, and generate embeddings
using a specified embedding model.

libraries used:
- langchain.document_loaders: For loading PDF documents.
- langchain.text_splitter: For splitting documents into smaller chunks.
- langchain.embeddings: For generating embeddings using HuggingFace models.
- langchain.schema: For defining the Document schema.

"""

#function to load pdf documents from a directory
def get_pdf(data):
    loader= DirectoryLoader(
        data,
        glob="*.pdf",
        loader_cls=PyPDFLoader,
    )

    document= loader.load()

    print(f"Number of documents loaded: {len(document)}")
    return document

#function to clean metadata of documents
def clean_docs(docs):
    cleaned_docs= []

    for doc in docs:
        metadata= doc.metadata["source"]

        cleaned_docs.append(
            Document(
                page_content= doc.page_content,
                metadata= {"source": metadata}
            )
        )
    return cleaned_docs


#function to split documents into smaller chunks
def split_docs(clean_docs):
    text_splitter= RecursiveCharacterTextSplitter(
        chunk_size= 500,
        chunk_overlap= 20,
    )

    splitted_docs= text_splitter.split_documents(clean_docs)

    print(f"Number of split documents: {len(splitted_docs)}")
    return splitted_docs


#function to get embedding model.
#This function initializes and returns a HuggingFaceBgeEmbeddings model using the specified model name.

def get_embeddings():
    model_name= "sentence-transformers/all-MiniLM-L6-v2"
    embedding_model= HuggingFaceBgeEmbeddings(
        model_name= model_name
        )

    return embedding_model

