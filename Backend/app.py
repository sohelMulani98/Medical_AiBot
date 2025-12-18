import os
from flask import Flask, request, jsonify
from langchain_openai import ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompts import *
from src.helpers import get_embeddings

# Load environment variables from a .env file
load_dotenv()

PINECONE_API_KEY= os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY= os.getenv("OPENAI_API_KEY")

#initiate flask application
app = Flask(__name__)

def get_llm():
    llm_model= ChatOpenAI(
        openai_api_key= OPENAI_API_KEY,
        model_name= "gpt-5-nano"
    ) 

    return llm_model

def get_vectors_store(index, embeddings):
    vectore_store= PineconeVectorStore.from_existing_index(
        index_name= index,
        embedding= embeddings
    )

    return vectore_store

#LLM pipeline/chain
def llm_pipeline(llm_model, retriver, query):
    prompt_template= ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("user", "{input}"),
    ]
    )

    retrieval_chain= create_stuff_documents_chain(
    llm= llm_model,
    prompt= prompt_template
    )

    rag_chain= create_retrieval_chain(retriver, retrieval_chain)

    
    response= rag_chain.invoke({"input": query})

    return response["answer"]

@app.route('/ask', methods=['POST'])
def home():
    data= request.get_json()
    question= data['question']
    if not question:
        return jsonify({"Error": "Please ask question to proceed"}), 400
    
    llm_model= get_llm()

    index_name= "medical-assistant"
    embeddings= get_embeddings()

    vector_store= get_vectors_store(index_name, embeddings)

    retriver= vector_store.as_retriever(search_type="similarity", search_kwargs={"k":3})

    answer= llm_pipeline(llm_model, retriver, question)

    return jsonify({"answer": f"{answer}"}), 200


if __name__ == '__main__':
    app.run(debug=True)

