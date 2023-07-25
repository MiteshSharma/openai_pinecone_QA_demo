import os
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
import pinecone
import streamlit as st 

pinecone.init(
    api_key=os.environ["PINECONE_API_KEY"],
    environment=os.environ["PINECONE_ENVIRONMENT_REGION"],
)

loader = TextLoader("./docs/index.txt")
raw_documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100, chunk_overlap=0, separators=["\n\n", "\n", " ", ""]
)
documents = text_splitter.split_documents(documents=raw_documents)

embeddings = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])
Pinecone.from_documents(documents, embeddings, index_name="langchain-doc-index")

st.write("Added to Pinecone")
