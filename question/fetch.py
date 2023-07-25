import os
from typing import Any

import streamlit as st 

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.vectorstores import Pinecone
import pinecone

pinecone.init(
    api_key=os.environ["PINECONE_API_KEY"],
    environment=os.environ["PINECONE_ENVIRONMENT_REGION"],
)

def make_qa(llm, retriever):
    # prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

    # {context}

    # Question: {question}
    # Answer:"""
    # prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    # chain_type_kwargs = {"prompt": prompt}
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
    )
    return qa

def get_answer(retrieval_qa, question):
    # Get the answer from the chain
    res = retrieval_qa({"query": question})
    answer, docs = res['result'], res['source_documents']

    return question, answer, docs

embeddings = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])
docsearch = Pinecone.from_existing_index(
    index_name="langchain-doc-index", embedding=embeddings
)
chat = ChatOpenAI(verbose=True, temperature=0)
retriever = docsearch.as_retriever()
qa = make_qa(chat, retriever)

question = st.text_input("Ask something from the file", placeholder="Find something similar to: ....this.... in the text?")    
if question:
    query, answer, docs = get_answer(qa, question)
    st.write(answer)
    
