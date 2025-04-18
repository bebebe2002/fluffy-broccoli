import os
import streamlit as st
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

from PyPDF2 import PdfReader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain

st.title("📄 PDF Chatbot with RAG")

uploaded_file = st.file_uploader("PDFをアップロードしてください", type="pdf")
question = st.text_input("質問を入力してください")

if uploaded_file and question:
    reader = PdfReader(uploaded_file)
    raw_text = ""
    for page in reader.pages:
        text = page.extract_text()
        if text:
            raw_text += text

    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_text(raw_text)

    embeddings = OpenAIEmbeddings()
    db = FAISS.from_texts(texts, embeddings)

    docs = db.similarity_search(question)
    llm = OpenAI(temperature=0)
    chain = load_qa_chain(llm, chain_type="stuff")
    response = chain.run(input_documents=docs, question=question)

    st.write("📌 回答：")
    st.success(response)
