import os
import json
import gradio as gr
from langchain_community.document_loaders import UnstructuredPDFLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA


api_keypath = "api/api.json"
with open(api_keypath, "r") as file:
    api_data = json.load(file)

GROQ_API_KEY = api_data["key"]
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

def load_documents():
    loader = DirectoryLoader("data/", glob="./*.pdf", loader_cls=UnstructuredPDFLoader)
    return loader.load()

def split_text(documents):
    text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=500)
    return text_splitter.split_documents(documents)

def create_vectorstore(text_chunks):
    persist_directory = "doc_db"
    embedding = HuggingFaceEmbeddings()
    vectorstore = Chroma.from_documents(documents=text_chunks, embedding=embedding, persist_directory=persist_directory)
    return vectorstore

def initialize_qa_chain(vectorstore):
    retriever = vectorstore.as_retriever()
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
    return qa_chain

def answer_query(qa_chain, query):
    response = qa_chain.invoke({"query": query})
    return response["result"]

documents = load_documents()
text_chunks = split_text(documents)
vectorstore = create_vectorstore(text_chunks)
qa_chain = initialize_qa_chain(vectorstore)

def gradio_interface(query):
    return answer_query(qa_chain, query)

iface = gr.Interface(
    fn=gradio_interface,
    inputs="text",
    outputs="text",
    live=True,
    title="GODEL AGENT: A SELF-REFERENTIAL FRAMEWORK FOR AGENTS RECURSIVELY SELF-IMPROVEMENT",
    description="Ask questions based on the loaded documents."
)

if __name__ == "__main__":
    iface.launch()
