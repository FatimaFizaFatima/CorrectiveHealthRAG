from langchain_core.tools import tool
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from dotenv import load_dotenv

from app.src.ingest import load_all_files, split_docs
import os

load_dotenv()


def get_file_name(file_path):

    file_name = os.path.basename(file_path)
    source_name, extention = os.path.splitext(file_name)
    return source_name

def build_vectorstore():
    docs = load_all_files()
    chunks = split_docs(docs)

    # embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = InMemoryVectorStore.from_documents(chunks, embeddings)
    return vectorstore

def build_retriever_tool():
    vectorstore = build_vectorstore()
    retriever = vectorstore.as_retriever()

    @tool
    def search_health_docs(query: str) -> str:
        """Search official health information about depression and hypertension.
        Use this for any question about symptoms, causes, diagnosis, or treatment
        of depression or high blood pressure/hypertension."""
        docs = retriever.invoke(query)
        return "\n\n".join(f"Content: {d.page_content} \n PDF Name: {get_file_name(d.metadata['source'])}" for d in docs)

    return search_health_docs

if __name__ == "__main__":
    
    tool = build_retriever_tool()
    result = tool.invoke({"query": "what are the symptoms of depression"})
    print(result[:500])