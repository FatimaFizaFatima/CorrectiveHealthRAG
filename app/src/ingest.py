from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path
from langchain_core.documents import Document

PDF_DIR = Path('data/source_docs/')

def load_pdfs(pdf_path : Path) -> list[Document]:

    pdf_loader = PyPDFLoader(file_path=pdf_path)

    pdf_document = pdf_loader.load()

    return pdf_document

def load_all_files()->list[Document]:

    all_docs = []

    pdf_files = PDF_DIR.glob('*.pdf')

    for pdf_path in pdf_files:

        pages = load_pdfs(pdf_path)

        all_docs += pages

    return all_docs    
       
def split_docs(docs : list[Document]) -> list[Document]:

    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size = 700,
        chunk_overlap = 100
    )

    chunks = splitter.split_documents(docs)
    return chunks

if __name__ == "__main__":

    docs = load_all_files()

    print(f"The total pages are {len(docs)}")

    chunks = split_docs(docs)

    print(f"The Total no. of chunks are {len(chunks)}")

    print(chunks[0].metadata)
    print(chunks[0].page_content[:200])

        