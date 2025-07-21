# from langchain.document_loaders import DirectoryLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
# from langchain.embeddings import OpenAIEmbeddings
# Using a simple TF-IDF based embedding as a fallback
from langchain_core.embeddings import Embeddings
from sklearn.feature_extraction.text import TfidfVectorizer
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
import os
import shutil
import pickle
from typing import List

class SimpleTFIDFEmbeddings(Embeddings):
    """Simple TF-IDF based embeddings for demo purposes"""
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=384, stop_words='english')
        self.fitted = False
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not self.fitted:
            self.vectorizer.fit(texts)
            self.fitted = True
        vectors = self.vectorizer.transform(texts)
        return vectors.toarray().tolist()
        
    def embed_query(self, text: str) -> List[float]:
        if not self.fitted:
            # If not fitted yet, fit with the query text
            self.vectorizer.fit([text])
            self.fitted = True
        vector = self.vectorizer.transform([text])
        return vector.toarray()[0].tolist()

# Load environment variables. Assumes that project contains .env file with API keys
load_dotenv()
# The OpenAI API key will be automatically loaded from the environment variable OPENAI_API_KEY

CHROMA_PATH = "chroma"
DATA_PATH = "data/books"


def main():
    generate_data_store()


def generate_data_store():
    documents = load_documents()
    chunks = split_text(documents)
    save_to_chroma(chunks)


def load_documents():
    loader = DirectoryLoader(DATA_PATH, glob="*.md")
    documents = loader.load()
    return documents


def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    document = chunks[10]
    print(document.page_content)
    print(document.metadata)

    return chunks


def save_to_chroma(chunks: list[Document]):
    # Clear out the database first.
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    # Create a simple TF-IDF based embedding function
    embeddings = SimpleTFIDFEmbeddings()

    # Create a new DB from the documents.
    db = Chroma.from_documents(
        chunks, embeddings, persist_directory=CHROMA_PATH
    )
    db.persist()
    
    # Save the fitted vectorizer for use during querying
    with open(f"{CHROMA_PATH}/vectorizer.pkl", "wb") as f:
        pickle.dump(embeddings.vectorizer, f)
    
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")


if __name__ == "__main__":
    main()
