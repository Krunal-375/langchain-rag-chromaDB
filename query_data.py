import argparse
# from dataclasses import dataclass
from langchain_community.vectorstores import Chroma
from langchain_openai import AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate
from sklearn.feature_extraction.text import TfidfVectorizer
from langchain_core.embeddings import Embeddings
from typing import List
import os
import pickle
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class SimpleTFIDFEmbeddings(Embeddings):
    """Simple TF-IDF based embeddings for demo purposes - must match create_database.py"""
    def __init__(self, vectorizer_path=None):
        if vectorizer_path and os.path.exists(vectorizer_path):
            # Load the saved vectorizer
            with open(vectorizer_path, "rb") as f:
                self.vectorizer = pickle.load(f)
            self.fitted = True
        else:
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

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""


def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text

    # Prepare the DB with the same embedding function used during creation.
    vectorizer_path = f"{CHROMA_PATH}/vectorizer.pkl"
    embedding_function = SimpleTFIDFEmbeddings(vectorizer_path=vectorizer_path)
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_relevance_scores(query_text, k=3)
    if len(results) == 0 or results[0][1] < 0.3:  # Lower threshold for TF-IDF
        print(f"Unable to find matching results.")
        return

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    print(prompt)

    # Use Azure OpenAI for the chat model
    model = AzureChatOpenAI(
        azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_version=os.environ["AZURE_OPENAI_API_VERSION"]
    )
    response_text = model.predict(prompt)

    sources = [doc.metadata.get("source", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)


if __name__ == "__main__":
    main()
