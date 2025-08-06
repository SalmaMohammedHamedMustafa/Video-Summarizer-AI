import os
from typing import List
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

import os
import warnings

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Suppress Python warnings
warnings.filterwarnings('ignore')



class QuestionAnswerer:
    def __init__(self, vector_store_path: str):
        # Set up embeddings and vector store
        self.embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
        self.vector_store = FAISS.load_local(
            vector_store_path, 
            self.embeddings, 
            allow_dangerous_deserialization=True  
        )

        # Set up Gemini model
        GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.2,
            convert_system_message_to_human=True,
            google_api_key=GOOGLE_API_KEY
        )

        self.system_prompt = """
        You are an expert AI assistant helping users understand the content of a technical video.
        Use the provided context to answer the user’s question clearly and concisely.
        If the context is insufficient, say so.
        """

    def ask(self, question: str, k: int = 4) -> str:
        # Retrieve relevant context chunks
        docs = self.vector_store.similarity_search(question, k=k)
        context = "\n\n".join([doc.page_content for doc in docs])

        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=f"""
Context:
\"\"\"{context}\"\"\"

Question:
{question}
""")
        ]

        # Get answer from Gemini
        response = self.llm.invoke(messages)
        return response.content
