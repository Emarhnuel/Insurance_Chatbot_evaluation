import streamlit as st
import os
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.retrievers import MultiQueryRetriever
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain.prompts import PromptTemplate
from langchain import hub
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv


# Load environment variables from .env file
load_dotenv()

# Set up OpenAI API key
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


# Define the chat_gen class
class chat_gen():
    def __init__(self):
        self.chat_history = []

    def load_doc(self, document_path):
        markdown_path = "Data/OUTPUT/policy-booklet-0923/policy-booklet-0923.md"
        loader = UnstructuredMarkdownLoader(markdown_path)
        documents = loader.load()

        # Split document in chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, separator="\n")
        documents = text_splitter.split_documents(documents)
        embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

        # Create vectors
        self.vector_store = FAISS.from_documents(documents, embeddings)

    def load_model(self):
        llm = ChatOpenAI(model_name="gpt-4o", temperature=0)

        # Define your system instruction
        system_instruction = """
        As an AI assistant, you must answer the query from the user from the retrieved content,
        if no relevant information is available, answer the question by using your knowledge about the topic"""

        # Define your template with the system instruction
        prompt_template = (f"{system_instruction} "
                    "Combine the chat history{chat_history} and follow up question into "
                    "a standalone question to answer from the {context}. "
                    "Follow up question: {question}")

        # Define the prompt template
        prompt = prompt_template.format(context="context", question="question", chat_history="chat_history")
        #prompt = PromptTemplate(template=system_instruction, input_variables=["question", "chat_history", "context"])

        retrieval_qa_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

        retriever = self.vector_store.as_retriever() if hasattr(self, 'vector_store') else None
        if not retriever:
            raise ValueError("Vector store is not initialized. Please run load_doc() first.")

        chain = MultiQueryRetriever.from_llm(
            llm=llm,
            retriever=retriever,
            prompt=prompt,
            # combine_docs_chain_kwargs={'prompt': prompt},
            chain=create_stuff_documents_chain(llm, retrieval_qa_prompt)
        )

        return chain

    def ask_pdf(self, query):
        result = self.load_model()({"question": query, "chat_history": self.chat_history})
        self.chat_history.append((query, result["answer"]))
        # print(result)
        return result['answer']