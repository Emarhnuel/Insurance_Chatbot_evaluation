import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.prompts.chat import (ChatPromptTemplate,
                                    HumanMessagePromptTemplate,
                                    SystemMessagePromptTemplate)
from langchain.retrievers import MultiQueryRetriever
from langchain.chains import RetrievalQA

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = openai_api_key

# Define the chatbot function
def main():
    # Streamlit app setup
    st.title("Churchill Insurance Chatbot")
    st.write("Ask questions about your insurance policy!")

    # Load and process the document
    markdown_path = "Data/OUTPUT/policy-booklet-0923/policy-booklet-0923.md"
    loader = UnstructuredMarkdownLoader(markdown_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    documents = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    vector_store = FAISS.from_documents(documents, embeddings)
    retriever = vector_store.as_retriever()

    # Define the prompt template
    messages = [
        SystemMessagePromptTemplate(
            content="You are a helpful and informative chatbot about Churchill Insurance policies. "
                    "Use your knowledge to answer the user's questions based on the provided context."
        ),
        HumanMessagePromptTemplate(content="{question}"),
    ]

    # Create the prompt
    prompt = ChatPromptTemplate.from_messages(messages)

    # Set up the LLM and retrieval chain
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    # Create the chain
    retriever_from_llm = MultiQueryRetriever.from_llm(retriever, llm=llm)
    qa_chain = RetrievalQA.from_chain_type(retriever_from_llm, prompt)

    # Streamlit input and interaction
    user_input = st.text_input("Enter your question here:")
    if user_input:
        response = qa_chain(user_input)
        st.write(response)


if __name__ == "__main__":
    main()
