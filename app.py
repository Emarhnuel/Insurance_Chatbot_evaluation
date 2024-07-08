import streamlit as st
import os
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.retrievers import MultiQueryRetriever
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
import openai



# Function to initialize or get API key from session state
def get_openai_api_key():
    api_key = st.sidebar.text_input("Enter your OpenAI API Key:", type="password")
    if api_key:
        return api_key
    else:
        return None




# Define prompt for the LLM
prompt = """
You are a helpful and knowledgeable insurance chatbot. 
You have access to a comprehensive insurance policy document. 
Please answer the user's question based on the information provided in the document. 
If the answer is not found in the document, please politely inform the user. 
Here is the user's question: {question}
Here are the relevant documents: {context}
"""

# Set the path to your Markdown file
markdown_path = "Data/OUTPUT/policy-booklet-0923/policy-booklet-0923.md"


def main():
    # Title and description
    st.title("Insurance Policy Chatbot")
    st.write("Your guide to understanding your insurance coverage.")

    # Get API key from sidebar
    openai_api_key = get_openai_api_key()

    # Chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Get user input
    if prompt := st.chat_input("Ask me about your policy"):
        # Append user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        # Check for valid API key before proceeding
        if openai_api_key is None:
            st.error("Please enter your OpenAI API key in the sidebar.")
            return
        try:
            # Embed and store the document chunks using user provided key
            loader = UnstructuredMarkdownLoader(markdown_path)
            documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            documents = text_splitter.split_documents(documents)
            embeddings = OpenAIEmbeddings()
            vector_store = FAISS.from_documents(documents, embeddings)
            retriever = vector_store.as_retriever()
            # Set up LLM and Retriever
            primary_qa_llm = ChatOpenAI(openai_api_key=openai_api_key, model_name="gpt-3.5-turbo", temperature=0)
            advanced_retriever = MultiQueryRetriever.from_llm(retriever=retriever, llm=primary_qa_llm)
            document_chain = create_stuff_documents_chain(primary_qa_llm, prompt)
            retrieval_chain = create_retrieval_chain(advanced_retriever, document_chain)

            # Get chatbot response
            response = retrieval_chain.invoke({"input": prompt})
            answer = response['answer']

            # Append chatbot response to chat history
            st.session_state.messages.append({"role": "assistant", "content": answer})
            # Display chatbot response in chat message container
            with st.chat_message("assistant"):
                st.markdown(answer)
        except Exception as e:
            st.error(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
