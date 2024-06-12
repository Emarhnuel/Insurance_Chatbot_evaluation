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


# Set up Streamlit secrets to securely store the OpenAI API key
openai.api_key = st.secrets["API"]["OPENAI_API_KEY"]
#os.environ["OPENAI_API_KEY"] = openai.api_key




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

# Load and process the Markdown file
loader = UnstructuredMarkdownLoader(markdown_path)
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
documents = text_splitter.split_documents(documents)
embeddings = OpenAIEmbeddings()
vector_store = FAISS.from_documents(documents, embeddings)
retriever = vector_store.as_retriever()

# Set up LLM and Retriever
primary_qa_llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
advanced_retriever = MultiQueryRetriever.from_llm(retriever=retriever, llm=primary_qa_llm)
document_chain = create_stuff_documents_chain(primary_qa_llm, prompt)
retrieval_chain = create_retrieval_chain(advanced_retriever, document_chain)

# Title and description
st.title("Insurance Policy Chatbot")
st.write("Your guide to understanding your insurance coverage.")

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

    try:
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
