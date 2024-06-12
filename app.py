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


# Load environment variables from .env file
load_dotenv()

# Set up OpenAI API key
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Define prompt for the LLM
prompt_template = """Answer the question based only on the following context. If you cannot answer the question with the context, please respond with 'I don't know':
Context: {context}
Question: {question}
"""

# Load and process the Markdown file
markdown_path = "Data/OUTPUT/policy-booklet-0923/policy-booklet-0923.md"
loader = UnstructuredMarkdownLoader(markdown_path)
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
documents = text_splitter.split_documents(documents)

# Create embeddings and vector store
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
vector_store = FAISS.from_documents(documents, embeddings)

# Create retriever and set up LLM
retriever = vector_store.as_retriever()
qa_llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
advanced_retriever = MultiQueryRetriever.from_llm(retriever=retriever, llm=qa_llm)
retrieval_qa_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

# Create the QA chain with the prompt template
document_chain = create_stuff_documents_chain(qa_llm, retrieval_qa_prompt)
retrieval_qa_prompt = ChatPromptTemplate.from_template(prompt_template)
qa_chain = create_retrieval_chain(document_chain, retrieval_qa_prompt)


# Streamlit App
st.title("Insurance Policy Chatbot")
st.write("Your guide to understanding your insurance coverage.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask me about your policy"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    try:
        response = qa_chain({"query": prompt})
        answer = response['result']

        st.session_state.messages.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.markdown(answer)

    except Exception as e:
        st.error(f"An error occurred: {e}")
