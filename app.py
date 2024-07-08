import os
import streamlit as st
from dotenv import load_dotenv
from langchain.document_loaders import UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.retrievers import MultiQueryRetriever
from langchain_core.documents import Document
from langchain.chains import RetrievalQA,  ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI

# Load environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

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

    # Embed and store the document chunks
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    vector_store = FAISS.from_documents(documents, embeddings)
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    # Define the prompt template
    prompt_template = """
    <|im_start|>system
    You are a helpful assistant.
    You are given relevant documents for context and a question. Provide a conversational answer.
    If you don't know the answer, just say "I do not know." Don't make up an answer.
    <|im_end|>
    <|im_start|>user
    {question}
    <|im_end|>
    <|im_start|>assistant
    {context} 
    <|im_end|>
    """
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    # Set up the LLM
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    #Conversational Memory
    memory = ConversationBufferMemory(memory_key="chat_history")

    retriever_from_llm = MultiQueryRetriever.from_llm(retriever, llm=llm)

    # Create the chain
    qa_chain = ConversationalRetrievalChain.from_llm(llm=llm,
                                                     chain_type="stuff",
                                                     retriever=retriever_from_llm,
                                                     memory=memory,
                                                     combine_docs_chain_kwargs={"prompt": PROMPT})

    # Streamlit input and interaction
    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            response = qa_chain({"question": prompt})
            st.write(response["answer"])
            st.session_state.messages.append({"role": "assistant", "content": response['answer']})

if __name__ == "__main__":
    main()
