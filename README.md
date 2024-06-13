# Insurance_Chatbot_evaluation

This project implements a chatbot that answers questions about Churchill Insurance policies using LangChain and OpenAI. It leverages document retrieval and a large language model to provide accurate and relevant responses.

## Features
* Document Loading and Processing: Loads an insurance policy document (Markdown format), splits it into chunks, and creates a vector representation for efficient retrieval.
* Question-Answering: Takes user questions, retrieves relevant information from the policy document, and generates answers using OpenAI's GPT-3.5-turbo model.
* Streamlit Interface: Provides a user-friendly Streamlit interface for interacting with the chatbot.

## Evaluation
The project includes an evaluation notebook (Ipynb file) that uses the Ragas library to assess the chatbot's performance on a variety of metrics:
* Faithfulness
* Answer Relevancy
* Context Precision
* Context Recall
* Answer Correctness

## Usage
* Use the Chatbot: [insurancechatbot1.streamlit.app](https://insurancechatbot1.streamlit.app/)
* Enter Questions: Type your insurance policy questions in the Streamlit interface.
* Get Answers: The chatbot will respond with answers based on the policy document.

## Evaluation
* Open the Notebook: Open the Ipynb file in a Jupyter Notebook environment.
* Run the Cells: Execute the cells in the notebook to generate a test dataset, run the chatbot, and evaluate its performance using Ragas.

## Customization
* Modify the markdown_path in app.py to use your document.
* Adjust parameters in the RecursiveCharacterTextSplitter to control how the document is split.
* Explore different OpenAI language models by changing the model_name in ChatOpenAI.
