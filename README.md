# Insurance_Chatbot_evaluation

This project implements a chatbot that answers questions about Churchill Insurance policies using LangChain and OpenAI. It leverages document retrieval and a large language model to provide accurate and relevant responses.

## Features
* Document Loading and Processing: Loads an insurance policy document (Markdown format), splits it into chunks, and creates a vector representation for efficient retrieval.
Question-Answering: Takes user questions, retrieves relevant information from the policy document, and generates answers using OpenAI's GPT-3.5-turbo model.
Streamlit Interface: Provides a user-friendly Streamlit interface for interacting with the chatbot.

Evaluation
The project includes an evaluation notebook (Ipynb file) that uses the Ragas library to assess the chatbot's performance on a variety of metrics:
Faithfulness
Answer Relevancy
Context Precision
Context Recall
Answer Correctness
