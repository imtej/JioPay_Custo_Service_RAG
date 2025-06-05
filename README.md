# JioPay_Customer_Service_RAG 
 

## RAG Architecture Overview 
 
This implements a Retrieval-Augmented Generation (RAG) system using `Streamlit`, `LangChain`, and `Hugging Face` for interactive question-answering based on JioPay Business Customer services and all the other information as a  knowledge base. The architecture integrates document embedding, retrieval, and generative AI for comprehensive and context-aware responses.

### Key Components

1. **Document Handling and Embedding:**
   - The documents are processed using `pymupdf4llm` to extract text data from the web scrapped comprehenssive pdf of the knowledge base both the sites from https://jiopay.com/business and https://jiopay.com/business/help-center.
   - The text is split into manageable `chunks (1024 tokens and overlapping 256)` using `RecursiveCharacterTextSplitter` for efficient handling during the embedding and retrieval process.
   - These text chunks are then embedded using a `HuggingFaceBgeEmbeddings` model, which generates vector representations for the text using the `thenlper/gte-small` model.
   - A `FAISS` index is created from these embeddings, enabling efficient vector-based document retrieval.

2. **User Interaction:**
   - Users interact with the system through a chat interface powered by `Streamlit`. They can input questions and view responses directly within the app.

3. **Language Model Integration:**
   - The system utilizes `Groq enpoind API` to interface with `llama-3.1-8b-instant` as a LLM
   - The selected model generates responses to user queries based on the most relevant document chunks retrieved from the FAISS index.

4. **Retrieval and Generation Workflow:**
   - The core of the RAG system is the retrieval process, where the top 5 most relevant document chunks are fetched from the FAISS index based on the user's query.
   - These chunks are passed to a `create_stuff_documents_chain` along with the language model, forming the input for response generation.
   - The system processes the query through this chain and returns a detailed, contextually relevant response, which is displayed in the chat interface.

### Additional Features

- **Chat History:** The application maintains a session-based chat history, displaying past interactions for reference.




## How to Use

Follow these steps to set up and run the project on your local machine:

### 1. Clone the Repository
First, clone the repository to your local machine using the following command:
```bash

git clone https://github.com/delta-tej/JioPay_Custo_Service_RAG.git
cd JioPay_Custo_Service_RAG
```

### 2. Install Dependencies
Make sure you have Python installed (version 3.11 recommended). Then, install the required dependencies using `pip`:
```bash
pip install -r requirements.txt
```


### 3. Run the Application
To start the application, run the following command:
```bash
streamlit run app.py
```


---

### Example Usage

Here’s an example of how to interact with the system:

2. Ask a question like: "What is the JioPay Business?"
3. The system will retrieve relevant document sections and provide a detailed response.
