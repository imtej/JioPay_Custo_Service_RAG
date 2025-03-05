import streamlit as st
import pickle
# from langchain.chat_models import ChatOpenAI
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate

# 1. Load the pre-indexed FAISS database
@st.cache_data
def load_faiss_db(pickle_path: str):
    with open(pickle_path, "rb") as f:
        db = pickle.load(f)
    return db

def main():
    st.title(" JioPay customer support agent")

    # -- Load or initialize the retrieval chain in session_state --
    if "retrieval_chain" not in st.session_state:
        # 1) Load our FAISS vector database
        db = load_faiss_db("faiss_index1.pkl")  # Load the pre-indexed FAISS database

        # 2) Create our LLM
        #    - If we have a Groq or other custom endpoint, replace ChatOpenAI with our own LLM class
        #    - For OpenAI, we'll need an environment variable OPENAI_API_KEY or pass openai_api_key=...

        groq_api_key =  "gjsjsjsjsjasjjajsajsjsjsjjs"

        llama3 = ChatOpenAI(api_key=groq_api_key,
                    base_url="https://api.groq.com/openai/v1",
                    model="llama-3.1-8b-instant",
                    temperature=0,
                    )
        llm = llama3

        # 3) Create a custom prompt
        prompt = ChatPromptTemplate.from_template("""
Instructions:
- You Are a helpful assistant. Designed and Developed by RAVI TEJ for JioPay Buisiness Customer Service.
- Be helpful and answer questions concisely. If you don't know the answer, say 'I don't know'
- Utilize the context provided for accurate and specific information.
- Incorporate your preexisting knowledge to enhance the depth and relevance of your response.
- Cite your sources
Context: {context}

Question: {input}
        """)

        # 4) Build the chain
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = db.as_retriever(search_kwargs={"k": 4})  # Adjust 'k' as needed
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        st.session_state["retrieval_chain"] = retrieval_chain

    # -- Chat Interface --
    user_query = st.text_input("Ask question:")
    if user_query:
        # Invoke the chain
        response = st.session_state["retrieval_chain"].invoke({"input": user_query})
        answer = response["answer"]

        # Display answer
        st.write("**Answer:** ", answer)

if __name__ == "__main__":
    main()
