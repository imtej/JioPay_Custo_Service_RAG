# We can save the db and then import

import pymupdf4llm
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
import pickle

# convert the document to markdown
md_text = pymupdf4llm.to_markdown("D:\Vizuara_Assignment\FAQ_and_Knowledge_base_JioPay_Vizuara_Assn.pdf")  # parsing the knowledge base from pdf to markdown

# Embedding Model for converting text to numerical representations
embedding_model = HuggingFaceEmbeddings(
    model_name='thenlper/gte-small'
)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=256)
docs = text_splitter.split_text(md_text)
len(docs)


# Create a local vector database
db = FAISS.from_texts(docs, embedding_model)

# Save the database to a file
with open("faiss_index1.pkl", "wb") as f:
    pickle.dump(db, f)

print("Database saved to faiss_index.pkl")
