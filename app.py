import streamlit as st
from  PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
print("GOOGLE_API_KEY =", os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_docs):
    st.write("Reading PDF files...")
    print("in pdf texts")
    text=""
    for pdf in pdf_docs:
        pdf_reader=PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    print("done pdftexts")
    return text

def get_text_chunks(text):
    st.write("Reading get_text_chunks files...")
    print("inget chunks")
    textsplitter= RecursiveCharacterTextSplitter(chunk_size=10000 , chunk_overlap= 1000)
    chunks = textsplitter.split_text(text)
    st.write("Finished reading PDFs.")
    print("finished reading pdfs")
    return chunks

def get_vector_store(text_chunks):
    print("get vector embeddings")
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_texts(text_chunks, embedding=embeddings)
    vectorstore.save_local("faiss-index")


def get_converstaional_chain():
    print("conversation chain")
     
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="models/gemini-1.5-pro-latest",temperature=0.3)
    prompt= PromptTemplate(template=prompt_template,input_variables=["context","question"])
    chain = load_qa_chain(model, chain_type="stuff",prompt=prompt)
    return chain


def user_input(user_question):
    print("in user input")
    print("GOOGLE_API_KEY =", os.getenv("GOOGLE_API_KEY"))
    embeddings= GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    new_db= FAISS.load_local("faiss-index",embeddings, allow_dangerous_deserialization=True )
    docs=new_db.similarity_search(user_question)

    chain=get_converstaional_chain()

    response=chain({
        "input_documents": docs , "question": user_question}, return_only_outputs=True)
    
    print(response)

    st.write("Reply:  ", response["output_text"])




def main():
    
    
    st.set_page_config(page_title="Chat PDF")  # Ensure this is at the top
    st.header("📄 Chat with PDF using Gemini 💁")

    # Show text input in main view
    user_question = st.text_input("Ask a question from the uploaded PDF files")

    if user_question:
        st.write("📩 Received question:", user_question)
        try:
            user_input(user_question)
        except Exception as e:
            st.error(f"Error in processing question: {e}")

    # Sidebar for file upload
    with st.sidebar:
        st.title("📁 Upload Menu")
        pdf_docs = st.file_uploader("Upload your PDF files", accept_multiple_files=True)
        st.write("📄 Uploaded files:", pdf_docs)
        if st.button("Submit & Process"):
            st.write("🔁 Processing started...")
            try:
                raw_text = get_pdf_text(pdf_docs)
                st.write("📝 Extracted text length:", len(raw_text))
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("✅ Processing complete!")
            except Exception as e:
                st.error(f"Error during processing: {e}")





if __name__ == "__main__":
    main()

