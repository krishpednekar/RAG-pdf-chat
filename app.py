import streamlit as st
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain

import gc

from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from htmlTemplates import css, bot_template, user_template


# ---------------- PAGE CONFIG ---------------- #

st.set_page_config(
    page_title="Chat with PDFs",
    page_icon="📚"
)

st.write(css, unsafe_allow_html=True)

st.markdown("""
<style>
.chat-message.user{
    background:#E8E3DB;
    color:#111;
}

.chat-message.bot{
    background:#F5F1EA;
    color:#111;
}

.chat-message{
    border-radius:20px;
    padding:18px;
    margin-bottom:14px;
}
</style>
""", unsafe_allow_html=True)


# ---------------- PDF TEXT EXTRACTION ---------------- #

def get_pdf_text(pdf_docs):

    text = ""

    for pdf in pdf_docs:

        # Limit PDF size to avoid Render memory crash
        if pdf.size > 5 * 1024 * 1024:
            st.warning(f"{pdf.name} is too large. Max size is 5MB.")
            continue

        try:
            pdf.seek(0)

            pdf_reader = PdfReader(pdf)

            for page in pdf_reader.pages:

                extracted = page.extract_text()

                if extracted:
                    text += extracted

        except Exception as e:
            st.error(f"Error reading PDF: {e}")

    return text


# ---------------- TEXT CHUNKING ---------------- #

def get_text_chunks(text):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=30,
        length_function=len
    )

    return text_splitter.split_text(text)


# ---------------- VECTOR STORE ---------------- #

@st.cache_resource
def get_vectorstore(text_chunks):

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

    vectorstore = Chroma.from_texts(
        texts=text_chunks,
        embedding=embeddings
    )

    return vectorstore
    gc.collect()


# ---------------- CONVERSATION CHAIN ---------------- #

def get_conversation_chain(vectorstore):

    llm = ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.3-70b-versatile",
        temperature=0
    )

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 2}),
        return_source_documents=False
    )

    return conversation_chain


# ---------------- USER INPUT ---------------- #

def handle_userinput(user_question):

    response = st.session_state.conversation.invoke({
        "question": user_question
    })

    answer = response["answer"]

    st.write(
        user_template.replace("{{MSG}}", user_question),
        unsafe_allow_html=True
    )

    st.write(
        bot_template.replace("{{MSG}}", answer),
        unsafe_allow_html=True
    )


# ---------------- MAIN APP ---------------- #

def main():

    load_dotenv()

    # API key checks
    if not os.getenv("GOOGLE_API_KEY"):
        st.error("GOOGLE_API_KEY not found in .env file")
        st.stop()

    if not os.getenv("GROQ_API_KEY"):
        st.error("GROQ_API_KEY not found in .env file")
        st.stop()

    # Session state
    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "user_question" not in st.session_state:
      st.session_state.user_question = ""

    # Header
    st.header("Chat with PDFs 📚")
    st.markdown("*Upload and chat with your PDF documents*")

    # User question
    if "user_question" not in st.session_state:
        st.session_state.user_question = ""

    user_question = st.text_input(
    "Ask a question about your documents:",
    key="user_question"
    )

    if user_question:

        if st.session_state.conversation:
            handle_userinput(user_question)

        else:
            st.warning("Please upload and process PDFs first.")

    # Sidebar
    with st.sidebar:

        st.subheader("Your Documents")

        pdf_docs = st.file_uploader(
            "Upload your PDFs here",
            accept_multiple_files=True,
            type=["pdf"]
        )

        if st.button("Process"):

            if not pdf_docs:
                st.warning("Please upload at least one PDF.")
                st.stop()

            with st.spinner("Processing PDFs..."):

                raw_text = get_pdf_text(pdf_docs)

                if not raw_text.strip():
                    st.error("No readable text found in PDFs.")
                    st.stop()

                text_chunks = get_text_chunks(raw_text)

                vectorstore = get_vectorstore(text_chunks)

                st.session_state.conversation = get_conversation_chain(
                    vectorstore
                )

                st.success("Processing complete!")


if __name__ == "__main__":
    main()