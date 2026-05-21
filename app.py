import streamlit as st
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain_groq import ChatGroq




st.set_page_config(
    page_title="Chat with PDFs",
    page_icon="📚",
    layout="wide"
)


st.markdown("""
<style>

html, body, [class*="css"] {
    background-color: #FFFFFF !important;
    color: #000000 !important;
}

.stApp {
    background-color: #FFFFFF;
    color: #000000;
}

section[data-testid="stSidebar"] {
    background-color: #FFFFFF !important;
    color: #000000 !important;
    border-right: 1px solid #E0E0E0;
}

.stTextInput input {
    background-color: #FFFFFF !important;
    color: #000000 !important;
    border: 1px solid #DADADA !important;
    border-radius: 10px !important;
}

[data-testid="stFileUploader"] {
    background-color: #FFFFFF !important;
    color: #000000 !important;
}

.stButton button {
    background-color: #FFFFFF !important;
    color: #000000 !important;
    border: 1px solid #DADADA !important;
    border-radius: 10px !important;
    padding: 0.5rem 1rem !important;
}

.stButton button:hover {
    background-color: #F5F5F5 !important;
}

.chat-message {
    border-radius: 16px;
    padding: 12px 16px;
    margin-bottom: 12px;
    border: 1px solid #DADADA;
    display: flex;
    align-items: flex-start;
    gap: 12px;
    background-color: #FFFFFF;
    color: #000000;
}

.chat-message.user {
    background-color: #FFFFFF;
}


.chat-message.bot {
    background-color: #FFFFFF;
}

.chat-message img {
    width: 42px;
    height: 42px;
    border-radius: 50%;
    object-fit: cover;
    border: 1px solid #DADADA;
}

.stMarkdown,
p,
span,
label,
div {
    color: #000000 !important;
}

/* Headers */
h1, h2, h3, h4, h5, h6 {
    color: #000000 !important;
}

/*Mobile responsiveness*/
@media (max-width: 768px){
    .chat-message{
        padding:14px;
        font-size: 15px;
    }
}

</style>
""", unsafe_allow_html=True)



bot_template = """
<div class="chat-message bot">
    <img src="https://cdn-icons-png.flaticon.com/512/4712/4712027.png">
    <div>{{MSG}}</div>
</div>
"""

user_template = """
<div class="chat-message user">
    <img src="https://cdn-icons-png.flaticon.com/512/847/847969.png">
    <div>{{MSG}}</div>
</div>
"""




def get_pdf_text(pdf_docs):

    text = ""

    for pdf in pdf_docs:

        # Prevent large uploads
        if pdf.size > 5 * 1024 * 1024:
            st.warning(f"{pdf.name} is too large. Max size is 5MB.")
            continue

        try:
            pdf.seek(0)

            pdf_reader = PdfReader(pdf)

            for page in pdf_reader.pages:

                extracted = page.extract_text()

                if extracted:
                    text += extracted + "\n"

        except Exception as e:
            st.error(f"Error reading PDF {pdf.name}: {e}")

    return text


def get_text_chunks(text):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )

    return text_splitter.split_text(text)


def get_vectorstore(text_chunks):

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_texts(
        texts=text_chunks,
        embedding=embeddings
    )

    return vectorstore


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



def handle_userinput(user_question):

    response = st.session_state.conversation.invoke({
        "question": user_question,
        "chat_history": st.session_state.chat_history
    })

    answer = response["answer"]

    # Save history
    st.session_state.chat_history.append(
        (user_question, answer)
    )

    # Display user message
    st.markdown(
        user_template.replace("{{MSG}}", user_question),
        unsafe_allow_html=True
    )

    # Display bot response
    st.markdown(
        bot_template.replace("{{MSG}}", answer),
        unsafe_allow_html=True
    )


def main():

    load_dotenv()

    # API key check
    if not os.getenv("GROQ_API_KEY"):
        st.error("GROQ_API_KEY not found in environment variables.")
        st.stop()

    # Session state
    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Header
    st.header("Chat with PDFs 📚")
    st.markdown(
        "Upload your PDFs and ask questions from your documents."
    )

    # User question
    user_question = st.text_input(
        "Ask a question about your documents:"
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
            "Upload PDF files",
            accept_multiple_files=True,
            type=["pdf"]
        )

        if st.button("Process PDFs"):

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

                st.session_state.conversation = (
                    get_conversation_chain(vectorstore)
                )

                # Reset chat history
                st.session_state.chat_history = []

                st.success("PDFs processed successfully!")



if __name__ == "__main__":
    main()