import os
import tempfile

import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

st.set_page_config(
    page_title="AI PDF Assistant",
    page_icon="📚",
)

st.title("📚 AI PDF Assistant")
st.write("Upload a PDF and ask questions about its contents.")

api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    st.error("GOOGLE_API_KEY is missing from your .env file.")
    st.stop()

uploaded_file = st.file_uploader(
    "Upload your PDF",
    type="pdf"
)

if uploaded_file:

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    ) as temp_file:

        temp_file.write(uploaded_file.read())
        pdf_path = temp_file.name

    with st.spinner("Reading PDF..."):

        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001"
        )

        vector_store = FAISS.from_documents(
            documents,
            embeddings
        )

    st.success("PDF processed successfully!")

    question = st.text_input(
        "Ask a question about your PDF:"
    )

    if question:

        with st.spinner("Thinking..."):

            docs = vector_store.similarity_search(
                question,
                k=4
            )

            context = "\n\n".join(
                doc.page_content
                for doc in docs
            )

            prompt = ChatPromptTemplate.from_template(
                """
                You are a helpful document assistant.

                Answer the question ONLY using
                the provided context.

                If the answer is not present in the
                context, say:
                "I could not find the answer in the document."

                Context:
                {context}

                Question:
                {question}
                """
            )

            llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                temperature=0
            )

            chain = prompt | llm

            response = chain.invoke({
                "context": context,
                "question": question
            })

            st.subheader("Answer")
            st.write(response.content)

            st.subheader("Sources")

            for doc in docs:
                page = doc.metadata.get(
                    "page",
                    "Unknown"
                )

                st.write(
                    f"Page {page + 1}"
                )
