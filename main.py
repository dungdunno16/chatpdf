import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.callbacks.base import BaseCallbackHandler

from dotenv import load_dotenv


load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")


st.title("Hỏi đáp từ tài liệu PDF với Google Gemini")

uploaded_file = st.file_uploader("Tải lên file PDF", type=["pdf"])

if uploaded_file is not None:
    with st.spinner("Đang đọc và xử lý tài liệu..."):
        pdf_path = f"temp.pdf"
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        loader = PyPDFLoader(pdf_path)
        docs = loader.load_and_split()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        split_docs = text_splitter.split_documents(docs)

        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        db = FAISS.from_documents(split_docs, embeddings)

    st.success("Tài liệu đã được xử lý xong!")
    query = st.text_input("Nhập câu hỏi của bạn:")

    prompt_template = """
       Bạn là một trợ lý AI có nhiệm vụ trả lời câu hỏi dựa trên tài liệu được cung cấp.
       - Chỉ sử dụng thông tin trong tài liệu (context).
       - Nếu không tìm thấy thông tin phù hợp trong tài liệu, hãy trả lời: "Xin lỗi, tôi không tìm thấy thông tin trong tài liệu."
       - Trình bày câu trả lời rõ ràng, ngắn gọn, dễ hiểu.

       Context:
       {context}

       Câu hỏi: {question}

       Trả lời:
       """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    if query:
        with st.spinner("Đang tìm câu trả lời..."):
            docs = db.similarity_search(query, k=3)
            context = "\n\n".join([doc.page_content for doc in docs])
            final_prompt = prompt.format(context=context, question=query)

            st.subheader("Trả lời:")
            container = st.empty()
            text = ""
            # st.write(response)
            llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0,streaming=True)
            # response = llm.predict(final_prompt)
            for chunk in llm.stream(final_prompt):
                if chunk.content:
                    text += chunk.content
                    container.markdown(text)

