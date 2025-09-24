import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
import tempfile

from dotenv import load_dotenv


load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

st.title("Hỏi đáp từ tài liệu PDF với Google Gemini")

uploaded_files = st.file_uploader("Tải lên file PDF", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    all_docs = []
    temp_dir = tempfile.mkdtemp()
    with st.spinner("Đang đọc và xử lý tài liệu..."):
        for uploaded_file in uploaded_files:
            original_filename = uploaded_file.name
            tmp_path = os.path.join(temp_dir, original_filename)
            with open(tmp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            loader = PyPDFLoader(tmp_path)
            docs = loader.load()
            all_docs.extend(docs)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        split_docs = text_splitter.split_documents(all_docs)

        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        db = FAISS.from_documents(split_docs, embeddings)

    st.success(f"Đã xử lý {len(uploaded_files)} tài liệu PDF!")
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
            llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0,streaming=True)
            for chunk in llm.stream(final_prompt):
                if chunk.content:
                    text += chunk.content
                    container.markdown(text)

            with st.expander("Các đoạn văn bản tham chiếu"):
                for i, doc in enumerate(docs, 1):
                    source = os.path.basename(doc.metadata.get("source", "Không rõ file"))
                    page_num = doc.metadata.get("page", "Không rõ")
                    st.markdown(
                        f"**Đoạn {i}** (File: `{source}`, Trang {page_num}):\n\n{doc.page_content}"
                    )