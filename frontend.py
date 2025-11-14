from rag_pipeline import answer_query, generate_summary, generate_audio_file, llm_model
from vector_database import build_faiss_from_pdf
import streamlit as st

#  Streamlit Page Setup
st.set_page_config(page_title="CaseVise - Legal Assistant", page_icon="⚖️", layout="centered")

#  Custom Styling
st.markdown("""
<style>
.upload-heading { margin-bottom: -10px; }
.ask-heading { margin-top: 20px; }
textarea {
    border-radius: 10px !important;
    box-shadow: 0 2px 6px rgba(0,0,0,0.1) !important;
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<h1 style="text-align: center; color: #2E86C1;">⚖️ CaseVise - Legal Assistant</h1>
<p style="text-align: center; color: gray;">
Upload one or more case documents (FIRs, witness statements, judgments, etc.) and ask questions or request summaries.
</p>
<hr style="border:1px solid #ccc;">
""", unsafe_allow_html=True)

#  Upload Section — Multiple PDFs
st.markdown('<h3 class="upload-heading" style="color:#2E86C1;">📂 Upload Your Case PDFs</h3>', unsafe_allow_html=True)
uploaded_files = st.file_uploader(" ", type="pdf", accept_multiple_files=True, label_visibility="collapsed")

#  Q&A Section
st.markdown('<h3 class="ask-heading" style="color:#2E86C1;">💬 Ask Your Question</h3>', unsafe_allow_html=True)
user_query = st.text_area(" ", height=120, placeholder="e.g. What was the main evidence used in the judgment?", label_visibility="collapsed")

#  Perspective Options
perspective = st.selectbox("Choose summary perspective:", ["Student", "Lawyer", "Judge"])
ask_question = st.button("Ask CaseVise", use_container_width=True)
summarize_btn = st.button("Generate Summary", use_container_width=True)

# ⚙️ Processing Logic
if ask_question or summarize_btn:
    if uploaded_files:
        with st.spinner("Processing your documents... Please wait ⏳"):
            #  Combine all uploaded PDFs into a single FAISS DB
            all_documents = []
            for file in uploaded_files:
                faiss_db = build_faiss_from_pdf(file)
                all_documents.extend(faiss_db.similarity_search(""))  # collect docs
        #  Handle Question or Summary
        if ask_question:
            st.chat_message("user").write(user_query)
            # Search within all documents
            retrieved_docs = []
            for file in uploaded_files:
                temp_db = build_faiss_from_pdf(file)
                retrieved_docs.extend(temp_db.similarity_search(user_query,k=5))

            response = answer_query(retrieved_docs, llm_model, user_query)
            st.chat_message("CaseVise").write(response)
            st.session_state["last_response"] = response

        elif summarize_btn:
            
            st.chat_message("user").write(f"Generate a {perspective.lower()} summary for all uploaded case files.")
            retrieved_docs = []
            for file in uploaded_files:
                temp_db = build_faiss_from_pdf(file)
                retrieved_docs.extend(temp_db.similarity_search("case summary"))

            summary = generate_summary(retrieved_docs, llm_model, perspective.lower())
            st.chat_message("CaseVise").write(summary)
            st.session_state["last_response"] = summary

        #  Inline Listen Player
        if "last_response" in st.session_state:
            audio_file = generate_audio_file(st.session_state["last_response"])
            if audio_file:
                st.audio(audio_file, format="audio/mp3", start_time=0)
    else:
        st.error("⚠️ Please upload at least one PDF file!")
