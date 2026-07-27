import io
import os
import tempfile
import zipfile
from pathlib import Path

import streamlit as st

try:
    from langchain_community.document_loaders import (
        Docx2txtLoader,
        PyPDFLoader,
        TextLoader,
        UnstructuredHTMLLoader,
    )
except ImportError:  # pragma: no cover - fallback for environments without LangChain
    PyPDFLoader = None
    Docx2txtLoader = None
    TextLoader = None
    UnstructuredHTMLLoader = None


LOADER_MAPPING = {
    ".pdf": PyPDFLoader,
    ".docx": Docx2txtLoader,
    ".txt": TextLoader,
    ".html": UnstructuredHTMLLoader,
    ".htm": UnstructuredHTMLLoader,
}


def load_document_text(file_path: str) -> str:
    """Load a document with the right loader and return the combined text."""
    ext = Path(file_path).suffix.lower()

    if ext not in LOADER_MAPPING:
        raise ValueError(f"Unsupported file type: {ext}")

    loader_class = LOADER_MAPPING[ext]
    if loader_class is None:
        if ext in {".txt", ".html", ".htm"}:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as handle:
                return handle.read()
        raise RuntimeError("LangChain loaders are not available in this environment.")

    loader = loader_class(file_path)
    documents = loader.load()
    return "\n\n".join(doc.page_content for doc in documents)


def save_uploaded_file_to_temp(uploaded_file) -> str:
    """Write an uploaded file to a temp file on disk so loaders can read it."""
    suffix = Path(uploaded_file.name).suffix.lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        return tmp.name


def make_zip(results: dict) -> bytes:
    """Bundle extracted texts into a single zip file for download."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for filename, text in results.items():
            zf.writestr(filename, text)
    buffer.seek(0)
    return buffer.read()


st.set_page_config(page_title="Document Extractor", page_icon="📄", layout="wide")

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg, #f7f9fd 0%, #eef4ff 100%);
    }
    .block-container {
        padding-top: 2rem;
        max-width: 880px;
    }
    h1 {
        font-size: 1.9rem !important;
        margin-bottom: 0.2rem !important;
    }
    .subtitle {
        color: #5b6b86;
        font-size: 1rem;
        margin-bottom: 1.4rem;
    }
    .stFileUploader > div {
        border: 2px dashed #7c8db8;
        border-radius: 18px;
        padding: 0.5rem 0;
        background: #fbfdff;
    }
    .stDownloadButton > button,
    .stButton > button {
        border-radius: 999px;
        padding: 0.6rem 1.1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("# Smart Document Extractor")
st.markdown(
    "<p class='subtitle'>Upload your PDF, DOCX, TXT, or HTML files and turn them into clean, readable text in seconds.</p>",
    unsafe_allow_html=True,
)

uploaded_files = st.file_uploader(
    "Drop files here or click to browse",
    type=["pdf", "docx", "txt", "html", "htm"],
    accept_multiple_files=True,
    label_visibility="collapsed",
)

if uploaded_files:
    st.markdown("#### Selected files")
    for uploaded_file in uploaded_files:
        st.caption(f"📄 {uploaded_file.name}")

    if st.button("Extract text", use_container_width=True, type="primary"):
        results = {}
        progress = st.progress(0)
        status = st.empty()

        with st.spinner("Extracting text from your files..."):
            for index, uploaded_file in enumerate(uploaded_files):
                ext = Path(uploaded_file.name).suffix.lower()
                if ext not in LOADER_MAPPING:
                    st.warning(f"Skipping unsupported file: {uploaded_file.name}")
                    continue

                temp_path = None
                try:
                    temp_path = save_uploaded_file_to_temp(uploaded_file)
                    text = load_document_text(temp_path)
                    output_name = Path(uploaded_file.name).stem + ".txt"
                    results[output_name] = text
                    status.info(f"Processed {uploaded_file.name}")
                except Exception as exc:
                    st.error(f"Failed to process {uploaded_file.name}: {exc}")
                finally:
                    if temp_path and os.path.exists(temp_path):
                        os.remove(temp_path)

                progress.progress((index + 1) / len(uploaded_files))

        if results:
            st.success(f"Processed {len(results)} file(s) successfully.")
            zip_bytes = make_zip(results)
            st.download_button(
                label="⬇️ Download all as ZIP",
                data=zip_bytes,
                file_name="extracted_texts.zip",
                mime="application/zip",
                use_container_width=True,
            )

            st.divider()
            for filename, text in results.items():
                with st.expander(f"📄 {filename}", expanded=False):
                    st.text_area("Extracted text", text, height=220, key=f"preview_{filename}")
                    st.download_button(
                        label=f"Download {filename}",
                        data=text,
                        file_name=filename,
                        mime="text/plain",
                        key=f"download_{filename}",
                    )
        else:
            st.info("No files could be extracted. Please try again with a supported format.")
else:
    st.info("Upload one or more files to get started.")