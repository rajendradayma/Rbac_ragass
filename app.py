import streamlit as st
import os
from indexer import (
    scan_university_folder,
    build_index,
    get_folder_structure,
    FAISS_INDEX_PATH,
    UNIVERSITY_ROOT,
    EMBEDDING_MODEL_NAME
)

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title = "University RAG",
    page_icon  = "🎓",
    layout     = "wide"
)

GROQ_API_KEY = "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"
GROQ_MODEL   = "llama-3.3-70b-versatile"

# ============================================================
# SESSION STATE
# ============================================================
if "chat_history"     not in st.session_state:
    st.session_state.chat_history     = []
if "selected_folders" not in st.session_state:
    st.session_state.selected_folders = []

# ============================================================
# RAG FUNCTION
# ============================================================
def ask_question(question: str, access_folders: List[str]) -> dict:
    """
    access_folders — list of folders user selected
    Pre-filter: only chunks whose folder is in access_folders
    """
    if not os.path.exists(FAISS_INDEX_PATH):
        return {
            "answer"  : "Index not found. Please index documents first.",
            "sources" : []
        }

    os.environ["GROQ_API_KEY"] = GROQ_API_KEY

    llm         = ChatGroq(model=GROQ_MODEL, temperature=0.1)
    embeddings  = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    vectorstore = FAISS.load_local(
        FAISS_INDEX_PATH, embeddings,
        allow_dangerous_deserialization=True
    )

    # PRE FILTER — folder based
    def rbac_filter(metadata: dict) -> bool:
        chunk_folder = metadata.get("folder", "")
        return any(
            chunk_folder.startswith(allowed)
            for allowed in access_folders
        )

    retriever = vectorstore.as_retriever(
        search_kwargs={"filter": rbac_filter, "k": 5}
    )

    system_prompt = (
        "You are a university assistant. Use the following retrieved context "
        "to answer the question. If the answer is not in the context, "
        "say you don't have that information in the accessible documents.\n\n"
        "<context>\n{context}\n</context>"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    qa_chain  = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, qa_chain)
    response  = rag_chain.invoke({"input": question})

    retrieved  = response.get("context", [])

    return {
        "answer" : response["answer"],
        "sources": [
            {
                "filename": doc.metadata.get("filename", "unknown"),
                "folder"  : doc.metadata.get("folder",   "unknown"),
                "content" : doc.page_content.strip()[:300]
            }
            for doc in retrieved
        ]
    }

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.title("🎓 University RAG")
    st.divider()

    # INDEX BUTTON
    st.subheader("📥 Index Documents")
    if st.button("🔄 Scan & Index", use_container_width=True):
        with st.spinner("Scanning university folder..."):
            records = scan_university_folder()
        with st.spinner(f"Indexing {len(records)} files..."):
            build_index(records)
        st.success(f"✅ Indexed {len(records)} files!")
        st.rerun()

    st.divider()

    # FOLDER SELECTOR
    st.subheader("📂 Select Folders to Query")
    structure = get_folder_structure()

    if not structure:
        st.warning("No folders found. Add PDFs to university/ folder and index.")
    else:
        # Select All button
        if st.button("Select All", use_container_width=True):
            st.session_state.selected_folders = list(structure.keys())

        # Clear All button
        if st.button("Clear All", use_container_width=True):
            st.session_state.selected_folders = []

        st.write("**Available Folders:**")
        for folder in structure.keys():
            checked = folder in st.session_state.selected_folders
            if st.checkbox(
                f"📂 {folder}",
                value   = checked,
                key     = f"chk_{folder}"
            ):
                if folder not in st.session_state.selected_folders:
                    st.session_state.selected_folders.append(folder)
            else:
                if folder in st.session_state.selected_folders:
                    st.session_state.selected_folders.remove(folder)

    st.divider()

    # SHOW SELECTED
    if st.session_state.selected_folders:
        st.subheader("✅ Selected Access")
        for f in st.session_state.selected_folders:
            st.success(f"🔓 {f}")
    else:
        st.warning("No folders selected")

# ============================================================
# MAIN AREA
# ============================================================
st.title("🎓 University Knowledge Assistant")

# Status bar
col1, col2, col3 = st.columns(3)
with col1:
    indexed = os.path.exists(FAISS_INDEX_PATH)
    st.metric(
        "Index Status",
        "✅ Ready" if indexed else "❌ Not Built"
    )
with col2:
    structure = get_folder_structure()
    st.metric("Total Folders", len(structure))
with col3:
    st.metric(
        "Selected Folders",
        len(st.session_state.selected_folders)
    )

st.divider()

# Tabs
tab1, tab2 = st.tabs(["💬 Chat", "📁 Folder Structure"])

# ──────────────────────────────────────────
# TAB 1 — CHAT
# ──────────────────────────────────────────
with tab1:

    if not st.session_state.selected_folders:
        st.warning(
            "⚠️ No folders selected. "
            "Please select folders from the sidebar to query."
        )
    else:
        st.info(
            f"🔐 Querying folders: "
            f"**{', '.join(st.session_state.selected_folders)}**"
        )

    # Chat history
    for chat in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(chat["question"])
        with st.chat_message("assistant"):
            st.write(chat["answer"])
            if chat.get("sources"):
                with st.expander(
                    f"📎 {len(chat['sources'])} Sources Used"
                ):
                    for src in chat["sources"]:
                        st.write(
                            f"📄 **{src['filename']}** "
                            f"| 📂 `{src['folder']}`"
                        )
                        st.caption(src["content"] + "...")

    # Question input
    question = st.chat_input("Ask a question...")

    if question:
        if not st.session_state.selected_folders:
            st.error("Please select at least one folder from sidebar first!")
        elif not os.path.exists(FAISS_INDEX_PATH):
            st.error("Please index documents first using the sidebar button!")
        else:
            with st.chat_message("user"):
                st.write(question)

            with st.chat_message("assistant"):
                with st.spinner("Searching..."):
                    result = ask_question(
                        question       = question,
                        access_folders = st.session_state.selected_folders
                    )

                st.write(result["answer"])

                if result["sources"]:
                    with st.expander(
                        f"📎 {len(result['sources'])} Sources Used"
                    ):
                        for src in result["sources"]:
                            st.write(
                                f"📄 **{src['filename']}** "
                                f"| 📂 `{src['folder']}`"
                            )
                            st.caption(src["content"] + "...")
                else:
                    st.warning(
                        "No chunks found in selected folders for this query."
                    )

            st.session_state.chat_history.append({
                "question": question,
                "answer"  : result["answer"],
                "sources" : result["sources"]
            })

# ──────────────────────────────────────────
# TAB 2 — FOLDER STRUCTURE
# ──────────────────────────────────────────
with tab2:
    st.subheader("📁 University Folder Structure")

    structure = get_folder_structure()

    if not structure:
        st.warning(
            "No folders found. "
            "Create university/ folder with PDFs and click Index."
        )
        st.code("""
university/
├── academic/
│     ├── CSE/       ← put CSE pdfs here
│     ├── ECE/       ← put ECE pdfs here
│     ├── Mechanical/
│     ├── Civil/
│     └── MBA/
└── administration/  ← put admin pdfs here
        """)
    else:
        for folder, files in structure.items():
            selected = folder in st.session_state.selected_folders
            icon     = "🔓" if selected else "📂"

            with st.expander(
                f"{icon} {folder} — {len(files)} files"
            ):
                for f in files:
                    st.write(f"  📄 {f}")
