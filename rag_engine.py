import os
from typing import List, Optional

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

FAISS_INDEX_PATH     = "./vector_store"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL_NAME      = "llama-3.3-70b-versatile"
groq_api_key = "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"



def ask_question(
    question    : str,
    access_dirs : List[str],
    groq_api_key: str,
    scope_type  : str           = "all",   # ← ADDED
    scope_value : Optional[str] = None     # ← ADDED
) -> dict:
    """
    scope_type = "all"    → search all accessible docs
    scope_type = "folder" → search only chosen folder
    scope_type = "file"   → search only chosen PDF file
    """
    os.environ["GROQ_API_KEY"] = groq_api_key

    if not os.path.exists(FAISS_INDEX_PATH):
        return {
            "answer"        : "Index not found. Please index documents first.",
            "retrieved_docs": []
        }

    llm         = ChatGroq(model=GROQ_MODEL_NAME, temperature=0.1)
    embeddings  = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    vectorstore = FAISS.load_local(
        FAISS_INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

    # ─────────────────────────────────────────────────────
    # RBAC FILTER
    # Step 1 — Always check base access first
    # Step 2 — Then apply scope on top
    # ─────────────────────────────────────────────────────
    def rbac_filter(metadata: dict) -> bool:
        chunk_folder   = metadata.get("read_access", "")
        chunk_filename = metadata.get("filename",    "")

        # STEP 1 — Base access check
        # Does this chunk belong to a folder user can access?
        has_base_access = any(
            chunk_folder.startswith(allowed_dir)
            for allowed_dir in access_dirs
        )

        if not has_base_access:
            return False  # blocked regardless of scope

        # STEP 2 — Scope filter on top of access
        if scope_type == "all":
            # All accessible docs
            return True

        elif scope_type == "folder":
            # Only chunks from selected folder
            return chunk_folder == scope_value

        elif scope_type == "file":
            # Only chunks from selected PDF file
            return chunk_filename == scope_value

        return False

    retriever = vectorstore.as_retriever(
        search_kwargs={"filter": rbac_filter, "k": 5}
    )

    system_prompt = (
        "You are a university assistant. "
        "Use the following retrieved context "
        "to answer the student's question. "
        "If the answer is not in the context, "
        "say you don't have that information "
        "in the selected documents.\n\n"
        "<context>\n{context}\n</context>"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    qa_chain  = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, qa_chain)

    response       = rag_chain.invoke({"input": question})
    retrieved_docs = response.get("context", [])

    return {
        "answer"        : response["answer"],
        "retrieved_docs": [
            {
                "file"   : doc.metadata.get("filename", "unknown"),
                "folder" : doc.metadata.get("folder",   "unknown"),
                "content": doc.page_content.strip()[:300]
            }
            for doc in retrieved_docs
        ]
    }
