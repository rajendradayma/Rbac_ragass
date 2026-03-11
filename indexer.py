import os
import json
import uuid
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict

from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ============================================================
# CONFIGS
# ============================================================
FAISS_INDEX_PATH     = "./vector_store"
MANIFEST_PATH        = "./manifest.json"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
UNIVERSITY_ROOT      = "./university"
CHUNK_SIZE           = 1000
CHUNK_OVERLAP        = 200

# ============================================================
# DATACLASS
# ============================================================
@dataclass
class DocumentRecord:
    file_id      : str
    name         : str
    folder       : str
    modified_time: Optional[str] = None
    chunk_ids    : List[str] = field(default_factory=list)

# ============================================================
# MANIFEST
# ============================================================
def load_manifest() -> Dict[str, dict]:
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH, "r") as f:
            return json.load(f)
    return {}

def save_manifest(manifest: Dict[str, dict]):
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=4)

# ============================================================
# SCAN UNIVERSITY FOLDER
# ============================================================
def scan_university_folder() -> List[DocumentRecord]:
    """
    Walks university/ folder
    For every PDF found:
      records which subfolder it lives in
      that subfolder becomes its access tag
    """
    print("\n--- Scanning University Folder ---")
    records = []

    for root, dirs, files in os.walk(UNIVERSITY_ROOT):
        for filename in files:
            if not filename.endswith(".pdf"):
                continue

            full_path  = os.path.join(root, filename)

            # Get relative path from university root
            # university/academic/CSE → academic/CSE
            rel_folder = os.path.relpath(root, UNIVERSITY_ROOT)
            rel_folder = rel_folder.replace("\\", "/")

            file_id    = f"{rel_folder}_{filename}".replace("/","_").replace(".pdf","")

            records.append(DocumentRecord(
                file_id      = file_id,
                name         = full_path,
                folder       = rel_folder,
                modified_time= str(os.path.getmtime(full_path))
            ))

            print(f"  Found  : {filename}")
            print(f"  Folder : {rel_folder}")
            print(f"  ID     : {file_id}")
            print()

    print(f"Total files found: {len(records)}")
    return records

# ============================================================
# BUILD FAISS INDEX
# ============================================================
def build_index(records: List[DocumentRecord]):
    """
    Indexes all documents into FAISS
    Stores folder name in metadata for RBAC filtering later
    """
    print("\n--- Building FAISS Index ---")
    embeddings    = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    manifest      = load_manifest()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size   = CHUNK_SIZE,
        chunk_overlap= CHUNK_OVERLAP,
        separators   = ["\n\n", "\n", " ", ""]
    )

    if os.path.exists(FAISS_INDEX_PATH):
        print("Loading existing FAISS index...")
        vectorstore = FAISS.load_local(
            FAISS_INDEX_PATH, embeddings,
            allow_dangerous_deserialization=True
        )
    else:
        print("No existing index. Creating fresh...")
        vectorstore = None

    changes_made = False

    for record in records:
        print(f"\nProcessing : {record.name}")
        print(f"Folder     : {record.folder}")

        # Smart sync check
        if record.file_id in manifest:
            existing = manifest[record.file_id]
            if existing.get("modified_time") == record.modified_time:
                print(f"Status     : SKIPPED (up to date)")
                continue
            else:
                print(f"Status     : UPDATING (file changed)")
                if vectorstore and existing.get("chunk_ids"):
                    try:
                        vectorstore.delete(existing["chunk_ids"])
                    except Exception as e:
                        print(f"Warning    : {e}")
        else:
            print(f"Status     : NEW FILE")

        if not os.path.exists(record.name):
            print(f"Status     : ERROR - file not found")
            continue

        # Load and split
        loader   = PyPDFLoader(record.name)
        raw_docs = loader.load()
        chunks   = text_splitter.split_documents(raw_docs)

        if not chunks:
            print(f"Status     : SKIPPED - no text extracted")
            continue

        # Inject metadata into every chunk
        page_records = []
        for chunk in chunks:
            chunk.metadata["file_id"]     = record.file_id
            chunk.metadata["folder"]      = record.folder   # ← for filtering
            chunk.metadata["filename"]    = os.path.basename(record.name)
            page_records.append(chunk)

        # Add to FAISS
        if vectorstore is None:
            new_ids     = [str(uuid.uuid4()) for _ in page_records]
            vectorstore = FAISS.from_documents(
                page_records, embeddings, ids=new_ids
            )
        else:
            new_ids = vectorstore.add_documents(page_records)

        record.chunk_ids         = new_ids
        manifest[record.file_id] = asdict(record)
        changes_made             = True
        print(f"Status     : DONE - {len(page_records)} chunks added")

    if changes_made and vectorstore:
        vectorstore.save_local(FAISS_INDEX_PATH)
        save_manifest(manifest)
        print("\nFAISS index and manifest saved!")
    else:
        print("\nNo changes made.")

# ============================================================
# GET FOLDER STRUCTURE
# ============================================================
def get_folder_structure() -> Dict[str, List[str]]:
    """
    Returns dict of folder → list of pdf files
    Used by Streamlit to show folder tree
    """
    structure = {}

    for root, dirs, files in os.walk(UNIVERSITY_ROOT):
        rel    = os.path.relpath(root, UNIVERSITY_ROOT).replace("\\", "/")
        pdfs   = [f for f in files if f.endswith(".pdf")]
        if pdfs:
            structure[rel] = pdfs

    return structure

# ============================================================
# RUN DIRECTLY TO TEST
# ============================================================
if __name__ == "__main__":
    records = scan_university_folder()
    build_index(records)

    print("\n--- Folder Structure ---")
    structure = get_folder_structure()
    for folder, files in structure.items():
        print(f"\n📂 {folder}")
        for f in files:
            print(f"   📄 {f}")
