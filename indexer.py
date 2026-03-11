import os
import json
import uuid
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict

from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

FAISS_INDEX_PATH     = "./vector_store"
MANIFEST_PATH        = "./manifest.json"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
UNIVERSITY_ROOT      = "./university"
CHUNK_SIZE           = 1000
CHUNK_OVERLAP        = 200

@dataclass
class DocumentRecord:
    file_id      : str
    name         : str
    folder       : str          # e.g. "academic/CSE"
    read_access  : List[str]    # same as folder for filtering
    modified_time: Optional[str] = None
    chunk_ids    : List[str] = field(default_factory=list)


def load_manifest() -> Dict[str, dict]:
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH, "r") as f:
            return json.load(f)
    return {}


def save_manifest(manifest: Dict[str, dict]):
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=4)


def scan_university_folder() -> List[DocumentRecord]:
    """
    Walks the university/ directory
    For each PDF found:
      - records which subfolder it lives in
      - that subfolder becomes its read_access tag
    """
    print("\n--- Scanning University Folder ---")
    records = []

    for root, dirs, files in os.walk(UNIVERSITY_ROOT):
        for filename in files:
            if not filename.endswith(".pdf"):
                continue

            full_path    = os.path.join(root, filename)

            # Get relative folder path from university root
            # e.g. university/academic/CSE → academic/CSE
            rel_folder   = os.path.relpath(root, UNIVERSITY_ROOT)
            rel_folder   = rel_folder.replace("\\", "/")  # Windows fix

            file_id      = f"{rel_folder}/{filename}".replace("/","_").replace(".pdf","")

            records.append(DocumentRecord(
                file_id      = file_id,
                name         = full_path,
                folder       = rel_folder,
                read_access  = [rel_folder],   # folder = access tag
                modified_time= str(os.path.getmtime(full_path))
            ))

            print(f"  Found : {filename}")
            print(f"  Folder: {rel_folder}")

    print(f"\nTotal files found: {len(records)}")
    return records


def build_index(records: List[DocumentRecord]):
    """
    Indexes all documents into FAISS
    Stores folder name in metadata for filtering
    """
    print("\n--- Building FAISS Index ---")
    embeddings    = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    manifest      = load_manifest()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size   = CHUNK_SIZE,
        chunk_overlap= CHUNK_OVERLAP,
        separators   = ["\n\n", "\n", " ", ""]
    )

    # Load or create vectorstore
    if os.path.exists(FAISS_INDEX_PATH):
        print("Loading existing FAISS index...")
        vectorstore = FAISS.load_local(
            FAISS_INDEX_PATH, embeddings,
            allow_dangerous_deserialization=True
        )
    else:
        print("Creating new FAISS index...")
        vectorstore = None

    changes_made = False

    for record in records:

        # Smart sync
        if record.file_id in manifest:
            existing = manifest[record.file_id]
            if existing.get("modified_time") == record.modified_time:
                print(f"  SKIP   : {record.name} (up to date)")
                continue
            else:
                print(f"  UPDATE : {record.name}")
                if vectorstore and existing.get("chunk_ids"):
                    try:
                        vectorstore.delete(existing["chunk_ids"])
                    except Exception as e:
                        print(f"  Warning: {e}")
        else:
            print(f"  ADD    : {record.name}")

        if not os.path.exists(record.name):
            print(f"  ERROR  : File not found")
            continue

        # Load and split
        loader   = PyPDFLoader(record.name)
        raw_docs = loader.load()
        chunks   = text_splitter.split_documents(raw_docs)

        # Inject metadata into every chunk
        page_records = []
        for chunk in chunks:
            chunk.metadata["file_id"]     = record.file_id
            chunk.metadata["folder"]      = record.folder
            chunk.metadata["read_access"] = record.folder  # ← KEY for filtering
            chunk.metadata["filename"]    = os.path.basename(record.name)
            page_records.append(chunk)

        if page_records:
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
            print(f"  DONE   : {len(page_records)} chunks added")

    if changes_made and vectorstore:
        vectorstore.save_local(FAISS_INDEX_PATH)
        save_manifest(manifest)
        print("\nFAISS index saved!")
    else:
        print("\nNo changes made")


def get_folder_structure() -> Dict:
    """
    Returns folder structure for Streamlit sidebar display
    """
    structure = {}

    for root, dirs, files in os.walk(UNIVERSITY_ROOT):
        rel = os.path.relpath(root, UNIVERSITY_ROOT).replace("\\", "/")
        pdfs = [f for f in files if f.endswith(".pdf")]
        if pdfs:
            structure[rel] = pdfs

    return structure
