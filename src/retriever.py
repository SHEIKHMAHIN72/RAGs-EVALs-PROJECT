"""import os
import re
import glob

from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

load_dotenv()  # loads OPENAI_API_KEY from .env

DATA_DIR = "data"
DB_DIR = "chroma_store"


# 1. LOAD ---- read each transcript, throw away the VTT timestamps
def load_transcripts():

    docs = []
    for path in glob.glob(f"{DATA_DIR}/*.vtt"):
        lines = []
        for line in open(path):
            line = line.strip()
            if not line or line == "WEBVTT" or "-->" in line:
                continue
            lines.append(line)
        text = " ".join(lines)

        session = re.search(r"Session[ _]*(\d+)", path).group(1)

        docs.append(Document(page_content=text, metadata={"session": session}))

    return docs


# 2. BUILD ---- chunk, embed once, and keep it on disk so we don't re-embed
def load_store():
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

    if os.path.exists(DB_DIR):
        return Chroma(persist_directory=DB_DIR, embedding_function=embeddings)

    docs = load_transcripts()

    chunks = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
    ).split_documents(docs)

    return Chroma.from_documents(chunks, embeddings, persist_directory=DB_DIR)


def build_retriever():
    return load_store().as_retriever(search_kwargs={"k": 5})


# 3. TRY IT ---- python src/retriever.py
if __name__ == "__main__":

    retriever = build_retriever()

    results = retriever.invoke("what is regression testing?")
    
    for r in results:
        print(f"[Session {r.metadata['session']}] {r.page_content[:150]}...\n")"""


import re
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ============================================================
# CONFIGURATION
# ============================================================

# Project root: parent directory of src/
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Directory containing WebVTT transcript files
DATA_DIR = PROJECT_ROOT / "data"

# Directory where ChromaDB persists its data
DB_DIR = PROJECT_ROOT / "chroma_store"

# Chroma collection name
COLLECTION_NAME = "langchain"

# Google Gemini embedding model
EMBEDDING_MODEL = "gemini-embedding-001"

# Text chunking configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

# Number of chunks returned for each query
RETRIEVAL_K = 5


# Load environment variables from the project root .env file
load_dotenv(PROJECT_ROOT / ".env")


# ============================================================
# 1. LOAD TRANSCRIPTS
# ============================================================

def load_transcripts() -> list[Document]:
    """
    Load .vtt transcript files from the data directory.

    Removes WebVTT headers, timestamps, cue settings,
    and common formatting tags.

    Returns:
        A list of LangChain Document objects.
    """

    print("\n" + "=" * 60)
    print("STEP 1: LOADING TRANSCRIPTS")
    print("=" * 60)

    if not DATA_DIR.exists():
        raise FileNotFoundError(
            f"Data directory does not exist: {DATA_DIR}"
        )

    transcript_files = sorted(
        path
        for path in DATA_DIR.iterdir()
        if path.is_file() and path.suffix.lower() == ".vtt"
    )

    print(f"Data directory: {DATA_DIR}")
    print(f"VTT files found: {len(transcript_files)}")

    if not transcript_files:
        raise FileNotFoundError(
            f"No .vtt transcript files found in {DATA_DIR}"
        )

    documents = []

    for file_path in transcript_files:
        print(f"\nReading: {file_path.name}")

        # utf-8-sig also handles a UTF-8 byte-order mark.
        with file_path.open(
            "r",
            encoding="utf-8-sig",
            errors="replace",
        ) as file:
            raw_text = file.read()

        cleaned_lines = []

        for line in raw_text.splitlines():
            line = line.strip()

            # Skip empty lines and the WebVTT header.
            if not line or line.upper() == "WEBVTT":
                continue

            # Skip metadata and timestamp lines.
            if line.startswith(
                ("NOTE", "STYLE", "REGION")
            ):
                continue

            # Skip timestamp cues, including optional cue settings.
            if "-->" in line:
                continue

            # Skip standalone numeric cue identifiers.
            if re.fullmatch(r"\d+", line):
                continue

            # Remove common WebVTT formatting tags.
            line = re.sub(
                r"</?(?:c(?:\.[^ >]+)*|v(?:\s+[^>]*)?|"
                r"lang(?:\s+[^>]*)?|b|i|u|ruby|rt)>",
                "",
                line,
                flags=re.IGNORECASE,
            )

            # Remove remaining HTML-like tags.
            line = re.sub(r"<[^>]+>", "", line)

            line = line.strip()

            if line:
                cleaned_lines.append(line)

        transcript_text = " ".join(cleaned_lines).strip()

        if not transcript_text:
            print("WARNING: No transcript text extracted; skipping.")
            continue

        # Extract the session number from the filename.
        match = re.search(
            r"Session[ _]*(\d+)",
            file_path.stem,
            flags=re.IGNORECASE,
        )

        session = match.group(1) if match else "unknown"

        document = Document(
            page_content=transcript_text,
            metadata={
                "session": session,
                "source": file_path.name,
            },
        )

        documents.append(document)

        print(f"Session: {session}")
        print(f"Extracted characters: {len(transcript_text):,}")
        print(f"Preview: {transcript_text[:150]}...")

    print(f"\nTranscript documents loaded: {len(documents)}")

    if not documents:
        raise ValueError(
            "Transcript files were found, but no usable transcript "
            "text could be extracted."
        )

    return documents


# ============================================================
# 2. INITIALIZE GEMINI EMBEDDINGS
# ============================================================

def create_embeddings() -> GoogleGenerativeAIEmbeddings:
    """
    Create the Gemini embedding model.

    Requires GOOGLE_API_KEY or GEMINI_API_KEY in the environment,
    depending on the installed integration version.
    """

    print("\n" + "=" * 60)
    print("STEP 2: INITIALIZING GEMINI EMBEDDINGS")
    print("=" * 60)

    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL
    )


# ============================================================
# 3. SPLIT DOCUMENTS INTO CHUNKS
# ============================================================

def split_documents(
    documents: list[Document],
) -> list[Document]:
    """
    Split transcript documents into smaller overlapping chunks.
    """

    print("\n" + "=" * 60)
    print("STEP 3: SPLITTING TRANSCRIPTS")
    print("=" * 60)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(documents)

    print(f"Chunk size: {CHUNK_SIZE}")
    print(f"Chunk overlap: {CHUNK_OVERLAP}")
    print(f"Total chunks generated: {len(chunks)}")

    if not chunks:
        raise ValueError(
            "The text splitter did not generate any chunks."
        )

    return chunks


# ============================================================
# 4. LOAD OR BUILD THE CHROMA VECTOR STORE
# ============================================================

def load_store() -> Chroma:
    """
    Open the existing Chroma collection.

    If the collection is empty, index the transcript files.
    If it already contains data, reuse it.

    This avoids relying on directory existence alone to decide
    whether indexing has already happened.
    """

    print("\n" + "=" * 60)
    print("STEP 4: INITIALIZING CHROMA")
    print("=" * 60)

    DB_DIR.mkdir(parents=True, exist_ok=True)

    embeddings = create_embeddings()

    store = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(DB_DIR),
        embedding_function=embeddings,
    )

    existing_count = store._collection.count()

    print(f"Chroma directory: {DB_DIR}")
    print(f"Collection name: {COLLECTION_NAME}")
    print(f"Existing chunks: {existing_count}")

    # Reuse an already-populated collection.
    if existing_count > 0:
        print("Existing vector data found. Reusing the collection.")
        return store

    print("The collection is empty. Starting transcript indexing.")

    # Load and split the transcript files.
    documents = load_transcripts()
    chunks = split_documents(documents)

    print("\nGenerating embeddings and storing chunks...")
    print("This may take some time and may incur API usage.")

    # Store the chunks. LangChain calls the embedding model
    # and adds the resulting vectors to Chroma.
    store.add_documents(chunks)

    final_count = store._collection.count()

    print(f"\nChunks generated: {len(chunks)}")
    print(f"Chunks stored in Chroma: {final_count}")

    if final_count == 0:
        raise RuntimeError(
            "Indexing finished, but Chroma still contains zero chunks."
        )

    print("Transcript indexing completed successfully.")

    return store


# ============================================================
# 5. BUILD THE RETRIEVER
# ============================================================

def build_retriever():
    """
    Create a retriever that returns the most relevant transcript chunks.
    """

    store = load_store()

    return store.as_retriever(
        search_kwargs={"k": RETRIEVAL_K}
    )


# ============================================================
# 6. INSPECT THE VECTOR DATABASE
# ============================================================

def show_database_stats(store: Chroma) -> None:
    """
    Display basic statistics about the Chroma collection.
    """

    print("\n" + "=" * 60)
    print("VECTOR DATABASE STATISTICS")
    print("=" * 60)

    print(f"Database directory: {DB_DIR}")
    print(f"Collection name: {COLLECTION_NAME}")
    print(f"Total stored chunks: {store._collection.count()}")

    sample = store.get(
        limit=3,
        include=["metadatas"],
    )

    metadatas = sample.get("metadatas") or []

    if metadatas:
        print("\nSample chunk metadata:")

        for index, metadata in enumerate(metadatas, start=1):
            print(f"{index}. {metadata}")


# ============================================================
# 7. TEST RETRIEVAL
# ============================================================

def test_retrieval() -> None:
    """
    Build the retriever and run a sample question.
    """

    retriever = build_retriever()

    store = load_store()
    show_database_stats(store)

    question = "What is regression testing?"

    print("\n" + "=" * 60)
    print("RETRIEVAL TEST")
    print("=" * 60)

    print(f"Question: {question}\n")

    results = retriever.invoke(question)

    if not results:
        print("No relevant chunks were returned.")
        return

    print(f"Retrieved chunks: {len(results)}\n")

    for index, document in enumerate(results, start=1):
        session = document.metadata.get("session", "unknown")
        source = document.metadata.get("source", "unknown")

        print("-" * 60)
        print(f"Result: {index}")
        print(f"Session: {session}")
        print(f"Source: {source}")
        print("-" * 60)
        print(document.page_content)
        print()


# ============================================================
# 8. ENTRY POINT
# ============================================================

if __name__ == "__main__":
    try:
        test_retrieval()

    except Exception as error:
        print("\n" + "=" * 60)
        print("ERROR")
        print("=" * 60)
        print(f"{type(error).__name__}: {error}")
        raise
