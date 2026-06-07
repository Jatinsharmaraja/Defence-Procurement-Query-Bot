from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from tqdm import tqdm
import os

# 1. Configuration
MANUALS_DIR = './manuals'
VAULT_NAME = "permanent_vault"
EMBED_MODEL = "nomic-embed-text"

# 2. Check for Manuals
if not os.path.exists(MANUALS_DIR):
    os.makedirs(MANUALS_DIR)
    print(f"❌ Error: Please put your 6 PDFs in the '{MANUALS_DIR}' folder.")
    exit()

# 3. Load All PDFs
print(f"📂 Phase 1: Reading manuals from {MANUALS_DIR}...")
loader = DirectoryLoader(MANUALS_DIR, glob="./*.pdf", loader_cls=PyPDFLoader)
docs = loader.load()

# 4. Add Metadata (Manual names)
for doc in docs:
    doc.metadata["source"] = os.path.basename(doc.metadata["source"])

# 5. Neural Chunking
print(f"✂️ Phase 2: Splitting {len(docs)} pages into segments...")
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)

# 6. Build Permanent Vault with Progress Bar
print(f"🧠 Phase 3: Building Neural Brain from {len(chunks)} chunks...")
embeddings = OllamaEmbeddings(model=EMBED_MODEL)
batch_size = 50
vector_db = None

for i in tqdm(range(0, len(chunks), batch_size), desc="Embedding Progress"):
    batch = chunks[i : i + batch_size]
    if vector_db is None:
        vector_db = FAISS.from_documents(batch, embeddings)
    else:
        vector_db.add_documents(batch)

# 7. Save
vector_db.save_local(VAULT_NAME)
print(f"\n✅ SUCCESS: '{VAULT_NAME}' created! You can now run 'app.py'.")