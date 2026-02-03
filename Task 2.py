import os
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

# -----------------------------
# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHUNKS_DIR = os.path.join(BASE_DIR, "txt_output")
DB_DIR = os.path.join(BASE_DIR, "vector_db")

os.makedirs(DB_DIR, exist_ok=True)

# -----------------------------
# Load documents (CLEANED)
documents = []

for file_name in os.listdir(CHUNKS_DIR):
    if file_name.endswith(".txt"):
        with open(os.path.join(CHUNKS_DIR, file_name), "r", encoding="utf-8") as f:
            content = f.read()

        chunks = content.split("--- Chunk")
        for chunk in chunks:
            chunk = chunk.strip()

            # remove numbering noise
            chunk = chunk.replace("\n", " ")
            if len(chunk) > 100:
                documents.append(
                    Document(
                        page_content=chunk,
                        metadata={"source": file_name}
                    )
                )

print(f"✅ Total chunks loaded: {len(documents)}")

# -----------------------------
# Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# -----------------------------
# Vector DB
vector_store = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    persist_directory=DB_DIR
)

print("✅ Vector database ready")

# -----------------------------
# DYNAMIC SEARCH
while True:
    query = input("\n🔎 Enter your search query (type 'exit' to quit): ").strip()

    if query.lower() == "exit":
        print("👋 Exiting")
        break

    results = vector_store.similarity_search(query, k=3)

    print("\n📌 Top Relevant Results:\n")

    for i, doc in enumerate(results, 1):
        print(f"Result {i}")
        print(f"Source: {doc.metadata['source']}")
        print(doc.page_content[:400])
        print("-" * 80)












# import os
# from langchain_community.vectorstores import Chroma
# from langchain_core.documents import Document
# from langchain_huggingface import HuggingFaceEmbeddings

# # -----------------------------
# # Paths
# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# CHUNKS_DIR = os.path.join(BASE_DIR, "txt_output")   # Output of Task 1
# DB_DIR = os.path.join(BASE_DIR, "vector_db")

# os.makedirs(DB_DIR, exist_ok=True)

# # -----------------------------
# # Load documents
# documents = []

# for file_name in os.listdir(CHUNKS_DIR):
#     if file_name.endswith(".txt"):
#         file_path = os.path.join(CHUNKS_DIR, file_name)

#         with open(file_path, "r", encoding="utf-8") as f:
#             content = f.read()

#         chunks = content.split("--- Chunk")
#         for chunk in chunks:
#             chunk = chunk.strip()
#             if chunk:
#                 documents.append(
#                     Document(
#                         page_content=chunk,
#                         metadata={"source": file_name}
#                     )
#                 )

# print(f"✅ Total chunks loaded: {len(documents)}")

# # -----------------------------
# # Embedding model
# embeddings = HuggingFaceEmbeddings(
#     model_name="sentence-transformers/all-MiniLM-L6-v2"
# )

# # -----------------------------
# # Create / Load Vector DB
# vector_store = Chroma.from_documents(
#     documents=documents,
#     embedding=embeddings,
#     persist_directory=DB_DIR
# )

# vector_store.persist()
# print("✅ Vector database created successfully")

# # -----------------------------
# # DYNAMIC SEARCH 
# while True:
#     query = input("\n🔎 Enter your search query (type 'exit' to quit): ").strip()

#     if query.lower() == "exit":
#         print("👋 Exiting search")
#         break

#     results = vector_store.similarity_search(query, k=3)

#     print("\n📌 Top Relevant Results:\n")

#     for i, doc in enumerate(results, start=1):
#         print(f"Result {i}")
#         print(f"Source: {doc.metadata['source']}")
#         print(doc.page_content[:500])
#         print("-" * 80)