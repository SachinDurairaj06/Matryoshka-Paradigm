from sentence_transformers import SentenceTransformer
import chromadb

# local embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# start local vector DB (creates folder automatically)
client = chromadb.Client()
collection = client.create_collection("memory")

texts = [
    "Matryoshka memory stores layered representations",
    "Vector databases store embeddings",
    "Graph memory connects related ideas"
]

embeddings = model.encode(texts).tolist()

collection.add(
    documents=texts,
    embeddings=embeddings,
    ids=["1", "2", "3"]
)

query = "How does memory connect ideas?"

q_emb = model.encode([query]).tolist()

results = collection.query(
    query_embeddings=q_emb,
    n_results=2
)

print(results["documents"])
