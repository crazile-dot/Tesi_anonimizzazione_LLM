from sentence_transformers import SentenceTransformer


MODEL_NAME = "all-MiniLM-L6-v2"


print("Caricamento modello...")
model = SentenceTransformer(MODEL_NAME)

text = "This is a test document for semantic similarity."

print("Generazione embedding...")
embedding = model.encode(text)

print("Embedding generato!")
print("Dimensione:", len(embedding))
print("Prime 5 componenti:", embedding[:5])