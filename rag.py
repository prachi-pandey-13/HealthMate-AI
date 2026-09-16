import os
import faiss
from sentence_transformers import SentenceTransformer


# -----------------------------
# Load Knowledge Base
# -----------------------------
def load_knowledge_base():

    file_path = os.path.join(
        "knowledge_base",
        "health_guidelines.txt"
    )

    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()

    return text


# -----------------------------
# Split Text into Chunks
# -----------------------------
def create_chunks(text):

    paragraphs = text.split("\n\n")

    chunks = []

    for paragraph in paragraphs:
        paragraph = paragraph.strip()

        if paragraph:
            chunks.append(paragraph)

    return chunks


# -----------------------------
# Create Embeddings
# -----------------------------
def create_embeddings(chunks):

    model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    embeddings = model.encode(
        chunks,
        convert_to_numpy=True
    )

    return model, embeddings


# -----------------------------
# Create FAISS Index
# -----------------------------
def create_index(embeddings):

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(embeddings)

    return index


# -----------------------------
# Retrieve Relevant Information
# -----------------------------
def retrieve_information(
    question,
    chunks,
    model,
    index,
    top_k=3
):

    question_embedding = model.encode(
        [question],
        convert_to_numpy=True
    )

    distances, indices = index.search(
        question_embedding,
        top_k
    )

    results = []

    for index_number in indices[0]:

        if index_number < len(chunks):
            results.append(
                chunks[index_number]
            )

    return results


# -----------------------------
# Test RAG
# -----------------------------
if __name__ == "__main__":

    text = load_knowledge_base()

    chunks = create_chunks(text)

    model, embeddings = create_embeddings(chunks)

    index = create_index(embeddings)

    question = input(
        "Ask a health question: "
    )

    results = retrieve_information(
        question,
        chunks,
        model,
        index
    )

    print("\nRelevant Information:\n")

    for result in results:
        print(result)
        print("-" * 50)