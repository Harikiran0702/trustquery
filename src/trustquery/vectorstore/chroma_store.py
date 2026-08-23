import chromadb


def get_collection(
    collection_name: str = "trustquery",
    persist_directory: str = ".chroma",
):
    client = chromadb.PersistentClient(path=persist_directory)

    collection = client.get_or_create_collection(
        name=collection_name
    )

    return collection

def add_chunks(collection, chunks, embeddings):
    if len(chunks) != len(embeddings):
        raise ValueError("Number of chunks must match number of embeddings")

    if not chunks:
        return

    ids = [f"chunk-{i}" for i in range(len(chunks))]
    documents = [chunk["text"] for chunk in chunks]

    metadatas = []
    for chunk in chunks:
        metadata = {
            key: value
            for key, value in chunk.items()
            if key != "text"
        }
        metadatas.append(metadata)

    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )

def search_chunks(collection, query_embedding, top_k=5):
    if top_k <= 0:
        raise ValueError("top_k must be greater than 0")

    if not query_embedding:
        raise ValueError("query_embedding cannot be empty")

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    retrieved_chunks = []

    for document, metadata, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        retrieved_chunks.append(
            {
                "text": document,
                "metadata": metadata,
                "distance": distance,
            }
        )

    return retrieved_chunks