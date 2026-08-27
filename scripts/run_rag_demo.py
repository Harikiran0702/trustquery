from trustquery.embeddings.embedder import embed_texts
from trustquery.generation.answer_generator import generate_grounded_answer
from trustquery.ingestion.chunker import chunk_documents
from trustquery.ingestion.loader import load_pdf
from trustquery.retrieval.hybrid_retriever import retrieve_and_rerank
from trustquery.retrieval.reranker import load_reranker
from trustquery.vectorstore.chroma_store import add_chunks, get_collection


PDF_PATH = "data/sample/access_control_policy.pdf"


def main():
    # 1. Load the PDF
    documents = load_pdf(PDF_PATH)

    print("\nDEBUG DOCUMENTS:")
    for index, document in enumerate(documents):
        print(f"Document {index}: {document}")

    # 2. Break documents into chunks
    chunks = chunk_documents(documents)

    print("\nDEBUG CHUNKS:")
    for index, chunk in enumerate(chunks):
        print(f"Chunk {index}: {chunk}")

    # 3. Generate embeddings
    texts = [chunk["text"] for chunk in chunks]
    embeddings = embed_texts(texts)

    # 4. Create demo collection
    collection = get_collection(
        collection_name="trustquery_demo",
        persist_directory=".chroma_demo",
    )

    # Clear previous demo data
    existing = collection.get()
    if existing["ids"]:
        collection.delete(ids=existing["ids"])

    # 5. Store chunks
    add_chunks(
        collection=collection,
        chunks=chunks,
        embeddings=embeddings,
    )

    # 6. Load CrossEncoder reranker
    reranker_model = load_reranker()

    # 7. Ask question
    question = "Does the company require multi-factor authentication?"

    # 8. Hybrid retrieve + rerank
    retrieved_chunks = retrieve_and_rerank(
        query=question,
        collection=collection,
        chunks=chunks,
        reranker_model=reranker_model,
        top_k=2,
        candidate_k=10,
    )

    print("\nRETRIEVED EVIDENCE:")

    for index, chunk in enumerate(retrieved_chunks, start=1):
        print(f"\nResult {index}")
        print("Source:", chunk.get("metadata", {}).get("source"))
        print("Page:", chunk.get("metadata", {}).get("page"))
        print("RRF score:", chunk.get("rrf_score"))
        print("Reranker score:", chunk.get("reranker_score"))
        print("Text:", chunk["text"])

    # 9. Generate grounded Gemini answer
    answer = generate_grounded_answer(
        question=question,
        retrieved_chunks=retrieved_chunks,
    )

    print("\nTRUSTQUERY ANSWER:")
    print(answer)


if __name__ == "__main__":
    main()
