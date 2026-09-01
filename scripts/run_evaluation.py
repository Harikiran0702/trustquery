from trustquery.embeddings.embedder import embed_texts
from trustquery.evaluation.dataset import load_golden_dataset
from trustquery.evaluation.evaluator import (
    evaluate_generation,
    evaluate_retrieval,
)
from trustquery.generation.answer_generator import generate_grounded_answer
from trustquery.ingestion.chunker import chunk_documents
from trustquery.ingestion.loader import load_pdf
from trustquery.retrieval.hybrid_retriever import retrieve_and_rerank
from trustquery.retrieval.reranker import load_reranker
from trustquery.vectorstore.chroma_store import add_chunks, get_collection


PDF_PATH = "data/sample/access_control_policy.pdf"
DATASET_PATH = "data/eval/golden_dataset.json"


def main():
    # 1. Load and chunk policy documents
    documents = load_pdf(PDF_PATH)
    chunks = chunk_documents(documents)

    # 2. Generate embeddings
    texts = [chunk["text"] for chunk in chunks]
    embeddings = embed_texts(texts)

    # 3. Create evaluation Chroma collection
    collection = get_collection(
        collection_name="trustquery_evaluation",
        persist_directory=".chroma_evaluation",
    )

    # Clear previous evaluation data
    existing = collection.get()
    if existing["ids"]:
        collection.delete(ids=existing["ids"])

    # 4. Store policy chunks
    add_chunks(
        collection=collection,
        chunks=chunks,
        embeddings=embeddings,
    )

    # 5. Load reranker
    reranker_model = load_reranker()

    # 6. Load golden dataset
    dataset = load_golden_dataset(DATASET_PATH)

    # 7. Define real TrustQuery retrieval function
    def retrieve(question):
        return retrieve_and_rerank(
            query=question,
            collection=collection,
            chunks=chunks,
            reranker_model=reranker_model,
            top_k=2,
            candidate_k=10,
        )

    # 8. Evaluate retrieval
    report = evaluate_retrieval(
        dataset=dataset,
        retrieve_fn=retrieve,
    )

    generation_report = evaluate_generation(
        dataset=dataset,
        retrieve_fn=retrieve,
        generate_fn=generate_grounded_answer,
    )

    # 9. Print retrieval evaluation report
    print("\nTrustQuery Retrieval Evaluation")
    print("=" * 40)

    for result in report["results"]:
        if not result["should_answer"]:
            status = "N/A"
        else:
            status = "PASS" if result["retrieval_hit"] else "FAIL"

        print(
            f"{result['id']} | "
            f"{status} | "
            f"{result['question']}"
        )

    answerable = sum(
        1
        for result in report["results"]
        if result["should_answer"]
    )

    hits = sum(
        1
        for result in report["results"]
        if result["should_answer"]
        and result["retrieval_hit"]
    )

    print("\nSummary")
    print("-" * 40)
    print(f"Answerable questions: {answerable}")
    print(f"Retrieval hits:       {hits}")
    print(
        f"Retrieval hit rate:   "
        f"{report['retrieval_hit_rate']:.2%}"
    )

    # 10. Generate answers for baseline evaluation
    print("\nTrustQuery Generation Evaluation")
    print("=" * 40)

    for result in generation_report["results"]:
        status = "PASS" if result["abstention_correct"] else "FAIL"

        print(
            f"{result['id']} | "
            f"{status} | "
            f"should_answer={result['should_answer']}"
        )

        print(f"Answer: {result['answer']}")
        print()

    print("Generation Summary")
    print("-" * 40)
    print(
        f"Abstention accuracy: "
        f"{generation_report['abstention_accuracy']:.2%}"
    )


if __name__ == "__main__":
    main()