from trustquery.embeddings.embedder import embed_texts


def test_embed_texts_returns_embeddings():
    texts = [
        "Multi-factor authentication is required.",
        "Backups are performed daily.",
    ]

    embeddings = embed_texts(texts)

    assert len(embeddings) == 2
    assert len(embeddings[0]) > 0
    assert isinstance(embeddings[0][0], float)


def test_embed_texts_empty_input():
    embeddings = embed_texts([])

    assert embeddings == []