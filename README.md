# TrustQuery

TrustQuery is a production-oriented Security Questionnaire Evidence Copilot.

The system helps B2B SaaS teams answer security and compliance questionnaires using evidence retrieved from internal policy and security documentation.

## Problem

Security questionnaires often contain hundreds of questions covering topics such as:

- Access control
- Encryption
- Incident response
- Backup and disaster recovery
- Vendor risk management
- Security awareness training
- Data retention

Answering these questionnaires manually requires teams to search through policies and supporting documentation, which is slow and error-prone.

TrustQuery retrieves relevant evidence and generates citation-backed draft responses for human review.

## Planned Architecture

- Document ingestion for PDF and Markdown files
- Document chunking and metadata extraction
- Sentence Transformer embeddings
- ChromaDB vector storage
- BM25 keyword retrieval
- Hybrid retrieval
- Cross-encoder reranking
- Groq LLM generation
- Citation enforcement
- Human review workflow
- Offline evaluation
- GitHub Actions CI quality gates

## Status

Under active development.