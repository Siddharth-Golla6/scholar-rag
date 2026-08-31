# Retrieval-Augmented Generation (RAG)

Retrieval-Augmented Generation grounds a language model's output in an external knowledge source
instead of relying only on parameters learned during training. A RAG pipeline has two stages.

First, documents are split into chunks, encoded into dense vector embeddings, and stored in a vector
database. At query time the question is embedded and the most similar chunks are retrieved. Retrieval
is often refined by re-ranking techniques such as Maximal Marginal Relevance (MMR), which balances
relevance to the query against diversity so that near-duplicate passages do not crowd out useful ones.

Second, the retrieved passages are inserted into the prompt and the language model generates an answer
conditioned on them, ideally citing the passages it used. Because claims can be traced to sources, RAG
reduces hallucination, and it lets a system answer questions about private or recent documents without
retraining.

Key design choices include chunk size and overlap, the embedding model, the number of passages
retrieved (top-k), and guardrails that instruct the model to say it does not know when the context is
insufficient. Production systems add evaluation, measuring faithfulness (is every claim grounded in the
retrieved context?) and answer relevance, so quality is measured rather than assumed. Agentic RAG
extends this by letting the model call tools, such as a web or arXiv search, when the local corpus
cannot answer the question.
