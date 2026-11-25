/** Text of `samples/rag_handbook.pdf`: one array per PDF page of `[kind, text]` blocks (h1/h2/p). */
export type Block = readonly [kind: "h1" | "h2" | "p", text: string];

export const TITLE = "The Retrieval Augmented Generation Handbook";

export const PAGES: readonly (readonly Block[])[] = [
  [
    ["h1", "Foundations of RAG"],
    ["h2", "What RAG Is"],
    [
      "p",
      "Retrieval-Augmented Generation, usually shortened to RAG, is a design pattern that gives a large language model access to external knowledge at answer time. Instead of relying only on what the model memorised during training, a RAG system first retrieves relevant passages from a document collection, then augments the prompt with those passages, and finally generates an answer grounded in them. The three steps are therefore called retrieve, augment and generate.",
    ],
    ["h2", "Why It Matters"],
    [
      "p",
      "RAG exists because language models have three well known weaknesses. Their knowledge is frozen at the training cutoff, they cannot see private company documents, and they sometimes hallucinate plausible but false statements. By supplying evidence in the prompt and asking the model to cite it, RAG reduces hallucination and makes answers verifiable.",
    ],
    ["h2", "Two Phases"],
    [
      "p",
      "A RAG system has two phases. The offline indexing phase runs once per document: the PDF is parsed, split into chunks, each chunk is converted into an embedding vector, and the vectors are stored in a vector database. The online query phase runs for every question: the question is embedded, the closest chunks are retrieved, a grounded prompt is assembled, and the language model writes the final answer.",
    ],
    ["h2", "RAG Versus Fine-Tuning"],
    [
      "p",
      "Fine-tuning is the main alternative to RAG. Fine-tuning changes the model weights and is good for teaching style or format, while RAG changes the prompt and is better for facts that change often, because updating a document requires no retraining.",
    ],
  ],
  [
    ["h1", "Indexing Chunking and Embeddings"],
    ["h2", "Chunking Strategies"],
    [
      "p",
      "Chunking is the process of cutting documents into passages small enough to embed and retrieve precisely. The simplest strategy is fixed-size chunking, which slides a window of a set number of words across the text with some overlap, so that sentences cut at a boundary still appear whole in a neighbouring chunk. A typical setting is 120 words with a 20 word overlap.",
    ],
    [
      "p",
      "Recursive chunking respects the natural structure of the text. It first splits on sections, then on paragraphs, then on sentences, and only falls back to single words when a sentence is still too long. Small pieces are merged back together until the size limit is reached. Heading-aware or semantic chunking goes one step further and keeps every section together with its heading.",
    ],
    [
      "p",
      "Parent-child chunking solves the central trade-off of chunk size. Small chunks of about 50 words produce precise embeddings, but give the language model too little context. Large chunks give context but blur the embedding. With parent-child chunking the small child chunks are indexed and searched, and when a child matches, its larger parent of about 200 words is returned to the language model.",
    ],
    ["h2", "Embeddings"],
    [
      "p",
      "An embedding model maps text to a vector so that texts with similar meaning land close together. The BGE small model, bge-small-en-v1.5, produces 384 dimensional vectors and runs comfortably on a CPU. A bi-encoder embeds the query and each passage independently, which allows all passage vectors to be computed once and stored.",
    ],
    [
      "p",
      "TF-IDF is a classic sparse alternative. It multiplies how often a word occurs in a chunk by how rare the word is across the whole collection, so common words such as the receive little weight. TF-IDF cannot match synonyms, but it needs no model download.",
    ],
  ],
  [
    ["h1", "Retrieval Fundamentals"],
    ["h2", "Vector Search and Similarity Metrics"],
    [
      "p",
      "Vector search finds the stored vectors closest to the query vector. A flat index simply compares the query against every stored vector, which is exact and fast enough for hundreds of thousands of chunks. Approximate indexes such as HNSW only become necessary at millions of vectors.",
    ],
    [
      "p",
      "Three similarity metrics are common. Cosine similarity measures the angle between two vectors and ignores their length, which makes it the default for text. The dot product also rewards vector length, and equals cosine similarity when vectors are normalised. L2 distance measures straight-line distance, so a smaller distance means a closer match.",
    ],
    ["h2", "Keyword Search with BM25"],
    [
      "p",
      "BM25 is the standard keyword ranking function. It scores a passage by summing, over the query terms, an inverse document frequency weight multiplied by a saturating term frequency. Two parameters control the behaviour: k1, usually 1.2, limits how much repeated occurrences of a word help, and b, usually 0.75, controls how strongly long passages are penalised.",
    ],
    [
      "p",
      "Keyword search is excellent for exact identifiers such as error codes, product names and numbers, which embedding models often blur. It fails when the question uses different words than the document, such as asking about an automobile when the text says car.",
    ],
  ],
  [
    ["h1", "Hybrid Search and Fusion"],
    ["h2", "Combining Two Retrievers"],
    [
      "p",
      "Hybrid search runs vector search and keyword search in parallel and merges the two ranked lists. The two retrievers fail in different ways, so their combination is more robust than either alone: embeddings capture meaning, while BM25 captures exact terms.",
    ],
    ["h2", "Weighted Fusion"],
    [
      "p",
      "The first fusion method is weighted score blending. Because cosine scores and BM25 scores live on different scales, each list is first normalised to the range zero to one. The final score is alpha times the vector score plus one minus alpha times the keyword score. An alpha of 1 means pure vector search and an alpha of 0 means pure keyword search.",
    ],
    ["h2", "Reciprocal Rank Fusion"],
    [
      "p",
      "The second method is Reciprocal Rank Fusion, or RRF. It ignores the raw scores completely and uses only the rank of each passage in each list. The RRF score of a passage is the sum over all lists of one divided by the constant k plus its rank, with k set to 60. A passage that appears near the top of both lists wins, and no score normalisation is needed. RRF is the default in most production search systems because it needs almost no tuning.",
    ],
    ["h2", "Metadata Filtering"],
    [
      "p",
      "Metadata pre-filtering narrows the search before ranking. Restricting a query to one document or a page range guarantees that all of the top results come from the allowed scope.",
    ],
  ],
  [
    ["h1", "Reranking and Query Transformation"],
    ["h2", "Cross-Encoder Reranking"],
    [
      "p",
      "A reranker is a second, more accurate scoring pass over the retrieved candidates. First-stage retrieval uses a bi-encoder, which encodes the query and passage separately. A cross-encoder reads the query and the passage together in a single forward pass and outputs a relevance score, which is far more accurate but too slow to run over the whole collection.",
    ],
    [
      "p",
      "The standard recipe is to retrieve about 30 candidates cheaply and then rerank them down to the best 5. A popular open model for this is ms-marco-MiniLM-L-6-v2. When no model is available, a heuristic reranker can combine stop-word-filtered term overlap, bigram matches and a boost for terms that appear early in the passage.",
    ],
    ["h2", "Improving the Question"],
    [
      "p",
      "Query transformation improves the question before searching. Query rewriting asks the language model to turn a chatty question into a keyword-rich search query. HyDE, short for Hypothetical Document Embeddings, asks the model to write a hypothetical answer and embeds that answer instead of the question, because an answer looks more like a document than a question does. Multi-query expansion generates several paraphrases of the question, retrieves for each one, and fuses the result lists with RRF.",
    ],
    [
      "p",
      "Conversation-aware RAG keeps the last three exchanges and condenses a follow-up such as what about its cost into a standalone question before retrieval.",
    ],
  ],
  [
    ["h1", "Generation and Evaluation"],
    ["h2", "Grounded Generation"],
    [
      "p",
      "The generation step builds a grounded prompt. Each retrieved passage is labelled as Source 1, Source 2 and so on, the model is told to use only the supplied context, and it is given an explicit fallback sentence to use when the context does not contain the answer. Citations such as [1] let the reader verify every claim against the original page.",
    ],
    [
      "p",
      "Contextual compression removes the sentences of each retrieved passage that are irrelevant to the question. This reduces prompt tokens and avoids the lost in the middle effect, where models overlook facts buried in long contexts.",
    ],
    ["h2", "Measuring Quality"],
    [
      "p",
      "Evaluation measures retrieval and generation separately. Recall at k asks whether a relevant passage appears in the top k results. Precision at k is the fraction of the top k that are relevant, and mean reciprocal rank rewards placing the first relevant passage high. For generation, faithfulness checks that every claim in the answer is supported by the retrieved context, and answer relevance checks that the answer addresses the question. A simple faithfulness check counts a claim as supported when at least 50 percent of its content words appear in the context.",
    ],
    [
      "p",
      "Common failure modes include chunks that are too large, a missing keyword retriever, no reranking step, and prompts that do not forbid outside knowledge. Measuring recall before and after each change is the only reliable way to know that an improvement is real.",
    ],
  ],
];
