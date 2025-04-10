import json
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from tqdm import tqdm
from rich import print


class Retriever:
    def __init__(self, train_data_path, train_label_path, valid_data_path, valid_label_path, model_name='emilyalsentzer/Bio_ClinicalBERT', top_k=10, hybrid_weight=0.5, use_null=False):
        """Initialize the Retriever with data paths and model."""
        self.train_data_path = train_data_path
        self.train_label_path = train_label_path
        self.valid_data_path = valid_data_path
        self.valid_label_path = valid_label_path
        self.model_name = model_name
        self.top_k = top_k
        self.hybrid_weight = hybrid_weight
        self.use_null = use_null

        # Load model
        print("Loading Bio-ClinicalBERT model...")
        self.model = SentenceTransformer(model_name)

        # Load data
        self._load_data()

        # Build or load indexes
        self._build_or_load_index()
        self._build_or_load_bm25()

    def _load_data(self):
        """Load and prepare the training data and labels."""
        print("Loading data...")
        with open(self.train_data_path, 'r', encoding='utf-8') as f:
            self.train_data = json.load(f)['data']  # Question data

        with open(self.train_label_path, 'r', encoding='utf-8') as f:
            self.train_label_data = json.load(f)  # label id -> SQL query

        with open(self.valid_data_path, 'r', encoding='utf-8') as f:
            self.valid_data = json.load(f)['data']  # Question data

        with open(self.valid_label_path, 'r', encoding='utf-8') as f:
            self.valid_label_data = json.load(f)  # label id -> SQL query

        # Filter out data with NULL labels if use_null is False
        if not self.use_null:
            self.train_data = [item for item in self.train_data if self.train_label_data[item['id']] != 'null']
            self.valid_data = [item for item in self.valid_data if self.valid_label_data[item['id']] != 'null']
            print(f"Filtered {len(self.train_data)} training data and {len(self.valid_data)} validation data")

        # Prepare question data
        self.questions = [item['question'] for item in self.train_data]
        self.question_ids = [item['id'] for item in self.train_data]

        # Prepare corpus for BM25
        self.corpus = [q.lower() for q in self.questions]

    def _build_or_load_index(self):
        """Build or load the FAISS index for similarity search."""
        import os

        index_filename = "faiss_index_with_null.bin" if self.use_null else "faiss_index_filtered.bin"

        if os.path.exists(index_filename):
            print(f"Loading existing FAISS index from {index_filename}...")
            self.index = faiss.read_index(index_filename)
            print(f"Loaded index with {self.index.ntotal} vectors")
        else:
            print("Building new FAISS index...")
            print("Embedding questions...")
            query_embeddings = self.model.encode(
                self.questions,
                convert_to_numpy=True,
                show_progress_bar=True
            )

            # L2 normalize for cosine similarity
            faiss.normalize_L2(query_embeddings)

            print("Building FAISS index...")
            self.index = faiss.IndexFlatIP(query_embeddings.shape[1])
            self.index.add(query_embeddings)

            print(f"Number of vectors in the index: {self.index.ntotal}")

            # Save FAISS index
            print(f"Saving FAISS index to {index_filename}...")
            faiss.write_index(self.index, index_filename)

    def _build_or_load_bm25(self):
        """Build or load BM25 index."""
        from rank_bm25 import BM25Okapi
        import pickle
        import os

        bm25_filename = "bm25_index_with_null.pkl" if self.use_null else "bm25_index_filtered.pkl"

        if os.path.exists(bm25_filename):
            print(f"Loading existing BM25 index from {bm25_filename}...")
            with open(bm25_filename, "rb") as f:
                self.bm25 = pickle.load(f)
        else:
            print("Building new BM25 index...")
            # Tokenize corpus
            tokenized_corpus = [doc.split() for doc in self.corpus]
            self.bm25 = BM25Okapi(tokenized_corpus)

            # Save BM25 index
            print(f"Saving BM25 index to {bm25_filename}...")
            with open(bm25_filename, "wb") as f:
                pickle.dump(self.bm25, f)

    def retrieve(self, query):
        """Retrieve top-k similar questions using hybrid search."""

        # Dense retrieval with FAISS
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)
        dense_scores, dense_indices = self.index.search(query_embedding, self.top_k)
        dense_scores = dense_scores[0]
        dense_indices = dense_indices[0]

        # Format FAISS results
        faiss_results = []
        for idx, score in zip(dense_indices, dense_scores):
            label_id = self.question_ids[idx]
            faiss_results.append({
                'label_id': label_id,
                'score': float(score),
                'sql': self.train_label_data[label_id],
                'question': self.questions[idx]
            })

        # Sparse retrieval with BM25
        tokenized_query = query.lower().split()
        sparse_scores = self.bm25.get_scores(tokenized_query)

        # Get top-k BM25 results
        bm25_top_k = np.argsort(sparse_scores)[-self.top_k:][::-1]
        bm25_results = []
        for idx in bm25_top_k:
            label_id = self.question_ids[idx]
            bm25_results.append({
                'label_id': label_id,
                'score': float(sparse_scores[idx]),
                'sql': self.train_label_data[label_id],
                'question': self.questions[idx]
            })

        # Normalize scores for hybrid search
        dense_scores = (dense_scores - dense_scores.min()) / (dense_scores.max() - dense_scores.min())
        sparse_scores = (sparse_scores - sparse_scores.min()) / (sparse_scores.max() - sparse_scores.min())

        # Combine scores
        final_scores = {}
        for idx, (dense_idx, dense_score) in enumerate(zip(dense_indices, dense_scores)):
            sparse_score = sparse_scores[dense_idx]
            hybrid_score = self.hybrid_weight * dense_score + (1 - self.hybrid_weight) * sparse_score
            final_scores[dense_idx] = hybrid_score

        # Sort by hybrid score
        sorted_indices = sorted(final_scores.keys(), key=lambda x: final_scores[x], reverse=True)[:self.top_k]

        # Format hybrid results
        hybrid_results = []
        for idx in sorted_indices:
            label_id = self.question_ids[idx]
            hybrid_results.append({
                'label_id': label_id,
                'score': float(final_scores[idx]),
                'sql': self.train_label_data[label_id],
                'question': self.questions[idx]
            })

        return faiss_results, bm25_results, hybrid_results

# Example usage
if __name__ == "__main__":
    # File paths
    TRAIN_DATA_PATH = 'data/augmented/train_data.json'
    TRAIN_LABEL_PATH = 'data/augmented/train_label.json'
    VALID_DATA_PATH = 'data/augmented/valid_data.json'
    VALID_LABEL_PATH = 'data/augmented/valid_label.json'

    # Initialize retriever with filtered data (no NULL labels)
    retriever = Retriever(
        TRAIN_DATA_PATH,
        TRAIN_LABEL_PATH,
        VALID_DATA_PATH,
        VALID_LABEL_PATH,
        use_null=True
    )

    # Test query
    test_query = "What are the precautions after the spinal canal explor nec procedure?"
    print(f"\nTest query: {test_query}")

    # Get results
    faiss_results, bm25_results, hybrid_results = retriever.retrieve(test_query)

    # Print FAISS results
    print("\nFAISS Results:")
    for rank, result in enumerate(faiss_results, 1):
        print(f"{rank}. Label ID: {result['label_id']} | Score: {result['score']:.4f}")
        print(f"   Question: {result['question']}")
        print(f"   SQL: {result['sql']}\n")

    # Print BM25 results
    print("\nBM25 Results:")
    for rank, result in enumerate(bm25_results, 1):
        print(f"{rank}. Label ID: {result['label_id']} | Score: {result['score']:.4f}")
        print(f"   Question: {result['question']}")
        print(f"   SQL: {result['sql']}\n")

    # Print Hybrid results
    print("\nHybrid Results:")
    for rank, result in enumerate(hybrid_results, 1):
        print(f"{rank}. Label ID: {result['label_id']} | Score: {result['score']:.4f}")
        print(f"   Question: {result['question']}")
        print(f"   SQL: {result['sql']}\n")

    print("Done ✅")
