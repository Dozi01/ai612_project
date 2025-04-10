import json
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from tqdm import tqdm

class Retriever:
    def __init__(self, data_path, label_path, model_name='emilyalsentzer/Bio_ClinicalBERT', top_k=5):
        """Initialize the Retriever with data paths and model."""   
        self.data_path = data_path
        self.label_path = label_path
        self.model_name = model_name
        self.top_k = top_k
        
        # Load model
        print("Loading Bio-ClinicalBERT model...")
        self.model = SentenceTransformer(model_name)
        
        # Load data
        self._load_data()
        
        # Build index
        self._build_index()
        
    def _load_data(self):
        """Load and prepare the training data and labels."""
        print("Loading data...")
        with open(self.data_path, 'r', encoding='utf-8') as f:
            self.train_data = json.load(f)['data']  # Question data
            
        with open(self.label_path, 'r', encoding='utf-8') as f:
            self.label_data = json.load(f)  # label id -> SQL query
            
        # Prepare label data
        self.label_texts = list(self.label_data.values())
        self.label_ids = list(self.label_data.keys())
        
    def _build_index(self):
        """Build the FAISS index for similarity search."""
        print("Embedding label data...")
        label_embeddings = self.model.encode(
            self.label_texts, 
            convert_to_numpy=True, 
            show_progress_bar=True
        )
        
        # L2 normalize for cosine similarity
        faiss.normalize_L2(label_embeddings)
        
        print("Building FAISS index...")
        self.index = faiss.IndexFlatIP(label_embeddings.shape[1])
        self.index.add(label_embeddings)
        
        print(f"Number of vectors in the index: {self.index.ntotal}")
        
    def retrieve(self, query):
        """Retrieve top-k similar SQL queries for a given question."""
        # Encode query
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)
        
        # Search
        D, I = self.index.search(query_embedding, self.top_k)
        
        # Format results
        results = []
        for score, idx in zip(D[0], I[0]):
            label_id = self.label_ids[idx]
            sql_text = self.label_data[label_id]
            # Find the corresponding question from train_data
            question = next((item['question'] for item in self.train_data if item['id'] == label_id), None)
            results.append({
                'label_id': label_id,
                'score': float(score),
                'sql': sql_text,
                'question': question
            })
            
        return results

# Example usage
if __name__ == "__main__":
    # File paths
    DATA_PATH = 'data/augmented/train_data.json'
    LABEL_PATH = 'data/augmented/train_label.json'
    
    # Initialize retriever
    retriever = Retriever(DATA_PATH, LABEL_PATH)
    
    # Test query
    test_query = "What are the precautions after the spinal canal explor nec procedure?"
    print(f"\nTest query: {test_query}")
    
    # Get results
    results = retriever.retrieve(test_query)
    
    # Print results
    print("\nTop-k Results:")
    for rank, result in enumerate(results, 1):
        print(f"{rank}. Label ID: {result['label_id']} | Score: {result['score']:.4f}")
        print(f"   Question: {result['question']}")
        print(f"   SQL: {result['sql']}\n")
        
    print("Done ✅")
