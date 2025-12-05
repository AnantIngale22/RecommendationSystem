import faiss
import numpy as np
import pickle
import os

class VectorSearchService:
    def __init__(self, dimension=64):
        self.dimension = dimension
        self.index = None
        self.user_ids = [] # Map index ID to User ID
        
    def build_index(self, embeddings, user_ids):
        """
        Build FAISS index from embeddings.
        embeddings: numpy array of shape (n_users, dimension)
        user_ids: list of user_ids corresponding to embeddings
        """
        print(f"🏗️ Building FAISS index for {len(embeddings)} vectors...")
        
        # Normalize vectors for Cosine Similarity (Inner Product on normalized vectors)
        # faiss.normalize_L2(embeddings) -> Causing Segfault on some macOS envs
        # Use numpy instead
        norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / (norm + 1e-10)
        
        # Create Inner Product Index
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings)
        
        self.user_ids = user_ids
        print(f"✅ Index built with {self.index.ntotal} vectors.")
        
    def search(self, query_vector, k=10):
        """
        Search for k nearest neighbors.
        query_vector: numpy array of shape (1, dimension)
        """
        if not self.index:
            raise ValueError("Index not built!")
            
        # Normalize query
        # faiss.normalize_L2(query_vector)
        norm = np.linalg.norm(query_vector, axis=1, keepdims=True)
        query_vector = query_vector / (norm + 1e-10)
        
        # Search
        distances, indices = self.index.search(query_vector, k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1:
                results.append({
                    'user_id': self.user_ids[idx],
                    'score': float(distances[0][i])
                })
                
        return results
        
    def save_index(self, index_path="user_vector_index.faiss", meta_path="user_index_meta.pkl"):
        """Save index and metadata to disk"""
        if not self.index:
            print("⚠️ No index to save.")
            return
            
        faiss.write_index(self.index, index_path)
        with open(meta_path, 'wb') as f:
            pickle.dump(self.user_ids, f)
        print(f"💾 Index saved to {index_path}")
        
    def load_index(self, index_path="user_vector_index.faiss", meta_path="user_index_meta.pkl"):
        """Load index and metadata from disk"""
        if not os.path.exists(index_path) or not os.path.exists(meta_path):
            print("⚠️ Index files not found.")
            return False
            
        self.index = faiss.read_index(index_path)
        with open(meta_path, 'rb') as f:
            self.user_ids = pickle.load(f)
        print(f"📂 Index loaded with {self.index.ntotal} vectors.")
        return True
