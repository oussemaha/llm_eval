import os
import json
import pickle
import csv
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
from dotenv import load_dotenv
import faiss
from sentence_transformers import SentenceTransformer
from src.controller.tools.Tool import Tool
from pydantic import BaseModel


@dataclass
class Document:
    """Represents a document stored in the retriever."""
    id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class RetrievalResult:
    """Represents a single retrieval result."""
    document: Document
    score: float
    rank: int

class RetrieverInput(BaseModel):
        query: str
        top_k: int = 5,
        score_threshold: Optional[float] = None
        filter_fn: Optional[Any] = None

load_dotenv()
class FAISSRetriever_Tool(Tool):
    """
    A FAISS-based document retriever for RAG pipelines.

    Supports adding, deleting, retrieving, and persisting documents
    with dense vector similarity search.

    Args:
        embedding_model: SentenceTransformer model name or instance.
        index_type: FAISS index type — "flat" (exact) or "ivf" (approximate).
        embedding_dim: Embedding dimension. Auto-detected if None.
        nlist: Number of IVF clusters (only used when index_type="ivf").
        persist_dir: Directory for saving/loading the index and metadata.
    """

    def __init__(
        self,
        index_type: str = "flat",
        embedding_dim: Optional[int] = None,
        nlist: int = 100,
    ):
        # Load or reuse embedding model
        embedding_model=os.getenv("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")

        if isinstance(embedding_model, str):
            self.model = SentenceTransformer(embedding_model)
        else:
            self.model = embedding_model

        self.index_type = index_type
        self.nlist = nlist

        # Resolve embedding dimension
        if embedding_dim is None:
            probe = self.model.encode(["probe"])
            self.embedding_dim = probe.shape[1]
        else:
            self.embedding_dim = embedding_dim

        # Internal state
        self._documents: Dict[str, Document] = {}   # doc_id → Document
        self._index_ids: List[str] = []              # positional id list (FAISS index ↔ doc_id)
        self._deleted_positions: set = set()         # soft-deleted positions

        # Build FAISS index
        self._index = self._build_index()

        # Load persisted state if available
        persist_dir = os.getenv("faiss_persist_dir")
        if persist_dir and os.path.exists(persist_dir):
            self._load(persist_dir)
        super().__init__(
            name="search_medical_research", # Underscores are often better for internal naming
            description="Search the local knowledge base for peer-reviewed medical research, clinical studies, and evidence-based treatment protocols.",
            schema=RetrieverInput,
            func=self.retrieve
        )
        

    # ─────────────────────────────────────────────
    # Index construction
    # ─────────────────────────────────────────────

    def _build_index(self) -> faiss.Index:
        if self.index_type == "flat":
            return faiss.IndexFlatIP(self.embedding_dim)   # inner-product (cosine after normalizing)
        elif self.index_type == "ivf":
            quantizer = faiss.IndexFlatIP(self.embedding_dim)
            index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, self.nlist, faiss.METRIC_INNER_PRODUCT)
            return index
        else:
            raise ValueError(f"Unknown index_type '{self.index_type}'. Choose 'flat' or 'ivf'.")

    # ─────────────────────────────────────────────
    # Embedding helpers
    # ─────────────────────────────────────────────

    def _embed(self, texts: List[str]) -> np.ndarray:
        """Embed and L2-normalize a list of texts (enables cosine similarity via IP)."""
        vectors = self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        vectors = vectors.astype(np.float32)
        faiss.normalize_L2(vectors)
        return vectors

    # ─────────────────────────────────────────────
    # Document management
    # ─────────────────────────────────────────────

    def add_documents(
        self,
        documents: List[Document] | List[Dict],
        batch_size: int = 64,
    ) -> List[str]:
        """
        Add one or more documents to the retriever.

        Args:
            documents: List of Document objects or dicts with 'id', 'content', and
                       optional 'metadata' keys.
            batch_size: Embedding batch size.

        Returns:
            List of document IDs that were added.
        """
        # Normalize to Document objects
        docs: List[Document] = []
        for d in documents:
            if isinstance(d, dict):
                docs.append(Document(**d))
            else:
                docs.append(d)

        # Deduplicate against existing ids
        new_docs = [d for d in docs if d.id not in self._documents]
        if not new_docs:
            return []

        # Train IVF index on first batch if needed
        if self.index_type == "ivf" and not self._index.is_trained:
            train_texts = [d.content for d in new_docs]
            train_vectors = self._embed(train_texts)
            self._index.train(train_vectors)

        # Embed in batches and add to FAISS
        added_ids = []
        for i in range(0, len(new_docs), batch_size):
            batch = new_docs[i : i + batch_size]
            texts = [d.content for d in batch]
            vectors = self._embed(texts)
            self._index.add(vectors)
            for doc in batch:
                self._documents[doc.id] = doc
                self._index_ids.append(doc.id)
                added_ids.append(doc.id)
            print(f"Added {len(batch)} documents (total: {len(self._documents)})")

        return added_ids

    def add_document(self, document: Document | Dict) -> str:
        """Convenience wrapper for adding a single document."""
        ids = self.add_documents([document])
        return ids[0] if ids else ""

    def delete_documents(self, doc_ids: List[str]) -> List[str]:
        """
        Soft-delete documents by ID.

        FAISS does not support in-place removal, so deleted documents are marked
        and excluded from retrieval results. Call `rebuild_index()` to reclaim memory.

        Returns:
            List of IDs that were successfully deleted.
        """
        removed = []
        for doc_id in doc_ids:
            if doc_id in self._documents:
                position = self._index_ids.index(doc_id)
                self._deleted_positions.add(position)
                del self._documents[doc_id]
                removed.append(doc_id)
        return removed

    def delete_document(self, doc_id: str) -> bool:
        """Convenience wrapper for deleting a single document."""
        return len(self.delete_documents([doc_id])) > 0

    def update_document(self, doc_id: str, new_content: str, new_metadata: Optional[Dict] = None) -> bool:
        """
        Update a document's content (and optionally its metadata).

        Internally deletes the old entry and re-adds it with the same ID.
        """
        if doc_id not in self._documents:
            return False
        old_doc = self._documents[doc_id]
        self.delete_document(doc_id)
        updated = Document(
            id=doc_id,
            content=new_content,
            metadata=new_metadata if new_metadata is not None else old_doc.metadata,
        )
        self.add_document(updated)
        return True

    def get_document(self, doc_id: str) -> Optional[Document]:
        """Retrieve a document by its ID."""
        return self._documents.get(doc_id)

    def list_documents(self) -> List[Document]:
        """Return all (non-deleted) documents."""
        return list(self._documents.values())

    # ─────────────────────────────────────────────
    # Retrieval
    # ─────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: Optional[float] = None,
        filter_fn: Optional[Any] = None,
    ) -> List[RetrievalResult]:
        """
        Retrieve the most relevant documents for a query.

        Args:
            query: Natural-language query string.
            top_k: Maximum number of results to return.
            score_threshold: If set, only return results with score >= threshold.
            filter_fn: Optional callable (Document) → bool for metadata filtering.

        Returns:
            List of RetrievalResult sorted by descending similarity score.
        """
        if self._index.ntotal == 0:
            return []
        print(f"Retrieving for query: '{query}' (top_k={top_k}, score_threshold={score_threshold})")
        # Over-fetch to account for deleted positions + filtering
        fetch_k = min(top_k * 4 + len(self._deleted_positions), self._index.ntotal)

        query_vector = self._embed([query])
        scores, indices = self._index.search(query_vector, fetch_k)

        results: List[RetrievalResult] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            if idx in self._deleted_positions:
                continue

            doc_id = self._index_ids[idx]
            doc = self._documents.get(doc_id)
            if doc is None:
                continue

            if score_threshold is not None and score < score_threshold:
                continue

            if filter_fn is not None and not filter_fn(doc):
                continue

            results.append(RetrievalResult(document=doc, score=float(score), rank=0))

            if len(results) >= top_k:
                break

        # Assign ranks
        for i, r in enumerate(results):
            r.rank = i + 1

        return results

    def retrieve_batch(
        self,
        queries: List[str],
        top_k: int = 5,
    ) -> List[List[RetrievalResult]]:
        """Retrieve results for multiple queries in a single batched call."""
        return [self.retrieve(q, top_k=top_k) for q in queries]

    # ─────────────────────────────────────────────
    # Index maintenance
    # ─────────────────────────────────────────────

    def rebuild_index(self) -> None:
        """
        Hard-rebuild the FAISS index from scratch, permanently removing
        soft-deleted entries and reclaiming memory.
        """
        live_docs = list(self._documents.values())
        self._index = self._build_index()
        self._index_ids = []
        self._deleted_positions = set()
        if live_docs:
            self.add_documents(live_docs)

    def clear(self) -> None:
        """Remove all documents and reset the index."""
        self._documents = {}
        self._index_ids = []
        self._deleted_positions = set()
        self._index = self._build_index()

    # ─────────────────────────────────────────────
    # Persistence
    # ─────────────────────────────────────────────

    def save(self, directory: Optional[str] = None) -> str:
        """
        Persist the FAISS index and document store to disk.

        Returns:
            The directory the state was saved to.
        """
        save_dir = directory or self.persist_dir
        if save_dir is None:
            raise ValueError("No persist_dir specified.")

        os.makedirs(save_dir, exist_ok=True)

        faiss.write_index(self._index, os.path.join(save_dir, "index.faiss"))

        meta = {
            "index_ids": self._index_ids,
            "deleted_positions": list(self._deleted_positions),
            "documents": {k: asdict(v) for k, v in self._documents.items()},
        }
        with open(os.path.join(save_dir, "metadata.json"), "w") as f:
            json.dump(meta, f, indent=2)

        return save_dir

    def _load(self, directory: str) -> None:
        """Load persisted state from disk."""
        index_path = os.path.join(directory, "index.faiss")
        meta_path = os.path.join(directory, "metadata.json")

        if os.path.exists(index_path):
            self._index = faiss.read_index(index_path)

        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                meta = json.load(f)
            self._index_ids = meta.get("index_ids", [])
            self._deleted_positions = set(meta.get("deleted_positions", []))
            self._documents = {
                k: Document(**v) for k, v in meta.get("documents", {}).items()
            }

    # ─────────────────────────────────────────────
    # Stats & utilities
    # ─────────────────────────────────────────────

    @property
    def stats(self) -> Dict[str, Any]:
        return {
            "total_vectors": self._index.ntotal,
            "live_documents": len(self._documents),
            "deleted_positions": len(self._deleted_positions),
            "index_type": self.index_type,
            "embedding_dim": self.embedding_dim,
        }

    def __len__(self) -> int:
        return len(self._documents)

    def __repr__(self) -> str:
        return (
            f"FAISSRetriever("
            f"docs={len(self)}, "
            f"index_type={self.index_type}, "
            f"dim={self.embedding_dim})"
        )


if __name__ == "__main__":
    import sys
    
    # CSV file path - update this to your CSV file location
    csv_file_path = "/home/oussema/Downloads/abstracts.csv"  # Should have 'doi' and 'abstract' columns
    
    if len(sys.argv) > 1:
        csv_file_path = sys.argv[1]
    
    if not os.path.exists(csv_file_path):
        print(f"Error: CSV file not found at {csv_file_path}")
        print("Usage: python retriever.py <path_to_csv_file>")
        sys.exit(1)
    
    # Initialize the retriever with persistence
    retriever = FAISSRetriever(
        embedding_model="neuml/pubmedbert-base-embeddings",
        index_type="flat",
        persist_dir="./retriever_data"
    )
    
    # Load documents from CSV file
    print(f"Loading documents from {csv_file_path}...")
    documents = []
    
    try:
        with open(csv_file_path, 'r', encoding='utf-8') as f:
            csv_reader = csv.DictReader(f)
            
            for row_num, row in enumerate(csv_reader, start=2):  # start=2 to account for header
                try:
                    doi = row.get('doi', '').strip()
                    abstract = row.get('abstract', '').strip()
                    
                    # Skip rows with missing doi or abstract
                    if not doi or not abstract:
                        continue
                    
                    doc = {
                        "id": doi,
                        "content": abstract,
                        "metadata": {"source": "csv", "doi": doi}
                    }
                    documents.append(doc)
                    
                    # Print progress every 1000 rows
                    if len(documents) % 1000 == 0:
                        print(f"  Processed {len(documents)} documents...")
                
                except Exception as e:
                    print(f"Warning: Error processing row {row_num}: {e}")
                    continue
        
        if not documents:
            print("No valid documents found in CSV file!")
            sys.exit(1)
        
        print(f"\nTotal documents loaded from CSV: {len(documents)}")
        
        # Add documents to the retriever in batches
        print("Adding documents to retriever...")
        added_ids = retriever.add_documents(documents, batch_size=64)
        print(f"Successfully added {len(added_ids)} documents to retriever")
        
        # Display retriever statistics
        print(f"\nRetriever stats: {retriever.stats}")
        
        # Example: Retrieve documents based on a query
        query = "machine learning"
        print(f"\nExample retrieval for query: '{query}'")
        results = retriever.retrieve(query, top_k=5)
        
        if results:
            for result in results:
                print(f"\nRank {result.rank} (Score: {result.score:.4f})")
                print(f"DOI: {result.document.id}")
                print(f"Abstract: {result.document.content[:200]}...")
        else:
            print("No results found")
        
        # Save the retriever state
        print(f"\nSaving retriever state...")
        save_path = retriever.save()
        print(f"Retriever saved to: {save_path}")
        
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        sys.exit(1)