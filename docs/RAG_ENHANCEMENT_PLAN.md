# RAG Enhancement Plan - Alpha Factory

## Current State Analysis

### What's Already Implemented
- Elite seed loading from JSON (`elite_seed.json`)
- Random DNA sampling for context
- Negative example retrieval from SQLite
- DNA hints from `generator_weights.json`
- Research themes from `research_papers.json`
- LLM prompt generation with context injection
- Local critic loop for quality validation

### What's Missing (True RAG Capabilities)
1. **No vector embeddings** - No semantic similarity search
2. **No BM25/sparse retrieval** - Simple random selection
3. **No AST-based structural similarity** - Can't detect functional duplicates
4. **No reranking** - Raw LLM generation without precision enhancement
5. **No hierarchical indexing** - Flat JSON, no category separation

---

## Implementation Plan

### Phase 1: Core Similarity Functions (Priority: HIGH)

#### 1.1 AST-Based Structural Similarity
```python
# Add to alpha_rag.py
def _compute_ast_similarity(expr1: str, expr2: str) -> float:
    # Uses parameter_agnostic_signature for exact match
    # Falls back to Jaccard token similarity
    # Returns 0-1 scale
```

#### 1.2 BM25-Style Keyword Scoring
```python
# Add to alpha_rag.py
def _bm25_score(query: str, candidate: str, avg_doc_len: float = 50.0) -> float:
    # Simplified BM25 without inverted index
    # Uses token overlap with TF normalization
```

### Phase 2: Enhanced Retrieval (Priority: HIGH)

#### 2.1 Diverse DNA Picking
```python
def _pick_diverse_dna(self, n: int = 3, min_similarity: float = 0.3) -> List[str]:
    # Greedy selection with minimum AST similarity threshold
    # Ensures selected seeds have different structural patterns
```

#### 2.2 Enhanced Negative Examples
- Expand default `n` from 3 to 10
- Add diversity filtering (same AST similarity logic)
- Query more failed alphas, select structurally diverse subset

#### 2.3 Hybrid Retrieval Wrapper
```python
def _hybrid_retrieve(self, n: int = 3, use_diverse: bool = True) -> tuple:
    # Combines _pick_diverse_dna + _fetch_negative_examples
    # Use this instead of separate calls in generate_f1_alphas
```

### Phase 3: Query Expansion (Priority: MEDIUM)

#### 3.1 Multi-Query Expansion (HyDE-style)
```python
def _expand_queries(self, base_query: str) -> List[str]:
    # Generate alternative phrasings
    # Add research themes as pseudo-documents
    # Max 5 queries per expansion
```

### Phase 4: Integration (Priority: HIGH)

#### 4.1 Update `generate_f1_alphas`
- Replace `_pick_context_dna()` with `_hybrid_retrieve()`
- Pass diverse negative examples to prompt

#### 4.2 Environment Controls
```env
RAG_USE_DIVERSE=true      # Enable diverse DNA picking
RAG_NEGATIVE_EXAMPLES=10  # Number of failed examples
RAG_MIN_SIMILARITY=0.3    # Diversity threshold
```

---

## Files to Modify

| File | Changes |
|------|---------|
| `alpha_rag.py` | Add similarity functions, update retrieval, integrate hybrid |
| `.env.example` | Add new environment variables |

---

## Testing

1. Run existing tests: `pytest tests/test_alpha_rag.py -v`
2. Verify AST similarity returns 1.0 for identical structures
3. Verify BM25 score increases with more token overlap
4. Verify diverse DNA picking selects structurally different seeds

---

## Future Enhancements (Not in Scope)

- Vector embeddings with sentence-transformers
- FAISS/Chroma vector store
- Cross-encoder reranking
- Domain-specific embeddings (FinBERT, BAM)
- Hierarchical category indexing

---

## Estimated Effort

| Phase | Complexity | Time |
|-------|------------|------|
| Phase 1 | Low | 1-2 hours |
| Phase 2 | Medium | 2-3 hours |
| Phase 3 | Medium | 1-2 hours |
| Phase 4 | Low | 1 hour |
| **Total** | - | **5-8 hours** |

---

## Recommendation

**Start with Phase 1 + Phase 2** - these give 80% of RAG improvement with 20% effort:
- AST similarity for immediate deduplication improvement
- BM25 for keyword overlap scoring  
- Diverse DNA picking for better context diversity
- Enhanced negative examples with diversity filtering
- Hybrid retrieval integration

This approach avoids:
- Adding heavy dependencies (sentence-transformers, FAISS)
- Complex vector store infrastructure
- Fine-tuning domain embeddings