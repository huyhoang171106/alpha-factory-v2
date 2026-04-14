
import os
import logging
from alpha_rag import RAGMutator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_config():
    rag = RAGMutator()
    print(f"Ollama Base: {rag.ollama_base}")
    print(f"Ollama Model: {rag.ollama_model}")
    print(f"Ollama API Key: {'Set' if rag.ollama_api_key else 'Not Set'}")
    
    if rag.ollama_api_key:
        print(f"API Key (first 5): {rag.ollama_api_key[:5]}...")

if __name__ == "__main__":
    test_config()
