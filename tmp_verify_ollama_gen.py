
import logging
import os
from alpha_rag import RAGMutator

logging.basicConfig(level=logging.INFO)

def test_generation():
    rag = RAGMutator()
    print(f"Testing generation with {rag.ollama_model} on {rag.ollama_base}...")
    try:
        candidates = rag.generate_f1_alphas(batch_size=1)
        if candidates:
            print(f"✅ Successfully generated {len(candidates)} candidates.")
            for c in candidates:
                print(f"Expression: {c.expression}")
                print(f"Hypothesis: {c.hypothesis}")
        else:
            print("❌ No candidates generated.")
    except Exception as e:
        print(f"❌ Error during generation: {e}")

if __name__ == "__main__":
    test_generation()
