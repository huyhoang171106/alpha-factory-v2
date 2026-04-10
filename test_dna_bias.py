import sys
import os

from generator import AlphaGenerator
from alpha_dna import DNAWeights

def test_bias():
    # Mock some weights
    w = DNAWeights.default()
    # Bias heavily towards vwap and ts_corr
    w.field_weights["vwap"] = 1.0
    w.field_weights["close"] = 0.01
    w.operator_weights["ts_corr"] = 1.0
    w.operator_weights["ts_delta"] = 0.01
    
    gen = AlphaGenerator(dna_weights=w)
    print("Testing DNA Biased Generation...")
    batch = gen.generate_batch(50)
    
    vwap_count = sum(1 for e in batch if "vwap" in e)
    ts_corr_count = sum(1 for e in batch if "ts_corr" in e)
    
    with open("test_results.txt", "w") as f:
        f.write(f"Batch size: {len(batch)}\n")
        f.write(f"Expressions with 'vwap': {vwap_count}\n")
        f.write(f"Expressions with 'ts_corr': {ts_corr_count}\n")
        f.write("\nSample biased alphas:\n")
        for e in batch[:10]:
            f.write(f"  {e}\n")
    print("Test results saved to test_results.txt")

if __name__ == "__main__":
    test_bias()
