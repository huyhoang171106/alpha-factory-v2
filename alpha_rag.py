import os
import json
import random
import logging
import requests
import re
from typing import List
from alpha_candidate import AlphaCandidate
from validator import normalize_expression_aliases

logger = logging.getLogger(__name__)

# Try importing dotenv for external configs
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

class RAGMutator:
    """
    RAG-powered Mutator Engine tailored for Elite F1 Alpha Pipeline.
    Loads highly successful alphas (DNA) and injects them into LLM prompt
    context to constrain and guide generation.
    """
    
    def __init__(self, seed_file: str = "data/elite_alphas/elite_seed.json", model_name="stepfun/step-3.5-flash:free"):
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        self.model_name = os.getenv("LLM_MODEL", model_name)
        self.api_key = os.getenv("OPENROUTER_API_KEY", "")
        self.last_attempted = 0
        self.last_succeeded = 0
        self.last_error_rate = 0.0
        if not self.api_key:
            logger.warning("⚠️ OPENROUTER_API_KEY missing. RAG generation is disabled.")
        
        # Load Elite Seeds
        self.elite_seeds = []
        seed_path = os.path.join(os.path.dirname(__file__), '..', seed_file)
        if os.path.exists(seed_path):
            try:
                with open(seed_path, 'r') as f:
                    self.elite_seeds = json.load(f)
                logger.info(f"🧬 Loaded {len(self.elite_seeds)} Elite DNA seeds from {seed_path}")
            except Exception as e:
                logger.error(f"❌ Failed to load Elite Seeds: {e}")
        else:
            logger.warning(f"⚠️ Seed file not found at {seed_path}. Will run without RAG context.")
            
    def _pick_context_dna(self, n=3):
        """Randomly pick top performing DNA strings."""
        if not self.elite_seeds:
            return []
        
        # Filter top seeds
        high_performers = [s['expression'] for s in self.elite_seeds if s.get('sharpe', 0) >= 1.25]
        if not high_performers:
            high_performers = [s['expression'] for s in self.elite_seeds]
            
        return random.sample(high_performers, min(n, len(high_performers)))

    def _build_prompt(self, dna_context: List[str]) -> str:
        prompt = "You are a Quantitative Researcher expert in WorldQuant Brain.\n"
        prompt += "Your task is to generate novel, mathematically sound alpha formulas using the WorldQuant Alpha platform syntax.\n\n"
        
        if dna_context:
            prompt += "Here are some highly successful 'Elite' alpha expressions that passed simulation:\n"
            for i, dna in enumerate(dna_context, 1):
                prompt += f"Elite Alpha {i}: {dna}\n"
            prompt += "\nINSTRUCTIONS:\n"
            prompt += "1. Study the structure, operator groupings, and logic of the Elite alphas above.\n"
            prompt += "2. Create a completely newly mixed alpha (F1 Generation). Do NOT strictly copy them.\n"
            prompt += "3. CRITICAL RESTRICTION: AVOID using 'rank()' as the primary outer wrapper. The platform is oversaturated with simple rank algorithms.\n"
            prompt += "4. Instead, build non-rank families! Use advanced operators like 'group_neutralize(', 'zscore(', 'ts_zscore(', 'group_zscore(', 'scale(', or continuous mathematical ratios (e.g., A/B or A-B).\n"
            prompt += "   Example of good structure: 'group_neutralize(ts_mean(returns, 20)/ts_std_dev(returns, 20), sector)'\n"
        else:
            prompt += "INSTRUCTIONS: Generate a robust short-term reversal or momentum alpha expression. DO NOT rely heavily on basic 'rank(...)'. Explore 'group_neutralize', 'zscore', or continuous mathematical expressions.\n"
            
        prompt += "\nFormat your response by ONLY providing the final mathematical expression. NO markdown, NO text, NO explanation. Just the raw string."
        return prompt

    def _clean_response(self, text: str) -> str:
        """Strip markdown and text around the formula."""
        lines = text.strip().split('\n')
        for line in reversed(lines):
            line = line.strip().strip('`').strip().strip('"').strip("'")
            if any(op in line for op in ['rank', 'volume', 'close', 'open', 'delay', 'returns', 'ts_']):
                return line
        return lines[-1].strip('`').strip()

    def generate_f1_alphas(self, batch_size=3) -> List[AlphaCandidate]:
        """
        Generate multiple F1 Elite Candidates using RAG LLM asynchronously.
        """
        import concurrent.futures
        
        candidates = []
        if not self.api_key:
            self.last_attempted = batch_size
            self.last_succeeded = 0
            self.last_error_rate = 1.0
            return candidates
        logger.info(f"🧠 Prompting OpenRouter for {batch_size} Elite AI Alphas in parallel...")
        
        # Optimize with Connection Pooling for parallel requests
        session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(pool_connections=32, pool_maxsize=32)
        session.mount('https://', adapter)
        session.mount('http://', adapter)
        session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "https://worldquant-alpha-factory.local", 
            "Content-Type": "application/json"
        })
        
        def _fetch_one():
            dna_context = self._pick_context_dna(n=3)
            prompt = self._build_prompt(dna_context)

            model_candidates = [self.model_name, "openrouter/auto"]
            for model_name in model_candidates:
                payload = {
                    "model": model_name,
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.8,
                    "top_p": 0.95
                }
                try:
                    response = session.post(self.api_url, json=payload, timeout=60)
                    if response.status_code == 200:
                        raw_text = response.json().get('choices', [{}])[0].get('message', {}).get('content', '')
                        clean_expr = normalize_expression_aliases(self._clean_response(raw_text))
                        if clean_expr and len(clean_expr) > 10:
                            logger.info(f"💡 AI Generated formula: {clean_expr}")
                            return AlphaCandidate(
                                expression=clean_expr,
                                theme="rag_llm",
                                family="elite_f1",
                                mutation_type="llm_rag_injection"
                            )
                    elif response.status_code in (400, 404) and model_name != "openrouter/auto":
                        logger.warning(
                            f"Model '{model_name}' unavailable ({response.status_code}), "
                            "retrying with openrouter/auto"
                        )
                        continue
                    else:
                        logger.error(f"OpenRouter API Error: {response.status_code} - {response.text}")
                        break
                except requests.exceptions.RequestException as e:
                    logger.error(f"Failed to connect to LLM: {e}")
                    break
                except Exception as ex:
                    logger.error(f"Unexpected error in _fetch_one: {ex}")
                    break
            return None

        # Execute all HTTP requests concurrently
        attempted = batch_size
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, batch_size)) as executor:
            for result in executor.map(lambda _: _fetch_one(), range(batch_size)):
                if result:
                    candidates.append(result)
        self.last_attempted = attempted
        self.last_succeeded = len(candidates)
        self.last_error_rate = 1.0 - (len(candidates) / attempted if attempted else 0.0)
        return candidates

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    rag = RAGMutator()
    cands = rag.generate_f1_alphas(batch_size=2)
    for c in cands:
        print(c.expression, c.family, c.theme)
