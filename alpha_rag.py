import os
import json
import random
import logging
import requests
import re
import time
from typing import List, Tuple

try:
    import anthropic
except ImportError:
    anthropic = None
from typing import List
import hashlib
from alpha_candidate import AlphaCandidate
from alpha_ast import operator_set, token_set, parameter_agnostic_signature, canonicalize_expression
from validator import normalize_expression_aliases, validate_expression
from alpha_ranker import score_expression
from alpha_policy import detect_survivorship_and_lookahead

BM25_K1 = 1.5
BM25_B = 0.75

logger = logging.getLogger(__name__)

# Try importing dotenv for external configs
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


# =============================================================================
# RAG Enhancement: Similarity Functions (Phase 1)
# =============================================================================

def _compute_ast_similarity(expr1: str, expr2: str) -> float:
    """
    Compute AST-based structural similarity between two alpha expressions.
    Uses parameter_agnostic_signature for exact structural match.
    Falls back to Jaccard token similarity for approximate match.
    Returns 0-1 scale.
    """
    sig1 = parameter_agnostic_signature(expr1)
    sig2 = parameter_agnostic_signature(expr2)

    # Exact structural match
    if sig1 == sig2:
        return 1.0

    # Jaccard token similarity (operators only, no params)
    tokens1 = operator_set(expr1)
    tokens2 = operator_set(expr2)

    if not tokens1 or not tokens2:
        return 0.0

    intersection = len(tokens1 & tokens2)
    union = len(tokens1 | tokens2)

    return intersection / union if union > 0 else 0.0


def _bm25_score(query: str, candidate: str, avg_doc_len: float = 50.0) -> float:
    """
    Compute BM25-style keyword overlap score between query and candidate.
    Simplified BM25 without inverted index - uses token overlap with TF normalization.
    """
    query_tokens = token_set(query, strip_numbers=True)
    cand_tokens = token_set(candidate, strip_numbers=True)

    if not query_tokens or not cand_tokens:
        return 0.0

    # Term frequency in candidate
    tf = sum(1 for t in query_tokens if t in cand_tokens)

    # BM25 formula (simplified)
    doc_len = len(cand_tokens)
    tf_component = (tf * (BM25_K1 + 1)) / (tf + BM25_K1 * (1 - BM25_B + BM25_B * doc_len / avg_doc_len))

    # IDF component (simplified - assume uniform)
    idf = 1.0

    return tf_component * idf


# =============================================================================
# RAG Enhancement: Diverse DNA Picking (Phase 2)
# =============================================================================

def _pick_diverse_dna(
    seeds: List[dict],
    n: int = 3,
    min_similarity: float = 0.3
) -> List[str]:
    """
    Greedy selection of structurally diverse DNA strings.
    Ensures selected seeds have different structural patterns (AST similarity < threshold).
    """
    if not seeds:
        return []

    # Filter to high performers
    high_performers = [
        s["expression"] for s in seeds if s.get("sharpe", 0) >= 1.25
    ]
    if not high_performers:
        high_performers = [s["expression"] for s in seeds]

    if len(high_performers) <= n:
        return high_performers

    selected = []
    remaining = high_performers.copy()

    while len(selected) < n and remaining:
        # Pick first remaining as seed
        candidate = remaining.pop(0)
        selected.append(candidate)

        # Remove too-similar candidates
        filtered_remaining = []
        for expr in remaining:
            sim = _compute_ast_similarity(candidate, expr)
            if sim < min_similarity:
                filtered_remaining.append(expr)
        remaining = filtered_remaining

    return selected


# =============================================================================
# RAG Enhancement: Hybrid Retrieval (Phase 2)
# =============================================================================

def _fetch_diverse_negative_examples(
    db_path: str,
    n: int = 10,
    min_similarity: float = 0.3
) -> List[str]:
    """
    Fetch structurally diverse failed alphas from database.
    Expands default n from 3 to 10 with diversity filtering.
    """
    import sqlite3

    recent_fails = []
    if not os.path.exists(db_path):
        return recent_fails

    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            # Get more failed alphas than needed (we'll filter for diversity)
            cursor.execute(
                """
                SELECT expression FROM alphas
                WHERE (theme='rag_llm' OR theme='llm')
                AND error='' AND sharpe < 0.5
                ORDER BY id DESC LIMIT ?
                """,
                (n * 2,),  # Fetch extra for diversity filtering
            )
            all_fails = [r[0] for r in cursor.fetchall()]
    except Exception as e:
        logger.warning(f"Could not load negative examples: {e}")
        return []

    if len(all_fails) <= n:
        return all_fails

    # Apply diversity filtering
    selected = []
    for fail_expr in all_fails:
        # Check similarity to already selected
        too_similar = False
        for selected_expr in selected:
            if _compute_ast_similarity(fail_expr, selected_expr) >= min_similarity:
                too_similar = True
                break

        if not too_similar:
            selected.append(fail_expr)

        if len(selected) >= n:
            break

    return selected


def _hybrid_retrieve(
    self,
    n_dna: int = 3,
    n_neg: int = 10,
    min_similarity: float = 0.3
) -> Tuple[List[str], List[str]]:
    """
    Hybrid retrieval combining diverse DNA picking + diverse negative examples.
    Use this instead of separate _pick_context_dna + _fetch_negative_examples calls.
    """
    # Diverse DNA picking
    dna_context = _pick_diverse_dna(
        self.elite_seeds,
        n=n_dna,
        min_similarity=min_similarity
    )

    # Diverse negative examples
    db_path = os.path.join(os.path.dirname(__file__), "alpha_results.db")
    neg_examples = _fetch_diverse_negative_examples(
        db_path,
        n=n_neg,
        min_similarity=min_similarity
    )

    return dna_context, neg_examples


class RAGMutator:
    """
    RAG-powered Mutator Engine tailored for Elite F1 Alpha Pipeline.
    Loads highly successful alphas (DNA) and injects them into LLM prompt
    context to constrain and guide generation.
    """

    def __init__(
        self,
        seed_file: str = "data/elite_alphas/elite_seed.json",
        model_name="stepfun/step-3.5-flash:free",
    ):
        self.openai_base = os.getenv("OPENAI_API_BASE", "https://openrouter.ai/api/v1")
        self.api_url = f"{self.openai_base.rstrip('/')}/chat/completions"
        self.model_name = os.getenv("LLM_MODEL", model_name)

        # Ollama support
        self.ollama_base = os.getenv("OLLAMA_API_BASE", "")
        self.ollama_model = os.getenv("OLLAMA_MODEL", "llama3")
        self.ollama_api_key = os.getenv("OLLAMA_API_KEY", "")
        self.ollama_client = None

        # API keys
        self.api_key = os.getenv("OPENAI_API_KEY") or os.getenv(
            "OPENROUTER_API_KEY", ""
        )
        self.anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")
        self.anthropic_base_url = os.getenv(
            "ANTHROPIC_BASE_URL", "https://api.anthropic.com"
        )

        # Initialize Anthropic client
        self.anthropic_client = None
        if anthropic and self.anthropic_key:
            try:
                self.anthropic_client = anthropic.Anthropic(
                    api_key=self.anthropic_key, base_url=self.anthropic_base_url
                )
                logger.info(
                    f"🦾 Anthropic client initialized with base_url: {self.anthropic_base_url}"
                )
            except Exception as e:
                logger.error(f"❌ Failed to initialize Anthropic client: {e}")

        # Initialize Ollama client
        if self.ollama_base:
            try:
                from ollama import Client
                # Create client with host and auth (extra kwargs passed to httpx)
                client_kwargs = {}
                if self.ollama_api_key:
                    client_kwargs['headers'] = {'Authorization': f'Bearer {self.ollama_api_key}'}
                self.ollama_client = Client(host=self.ollama_base, **client_kwargs)
                logger.info(f"🦙 Ollama client initialized: {self.ollama_base} with model {self.ollama_model}")
            except ImportError:
                logger.warning("⚠️ ollama package not installed. Run: pip install ollama")
            except Exception as e:
                logger.error(f"❌ Failed to initialize Ollama client: {e}")

        self.last_attempted = 0
        self.last_succeeded = 0
        self.last_error_rate = 0.0

        # Check if any LLM is available
        has_llm = bool(self.api_key or self.anthropic_key or self.ollama_base)
        if not has_llm:
            logger.warning(
                "⚠️ No LLM API keys (OPENROUTER/ANTHROPIC/OPENAI/OLLAMA) found. RAG generation is disabled."
            )
        else:
            # Determine provider
            if self.ollama_base:
                provider = f"Ollama ({self.ollama_base})"
            elif "openrouter.ai" not in self.openai_base:
                provider = "OpenAI-Compatible"
            else:
                provider = "OpenRouter"
            logger.info(
                f"🤖 RAG Mutator using {provider} with model {self.model_name}"
            )

        # Load Elite Seeds
        self.elite_seeds = []
        seed_path = os.path.join(os.path.dirname(__file__), seed_file)
        if os.path.exists(seed_path):
            try:
                with open(seed_path, "r") as f:
                    self.elite_seeds = json.load(f)
                logger.info(
                    f"🧬 Loaded {len(self.elite_seeds)} Elite DNA seeds from {seed_path}"
                )
            except Exception as e:
                logger.error(f"❌ Failed to load Elite Seeds: {e}")
        else:
            logger.warning(
                f"⚠️ Seed file not found at {seed_path}. Will run without RAG context."
            )

    def _pick_context_dna(self, n=3):
        """Randomly pick top performing DNA strings."""
        if not self.elite_seeds:
            return []

        # Filter top seeds
        high_performers = [
            s["expression"] for s in self.elite_seeds if s.get("sharpe", 0) >= 1.25
        ]
        if not high_performers:
            high_performers = [s["expression"] for s in self.elite_seeds]

        return random.sample(high_performers, min(n, len(high_performers)))

    def _fetch_negative_examples(self, n=3) -> List[str]:
        """Fetch recently failed alphas generated by RAG."""
        db_path = os.path.join(os.path.dirname(__file__), "alpha_results.db")
        recent_fails = []
        if os.path.exists(db_path):
            try:
                import sqlite3

                with sqlite3.connect(db_path) as conn:
                    cursor = conn.cursor()
                    # Get recent RAG alphas that either failed structural checks or got poor Sharpe
                    cursor.execute(
                        """
                        SELECT expression FROM alphas 
                        WHERE theme='rag_llm' AND error='' AND sharpe < 0.5 
                        ORDER BY id DESC LIMIT ?
                    """,
                        (n,),
                    )
                    recent_fails = [r[0] for r in cursor.fetchall()]
            except Exception as e:
                logger.warning(f"Could not load recent fails: {e}")
        return recent_fails

    def _fetch_dna_hints(self) -> str:
        """Fetch statistical operator hints from AlphaDNA generator_weights.json."""
        weights_path = os.path.join(os.path.dirname(__file__), "generator_weights.json")
        if os.path.exists(weights_path):
            try:
                with open(weights_path, "r") as f:
                    w_data = json.load(f)
                    ops = w_data.get("operator_weights", {})
                    if ops:
                        sorted_ops = sorted(ops.items(), key=lambda x: x[1])
                        worst = [op for op, v in sorted_ops[:3]]
                        best = [op for op, v in sorted_ops[-4:]]
                        return f"STATISTICAL HINT: Avoid operators with recent low winrates like {worst}. Strongly prefer top-performing operators like {best}."
            except Exception as e:
                logger.warning(f"Could not load DNA weights: {e}")
        return ""

    def _build_prompt(
        self,
        dna_context: List[str],
        recent_fails: List[str] = None,
        dna_hints: str = "",
    ) -> str:
        prompt = "You are a Quantitative Researcher expert in WorldQuant Brain.\n"
        prompt += "Your task is to generate novel, mathematically sound alpha formulas using the WorldQuant Alpha platform syntax.\n\n"

        if dna_context:
            prompt += "✅ POSITIVE EXAMPLES (Learn from these):\n"
            prompt += "Here are some highly successful 'Elite' alpha expressions that passed simulation with high Sharpe:\n"
            for i, dna in enumerate(dna_context, 1):
                prompt += f"Elite Alpha {i}: {dna}\n"

        if recent_fails:
            prompt += "\n🔴 NEGATIVE ANTI-PATTERNS (AVOID THESE):\n"
            prompt += "The following alphas were recently generated but FAILED simulation (e.g. negative Sharpe).\n"
            prompt += "Analyze their structures and DO NOT generate formulas suspiciously similar to these:\n"
            for i, fail in enumerate(recent_fails, 1):
                prompt += f"Failed Alpha {i}: {fail}\n"

        if dna_hints:
            prompt += f"\n💡 {dna_hints}\n"

        if dna_context:
            prompt += "\nINSTRUCTIONS:\n"
            prompt += "1. Study the structure, operator groupings, and logic of the Elite alphas above.\n"
            prompt += "2. CRITICAL: Pay attention to the SIGN of the expressions. Many successful alphas are REVERSAL signals (starting with -1 * or -rank).\n"
            prompt += "3. STABILITY: Avoid using raw 1-day 'returns' as the core signal. Use longer lookbacks (e.g., ts_delta(x, 5), ts_mean(x, 20), ts_corr(x, y, 10)) to keep Turnover < 70%.\n"
            prompt += "4. NOVELTY: Do NOT strictly copy. Create a completely newly mixed alpha (F1 Generation).\n"
            prompt += "5. STRUCTURE: AVOID using 'rank()' as the primary outer wrapper. Use advanced operators like 'group_neutralize(', 'zscore(', 'ts_zscore(', 'group_zscore(', or 'scale('.\n"
            prompt += "   Example of robust reversal: 'group_neutralize(-1 * ts_rank(ts_delta(close, 5), 20), subindustry)'\n"
        else:
            prompt += "INSTRUCTIONS: Generate a robust mean-reversion (reversal) alpha. Use signs like -1 * ... and lookbacks > 5 days. DO NOT rely on basic 'rank(...)'. Explore 'group_neutralize' or 'zscore'.\n"

        prompt += "\nFormat your response by ONLY providing the final mathematical expression. NO markdown, NO text, NO explanation. Just the raw string."
        return prompt

    def _clean_response(self, text: str) -> str:
        """Strip markdown and text around the formula."""
        lines = text.strip().split("\n")
        for line in reversed(lines):
            line = line.strip().strip("`").strip().strip('"').strip("'")
            if any(
                op in line
                for op in ["rank", "volume", "close", "open", "delay", "returns", "ts_"]
            ):
                return line
        return lines[-1].strip("`").strip()

    def generate_f1_alphas(self, batch_size=3) -> List[AlphaCandidate]:
        """
        Generate multiple F1 Elite Candidates using RAG LLM asynchronously.
        """
        import concurrent.futures

        candidates = []
        # Check if any LLM is available (OpenRouter, Anthropic, or Ollama)
        has_llm = bool(self.api_key or self.ollama_base)
        if not has_llm:
            self.last_attempted = batch_size
            self.last_succeeded = 0
            self.last_error_rate = 1.0
            return candidates
        logger.info(
            f"🧠 Prompting OpenRouter for {batch_size} Elite AI Alphas in parallel..."
        )

        # Optimize with Connection Pooling for parallel requests
        session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(pool_connections=32, pool_maxsize=32)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        session.headers.update(
            {
                "HTTP-Referer": "https://worldquant-alpha-factory.local",
                "Content-Type": "application/json",
            }
        )
        if self.api_key:
            session.headers["Authorization"] = f"Bearer {self.api_key}"

        # Cache global hints so we don't query DB for each thread concurrently
        # Use hybrid retrieval for diverse DNA + diverse negative examples
        n_dna = int(os.getenv("RAG_N_DNA", "3"))
        n_neg = int(os.getenv("RAG_NEGATIVE_EXAMPLES", "10"))
        min_sim = float(os.getenv("RAG_MIN_SIMILARITY", "0.3"))
        use_diverse = os.getenv("RAG_USE_DIVERSE", "true").lower() == "true"

        if use_diverse:
            dna_context, global_recent_fails = _hybrid_retrieve(
                self, n_dna=n_dna, n_neg=n_neg, min_similarity=min_sim
            )
        else:
            dna_context = self._pick_context_dna(n=n_dna)
            global_recent_fails = self._fetch_negative_examples(n=n_neg)

        global_dna_hints = self._fetch_dna_hints()

        def _fetch_one():
            dna_context = self._pick_context_dna(n=3)
            prompt = self._build_prompt(
                dna_context, global_recent_fails, global_dna_hints
            )
            max_retries = 10
            retry_delay = 10

            # Option A: Ollama (local/cloud)
            if self.ollama_client and self.ollama_base:
                for attempt in range(max_retries):
                    try:
                        logger.info(f"🦙 Prompting Ollama ({self.ollama_model}) [Attempt {attempt + 1}/{max_retries}]...")
                        response = self.ollama_client.chat(
                            model=self.ollama_model,
                            messages=[{"role": "user", "content": prompt}],
                            options={"temperature": 0.8, "top_p": 0.95}
                        )
                        raw_text = response.get("message", {}).get("content", "")
                        clean_expr = normalize_expression_aliases(
                            self._clean_response(raw_text)
                        )
                        if clean_expr and len(clean_expr) > 10:
                            logger.info(f"💡 AI (Ollama) Generated formula: {clean_expr}")
                            return AlphaCandidate(
                                expression=clean_expr,
                                theme="rag_llm",
                                family="elite_f1",
                                mutation_type="llm_rag_injection",
                            )
                    except Exception as e:
                        logger.warning(f"🦙 Ollama API Error (Attempt {attempt + 1}/{max_retries}): {e}")
                        if attempt < max_retries - 1:
                            time.sleep(retry_delay)
                            continue

            # Option B: Anthropic
            is_anthropic_model = self.model_name.startswith("claude")
            if self.anthropic_client and (is_anthropic_model or not self.api_key):
                for attempt in range(max_retries):
                    try:
                        logger.info(
                            f"🔮 Prompting Anthropic ({self.model_name}) [Attempt {attempt + 1}/{max_retries}]..."
                        )
                        message = self.anthropic_client.messages.create(
                            model=self.model_name,
                            max_tokens=1024,
                            messages=[{"role": "user", "content": prompt}],
                        )
                        raw_text = message.content[0].text if message.content else ""
                        clean_expr = normalize_expression_aliases(
                            self._clean_response(raw_text)
                        )
                        if clean_expr and len(clean_expr) > 10:
                            logger.info(
                                f"💡 AI (Anthropic) Generated formula: {clean_expr}"
                            )
                            return AlphaCandidate(
                                expression=clean_expr,
                                theme="rag_llm",
                                family="elite_f1",
                                mutation_type="llm_rag_injection",
                            )
                    except Exception as e:
                        logger.error(
                            f"Anthropic API Error (Attempt {attempt + 1}): {e}"
                        )
                        if attempt < max_retries - 1:
                            time.sleep(retry_delay)
                            continue
                if not self.api_key:
                    return None  # Fallback only if we have OpenRouter

            # Option B: OpenRouter
            if not self.api_key:
                return None

            model_candidates = [self.model_name, "openrouter/auto"]
            for model_name in model_candidates:
                if model_name.startswith("claude") and not is_anthropic_model:
                    pass

                for attempt in range(max_retries):
                    payload = {
                        "model": model_name,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.8,
                        "top_p": 0.95,
                        "stream": False,
                    }
                    try:
                        response = session.post(self.api_url, json=payload, timeout=60)
                        if response.status_code == 200:
                            raw_text = (
                                response.json()
                                .get("choices", [{}])[0]
                                .get("message", {})
                                .get("content", "")
                            )
                            clean_expr = normalize_expression_aliases(
                                self._clean_response(raw_text)
                            )
                            if clean_expr and len(clean_expr) > 10:
                                logger.info(
                                    f"💡 AI (OpenRouter) Generated formula: {clean_expr}"
                                )
                                return AlphaCandidate(
                                    expression=clean_expr,
                                    theme="rag_llm",
                                    family="elite_f1",
                                    mutation_type="llm_rag_injection",
                                )
                        elif response.status_code in (502, 503, 504, 429):
                            logger.warning(
                                f"OpenRouter API {response.status_code} (Attempt {attempt + 1}/{max_retries}), retrying in {retry_delay}s..."
                            )
                            if attempt < max_retries - 1:
                                time.sleep(retry_delay)
                                continue
                        elif (
                            response.status_code in (400, 404)
                            and model_name != "openrouter/auto"
                        ):
                            logger.warning(
                                f"Model '{model_name}' unavailable ({response.status_code}), "
                                "retrying with openrouter/auto"
                            )
                            break  # Try next model
                        else:
                            logger.error(
                                f"OpenRouter API Error: {response.status_code} - {response.text}"
                            )
                            break  # Non-retryable
                    except requests.exceptions.RequestException as e:
                        logger.error(
                            f"Failed to connect to LLM (Attempt {attempt + 1}): {e}"
                        )
                        if attempt < max_retries - 1:
                            time.sleep(retry_delay)
                            continue
                    except Exception as ex:
                        logger.error(
                            f"Unexpected error in _fetch_one (Attempt {attempt + 1}): {ex}"
                        )
                        break
            return None

        # Execute all HTTP requests concurrently
        attempted = batch_size
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(8, batch_size)
        ) as executor:
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
