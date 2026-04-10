"""
generator.py — Alpha Expression Generator v2
5 strategies: Theme-Driven, Template Mutation, Composite, Group-Aware, Seed-Based
"""

import os
import random
import hashlib
from typing import List, Set, Dict
from alpha_seeds import (
    PRICES, VOLUMES, LOOKBACKS, GROUPS,
    TS_OPS_1D, RANK_OPS, MUTATION_TEMPLATES, COMPOSITE_TEMPLATES,
    get_all_seeds, get_seeds_by_theme, get_random_seeds,
    ARXIV_101, SMC_ALPHAS, COMMUNITY_ALPHAS,
    MICROSTRUCTURE_ALPHAS, QUALITY_ALPHAS, BEHAVIOURAL_ALPHAS,
    CROSS_SECTIONAL_ALPHAS, REGIME_ALPHAS, FUNDAMENTAL_ALPHAS,
    STAT_ARB_ALPHAS, SIGNAL_PROCESSING_ALPHAS,
    ARXIV_ELITE_ALPHAS, ADVANCED_OPS_ALPHAS
)
from alpha_dna import DNAWeights
from alpha_candidate import AlphaCandidate
from alpha_policy import compute_llm_budget_ratio
try:
    from alpha_rag import RAGMutator
except ImportError:
    RAGMutator = None



import json

class AlphaGenerator:
    """
    Generates unique alpha expressions using 5 hierarchical strategies.
    
    Priority order (quality → quantity):
    1. Theme-driven hypotheses  → high hit rate (financial edge)
    2. Composite (signal×filter) → medium-high hit rate
    3. Group-aware signals       → sector/industry alpha
    4. Template mutations        → volume filler
    5. Seed sampling             → baseline variety

    Bias:
    If a DNAWeights object is provided, the generator will favor
    operators and fields that have performed well in previous runs.
    """

    THEMES = [
        "arxiv", "smc", "community",
        "microstructure", "quality", "behavioural", "cross_sectional", "regime",
        "fundamental", "stat_arb", "signal_processing",
        "arxiv_elite", "advanced_ops"
    ]

    HYPOTHESIS_BLOCKS = {
        "mean_reversion": {
            "base": [
                lambda d: f"returns",
                lambda d: f"ts_delta(close, {d})",
            ],
            "filters": [
                lambda d: f"abs(returns)",
                lambda d: f"volume / adv20",
                lambda d: f"adv20 / ts_mean(adv20, {d})",
            ],
            "direction": "-",
        },
        "momentum": {
            "base": [
                lambda d: f"returns",
                lambda d: f"ts_mean(returns, {d})",
            ],
            "filters": [
                lambda d: f"volume / adv20",
                lambda d: f"adv20 / ts_mean(adv20, {d})",
            ],
            "direction": "+",
        },
        "liquidity": {
            "base": [
                lambda d: f"volume / adv20",
                lambda d: f"adv20 / ts_mean(adv20, {d})",
            ],
            "filters": [
                lambda d: f"returns",
                lambda d: f"abs(returns)",
            ],
            "direction": "+",
        },
        "overreaction": {
            "base": [
                lambda d: f"returns",
                lambda d: f"abs(returns)",
            ],
            "filters": [
                lambda d: f"adv20 / ts_mean(adv20, {d})",
                lambda d: f"volume / adv20",
            ],
            "direction": "-",
        },
    }

    def __init__(
        self,
        dna_weights: DNAWeights = None,
        mining_level: int = 4,
        generation_mode: str | None = None,
    ):
        self._seen: Set[str] = set()
        self._seeds = get_all_seeds()
        self._theme_seeds: Dict[str, List[str]] = {
            t: get_seeds_by_theme(t) for t in self.THEMES
        }
        self.weights = dna_weights or DNAWeights.default()
        self.mining_level = max(1, min(5, mining_level))
        self.rag_mutator = RAGMutator() if RAGMutator else None
        self._hypotheses = self._load_hypotheses()
        self.generation_mode = (generation_mode or os.getenv("GENERATOR_MODE", "legacy")).strip().lower()

    def _load_hypotheses(self) -> List[dict]:
        path = os.path.join(os.path.dirname(__file__), "..", "data", "hypotheses", "iqc_core.json")
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []

    def _hash(self, expr: str) -> str:
        normalized = expr.strip().lower().replace(" ", "")
        return hashlib.md5(normalized.encode()).hexdigest()

    def _add_if_new(
        self,
        expr: str,
        results: list,
        theme: str = "unknown",
        mutation_type: str = "seed",
        family: str = "",
    ) -> bool:
        h = self._hash(expr)
        if h not in self._seen:
            self._seen.add(h)
            candidate = AlphaCandidate(
                expression=expr,
                theme=theme,
                family=family,
                mutation_type=mutation_type,
            )
            if hasattr(self, '_current_hypothesis'):
                candidate.hypothesis = self._current_hypothesis
            results.append(candidate)
            return True
        return False

    def _get_biased_lookback(self, range_type: str = "mid") -> int:
        """Sample a lookback based on DNA preferences if available"""
        import random
        dist = self.weights.lookback_distribution
        # Use simple random if no specific bias logic is in DNAWeights beyond defaults
        # Actually, let's implement a simplified version of AlphaDNA.get_preferred_lookback here
        mode = dist.get("pref_mode", range_type)
        if mode == "short":
            lo, hi = dist.get("short_min", 3), dist.get("short_max", 10)
        elif mode == "long":
            lo, hi = dist.get("long_min", 40), dist.get("long_max", 120)
        else: # mid
            lo, hi = dist.get("mid_min", 10), dist.get("mid_max", 30)
        return random.randint(lo, hi)

    def _get_weighted_fields(self, n: int = 1) -> List[str]:
        """Sample fields weighted by DNA performance"""
        fields = list(self.weights.field_weights.keys())
        # Provide some fallback if weights are empty
        if not fields: return random.sample(PRICES, min(n, len(PRICES)))
        
        weights = [self.weights.field_weights[f] for f in fields]
        return random.choices(fields, weights=weights, k=n)

    def _get_weighted_op(self, pool: List[str]) -> str:
        """Sample an operator from a pool, weighted by DNA performance if tracked"""
        # Intersection of pool and tracked dna weights
        tracked = [op for op in pool if op in self.weights.operator_weights]
        if not tracked: return random.choice(pool)

        weights = [self.weights.operator_weights[op] for op in tracked]
        return random.choices(tracked, weights=weights, k=1)[0]

    @staticmethod
    def _is_structurally_valid_hypothesis_expr(expr: str) -> bool:
        """
        Hard constraints for hypothesis-driven generation:
        - must contain cross-sectional rank
        - must contain direction signal (+/-)
        - must be group neutralized
        """
        text = (expr or "").replace(" ", "")
        if "rank(" not in text:
            return False
        if "group_neutralize(" not in text:
            return False
        if "group_neutralize(ts_decay_linear(" not in text:
            return False
        has_negative = "-rank(" in text or "*-" in text or "(-rank(" in text
        has_positive = "*rank(" in text or "rank(" in text
        return has_negative or has_positive

    def _build_hypothesis_expression(self, hypothesis: str) -> str:
        spec = self.HYPOTHESIS_BLOCKS[hypothesis]
        d_base = self._get_biased_lookback("short")
        d_filter = self._get_biased_lookback("mid")
        decay = random.choice([5, 6, 7, 8, 9])

        base_signal = random.choice(spec["base"])(d_base)
        filter_signal = random.choice(spec["filters"])(d_filter)

        # Borrowed from BRAIN community tips:
        # - regression_neut() to orthogonalize core signal vs broad-market drift
        # - pasteurize() for cross-universe robustness in cross-sectional ops
        if random.random() < 0.22:
            base_signal = f"regression_neut({base_signal}, ts_mean(returns, {d_filter}))"
        if random.random() < 0.28:
            filter_signal = f"pasteurize({filter_signal})"

        combined = f"rank({base_signal}) * rank({filter_signal})"

        if spec["direction"] == "-":
            directional = f"-({combined})"
        else:
            directional = combined

        smoothed = f"ts_decay_linear({directional}, {decay})"
        group_expr = self._choose_neutralization_group()
        return f"group_neutralize({smoothed}, {group_expr})"

    @staticmethod
    def _choose_neutralization_group() -> str:
        """
        Include classic and custom/double neutralization groups.
        """
        options = [
            "subindustry",
            "industry",
            "sector",
            "bucket(rank(cap), range=\"0.2,1,0.2\")",
            "densify(industry * 1000 + bucket(rank(cap), range=\"0.2,1,0.2\"))",
        ]
        weights = [0.52, 0.22, 0.14, 0.07, 0.05]
        return random.choices(options, weights=weights, k=1)[0]

    def generate_hypothesis_driven(self, n: int = 20) -> List[AlphaCandidate]:
        """
        Agent-style hypothesis-first generator:
        hypothesis -> base -> filter -> direction -> smooth -> neutralize
        """
        results: List[AlphaCandidate] = []
        attempts = 0
        max_attempts = max(n * 12, 50)
        hypotheses = list(self.HYPOTHESIS_BLOCKS.keys())

        while len(results) < n and attempts < max_attempts:
            attempts += 1
            hypothesis = random.choice(hypotheses)
            expr = self._build_hypothesis_expression(hypothesis)
            if not self._is_structurally_valid_hypothesis_expr(expr):
                continue

            self._current_hypothesis = hypothesis
            self._add_if_new(
                expr,
                results,
                theme=hypothesis,
                mutation_type="hypothesis_driven",
                family=hypothesis,
            )
            if hasattr(self, "_current_hypothesis"):
                delattr(self, "_current_hypothesis")
        return results

    # =========================================================
    # Strategy 1: Theme-Driven Hypothesis Generator
    # Mỗi theme có financial rationale rõ ràng → pass rate cao hơn
    # =========================================================
    def generate_from_themes(self, n: int = 20) -> List[AlphaCandidate]:
        """
        Generate from the 5 factor themes. Each theme has proven financial edge.
        Randomly picks from theme seeds + mutates parameters.
        """
        # Tagged theme builders: (theme_name, builder_fn)
        tagged_builders = [
            # --- MICROSTRUCTURE ---
            ("microstructure", lambda: f"-rank(ts_decay_linear(abs(returns) / ({self._get_weighted_fields()[0]} + 1), {self._get_biased_lookback('short')}))"),
            ("microstructure", lambda: f"group_neutralize(rank(ts_sum(returns * volume, {self._get_biased_lookback('short')}) / (ts_sum(abs(returns) * volume, {self._get_biased_lookback('short')}) + 0.001)), subindustry)"),
            ("microstructure", lambda: f"rank(abs(ts_delta(close, 1)) / (ts_mean(adv20, {self._get_biased_lookback('mid')}) + 1))"),
            ("microstructure", lambda: f"-rank(ts_corr(abs(returns), volume, {self._get_biased_lookback('mid')})) * rank(ts_decay_linear(volume, {self._get_biased_lookback('short')}))"),
            ("microstructure", lambda: f"rank(ts_sum(sign(ts_delta(close, 1)) * volume, {self._get_biased_lookback('mid')}) / adv20)"),
            ("microstructure", lambda: f"rank(volume / adv20) * (-rank(abs(returns)))"),
            ("microstructure", lambda: f"rank(ts_mean(close / vwap - 1, {self._get_biased_lookback('short')}))"),

            # --- QUALITY (Risk-Adjusted) ---
            ("quality", lambda: f"-rank(ts_decay_linear({self._get_weighted_op(['ts_std_dev', 'ts_skewness'])}(returns, {self._get_biased_lookback('long')}), {self._get_biased_lookback('short')}))"),
            ("quality", lambda: f"group_neutralize(rank(ts_mean(returns, {self._get_biased_lookback('mid')}) / (ts_std_dev(returns, {self._get_biased_lookback('mid')}) + 0.001)), sector)"),
            ("quality", lambda: f"rank(ts_corr(close, ts_decay_linear(close, {self._get_biased_lookback('long')}), {self._get_biased_lookback('mid')}))"),
            ("quality", lambda: f"-rank(ts_std_dev({self._get_weighted_fields()[0]}, {self._get_biased_lookback('mid')}) / ts_mean({self._get_weighted_fields()[0]}, {self._get_biased_lookback('mid')}))"),
            ("quality", lambda: f"rank(ts_sum(sign(close - ts_mean(close, {self._get_biased_lookback('mid')})), {self._get_biased_lookback('mid')}))"),
            ("quality", lambda: f"group_rank(rank(ts_mean(returns, {self._get_biased_lookback('short')}) / (ts_std_dev(returns, {self._get_biased_lookback('mid')}) + 0.001)), 1, subindustry)"),
            ("quality", lambda: f"-rank(ts_skewness(returns, {self._get_biased_lookback('long')})) * rank(return_equity)"),

            # --- IQC STRATEGIST LAYER: Price Shock (Mean Reversion) ---
            ("price_shock", lambda: f"-rank(ts_decay_linear(abs(returns), {self._get_biased_lookback('short')})) * rank(ts_delta(close, {self._get_biased_lookback('short')}))"),
            ("price_shock", lambda: f"group_neutralize(rank(ts_sum(returns, {self._get_biased_lookback('short')}) / ts_std_dev(close, {self._get_biased_lookback('mid')})), subindustry)"),
            ("price_shock", lambda: f"-rank((close - ts_min(low, {self._get_biased_lookback('mid')})) / (ts_max(high, {self._get_biased_lookback('mid')}) - ts_min(low, {self._get_biased_lookback('mid')})))"),

            # --- IQC STRATEGIST LAYER: Volume Anomaly (Institutional Flow) ---
            ("volume_anomaly", lambda: f"rank((volume - adv20) / adv20) * -rank(returns)"),
            ("volume_anomaly", lambda: f"group_neutralize(rank(ts_sum(returns * volume, {self._get_biased_lookback('short')})), industry)"),
            ("volume_anomaly", lambda: f"rank(ts_delta(volume, {self._get_biased_lookback('short')})) * rank(ts_decay_linear(close, {self._get_biased_lookback('short')}))"),

            # --- IQC STRATEGIST LAYER: Volatility Compression (Breakout) ---
            ("vol_compression", lambda: f"rank(ts_std_dev(close, {self._get_biased_lookback('short')}) / ts_std_dev(close, {self._get_biased_lookback('long')})) * rank(returns)"),
            ("vol_compression", lambda: f"group_neutralize(-rank(ts_std_dev(returns, {self._get_biased_lookback('mid')})), sector)"),
            ("vol_compression", lambda: f"rank(ts_delta(high - low, {self._get_biased_lookback('short')}))"),

            # --- CROSS-SECTIONAL (Orthogonalized Momentum) ---
            ("cross_sectional", lambda: f"group_neutralize(rank(returns - group_mean(returns, 1, {random.choice(GROUPS)})), subindustry)"),
            ("cross_sectional", lambda: f"group_neutralize(rank(ts_decay_linear(returns, {self._get_biased_lookback('mid')})), {random.choice(['sector','industry'])})"),
            ("cross_sectional", lambda: f"group_rank(rank(ts_mean(returns, {self._get_biased_lookback('long')})), 1, {random.choice(['sector','industry'])})"),
            ("cross_sectional", lambda: f"-rank(ts_corr(returns, group_mean(returns, 1, {random.choice(['sector','industry'])}), {self._get_biased_lookback('mid')}))"),
            ("cross_sectional", lambda: f"group_neutralize(rank(ts_delta(close, {self._get_biased_lookback('mid')})), {random.choice(['sector','industry'])})"),
            ("cross_sectional", lambda: f"group_neutralize(rank(volume / adv20), {random.choice(['sector','industry'])})"),
            ("cross_sectional", lambda: f"rank(close / vwap) - group_mean(rank(close / vwap), 1, {random.choice(['sector','industry'])})"),

            # --- REGIME & FUNDAMENTALS ---
            ("regime", lambda: f"group_neutralize(rank(ts_std_dev(returns, {self._get_biased_lookback('short')})) * (-rank(ts_delta(close, {self._get_biased_lookback('short')}))), subindustry)"),
            ("regime", lambda: f"-rank(ts_std_dev(returns, {self._get_biased_lookback('short')})) * rank(ts_mean(returns, {self._get_biased_lookback('mid')}))"),
            ("regime", lambda: f"rank(volume / adv20) * rank(ts_mean(returns, {self._get_biased_lookback('short')}))"),
            ("regime", lambda: f"group_neutralize(rank(operating_margin), subindustry) * -rank(returns)"),
            ("regime", lambda: f"rank(ts_delta(ebitda, {self._get_biased_lookback('long')}))"),
            ("regime", lambda: f"rank(ts_sum(sign(ts_delta(close, 1)) * volume, {self._get_biased_lookback('mid')}) / ts_sum(volume, {self._get_biased_lookback('mid')}))"),
            ("regime", lambda: f"rank(ts_corr(returns, ts_delta(volume, 1), {self._get_biased_lookback('short')}))"),
            ("regime", lambda: f"-rank(ts_delta(close, {self._get_biased_lookback('short')})) * rank(ts_std_dev(returns, {self._get_biased_lookback('short')}) / ts_std_dev(returns, {self._get_biased_lookback('mid')}))"),
            ("regime", lambda: f"rank(ts_std_dev(returns, {self._get_biased_lookback('short')}) / ts_std_dev(returns, {self._get_biased_lookback('long')}))"),
        ]

        # Map seed lists to themes
        theme_seed_map = [
            ("arxiv", ARXIV_101),
            ("smc", SMC_ALPHAS),
            ("community", COMMUNITY_ALPHAS),
            ("microstructure", MICROSTRUCTURE_ALPHAS),
            ("quality", QUALITY_ALPHAS),
            ("behavioural", BEHAVIOURAL_ALPHAS),
            ("cross_sectional", CROSS_SECTIONAL_ALPHAS),
            ("regime", REGIME_ALPHAS),
            ("fundamental", FUNDAMENTAL_ALPHAS),
            ("stat_arb", STAT_ARB_ALPHAS),
            ("signal_processing", SIGNAL_PROCESSING_ALPHAS),
            ("arxiv_elite", ARXIV_ELITE_ALPHAS),
            ("advanced_ops", ADVANCED_OPS_ALPHAS),
        ]

        results = []
        
        # First: use raw theme seeds with proper tagging
        all_tagged_seeds = []
        for theme, seeds in theme_seed_map:
            for s in seeds:
                all_tagged_seeds.append((theme, s))
        sampled = random.sample(all_tagged_seeds, min(n // 2, len(all_tagged_seeds)))
        for theme, seed in sampled:
            self._add_if_new(seed, results, theme=theme, mutation_type="seed")
            if len(results) >= n:
                return results

        # --- PHASE 2 HYPOTHESIS ENGINE ---
        self._generate_from_structured_hypotheses(results)
        
        # Then: fill remaining with parameterized builders
        attempts = 0
        max_attempts = (n - len(results)) * 8
        while len(results) < n and attempts < max_attempts:
            attempts += 1
            try:
                theme, builder = random.choice(tagged_builders)
                raw_expr = builder()
                
                # Base formula
                self._add_if_new(raw_expr, results, theme=theme, mutation_type="theme_direct")
                if len(results) >= n: break
                
                # Smoothed logic
                smoothed = f"ts_decay_linear({raw_expr}, 5)"
                self._add_if_new(smoothed, results, theme=theme, mutation_type="theme_smoothed")
                if len(results) >= n: break
                
                # Neutralized logic
                if "group_neutralize" not in raw_expr:
                    neutralized = f"group_neutralize({raw_expr}, subindustry)"
                    self._add_if_new(neutralized, results, theme=theme, mutation_type="theme_neutralized")
                    
            except Exception:
                continue

        return results

    def _generate_from_structured_hypotheses(self, results: list):
        """Build Step B (Basic) and Step C (Operators) from explicitly stated JSON hypotheses."""
        for hyp in self._hypotheses:
            theme = hyp.get("theme", "unknown")
            hypothesis_reason = hyp.get("hypothesis", "")
            
            # Format lookbacks
            mapping = {
                "short_lb": self._get_biased_lookback("short"),
                "mid_lb": getattr(self, '_get_biased_lookback')("mid"),
                "long_lb": self._get_biased_lookback("long")
            }
            
            base = hyp["base_signal"].format(**mapping)
            filter_sig = hyp["filter_signal"].format(**mapping)
            context = hyp["volatility_context"].format(**mapping)
            
            self._current_hypothesis = hypothesis_reason
            
            # Variant 1: Basic (Step B)
            v1 = f"({base}) * ({filter_sig}) * ({context})"
            self._add_if_new(v1, results, theme=theme, mutation_type="hypo_basic", family=hyp["id"])
            
            # Variant 2: Rank & Delay (Step C)
            v2 = f"rank(delay({base}, 1)) * ({filter_sig})"
            self._add_if_new(v2, results, theme=theme, mutation_type="hypo_delayed", family=hyp["id"])
            
            # Variant 3: Group Neutralization (Step C)
            v3 = f"group_neutralize(rank({base}), subindustry) * rank({filter_sig})"
            self._add_if_new(v3, results, theme=theme, mutation_type="hypo_neutralized", family=hyp["id"])
            
            # Clean up tracking
            if hasattr(self, '_current_hypothesis'):
                delattr(self, '_current_hypothesis')

    # =========================================================
    # Strategy 2: Composite (signal × confirming filter)
    # High precision: two independent signals must agree
    # =========================================================
    def generate_composites(self, n: int = 15) -> List[AlphaCandidate]:
        """
        Combine a primary signal with a confirming filter.
        E.g.: mean-reversion signal x regime filter
        """
        results = []
        attempts = 0
        max_attempts = n * 10

        while len(results) < n and attempts < max_attempts:
            attempts += 1
            template = random.choice(COMPOSITE_TEMPLATES)

            d1 = self._get_biased_lookback("short")
            d2 = self._get_biased_lookback("mid")
            if d2 <= d1:
                d2 = d1 * 3

            try:
                expr = template.format(D1=d1, D2=d2)
                self._add_if_new(expr, results, theme="composite", mutation_type="composite")
            except (KeyError, TypeError):
                continue

        return results

    # =========================================================
    # Strategy 3: Group-Aware Signals
    # Sector/industry neutralize to extract pure stock-specific alpha
    # =========================================================
    def generate_group_aware(self, n: int = 15) -> List[AlphaCandidate]:
        """
        Generate group-neutralized / group-relative expressions.
        These alpha are less correlated with each other (diversity boost).
        """
        patterns = [
            lambda p, d, g: f"group_neutralize(rank(ts_delta({p}, {d})), {g})",
            lambda p, d, g: f"group_neutralize(rank(ts_mean({p}, {d})), {g})",
            lambda p, d, g: f"group_neutralize(rank({p} / ts_mean({p}, {d})), {g})",
            lambda p, d, g: f"group_neutralize(-rank(ts_std_dev({p}, {d})), {g})",
            lambda p, d, g: f"group_rank(rank(ts_delta({p}, {d})), 1, {g})",
            lambda p, d, g: f"group_rank(rank(ts_mean(returns, {d})), 1, {g})",
            lambda p, d, g: f"rank(ts_delta({p}, {d})) - group_mean(rank(ts_delta({p}, {d})), 1, {g})",
            lambda p, d, g: f"rank(ts_mean(returns, {d})) - group_mean(rank(ts_mean(returns, {d})), 1, {g})",
            lambda p, d, g: f"-rank(ts_corr(returns, group_mean(returns, 1, {g}), {d}))",
            lambda p, d, g: f"group_neutralize(rank(volume / adv20), {g})",
            lambda p, d, g: f"group_neutralize(rank(ts_corr({p}, volume, {d})), {g})",
            lambda p, d, g: f"group_neutralize(rank(ts_mean(returns, {d}) / (ts_std_dev(returns, {d}) + 0.001)), {g})",
        ]

        results = []
        attempts = 0

        while len(results) < n and attempts < n * 8:
            attempts += 1
            try:
                p = self._get_weighted_fields()[0]
                d = self._get_biased_lookback("mid")
                g = random.choice(GROUPS)
                fn = random.choice(patterns)
                expr = fn(p, d, g)
                self._add_if_new(expr, results, theme="cross_sectional", mutation_type="group_aware")
            except Exception:
                continue

        return results

    # =========================================================
    # Strategy 4: Template Mutations (volume filler)
    # =========================================================
    def generate_mutations(self, n: int = 15) -> List[AlphaCandidate]:
        """Fill remaining quota with template mutations"""
        results = []
        attempts = 0
        max_attempts = n * 10

        while len(results) < n and attempts < max_attempts:
            attempts += 1
            template = random.choice(MUTATION_TEMPLATES)

            params = {
                "P1": self._get_weighted_fields()[0],
                "P2": self._get_weighted_fields()[0],
                "D":  self._get_biased_lookback("mid"),
                "D1": self._get_biased_lookback("short"),
                "D2": self._get_biased_lookback("long"),
                "W":  self._get_biased_lookback("mid"),
                "G":  random.choice(GROUPS),
            }

            try:
                expr = template.format(**params)
                self._add_if_new(expr, results, theme="template", mutation_type="template_mutation")
            except (KeyError, TypeError):
                continue

        return results

    # =========================================================
    # Strategy 5: Seed mutations (diversity from known alphas)
    # =========================================================
    def generate_from_seed_mutations(self, seed_expr: str, n: int = 5, parent_theme: str = "unknown") -> List[AlphaCandidate]:
        """Mutate an existing alpha expression"""
        import re
        results = []
        # Use seed expression as the family root
        seed_family = hashlib.md5(seed_expr.strip().lower()[:60].encode()).hexdigest()[:12]

        for _ in range(n * 3):
            expr = seed_expr
            mutation = random.choice(["lookback", "price", "wrap", "sign", "group"])

            if mutation == "lookback":
                numbers = re.findall(r'\b(\d+)\b', expr)
                if numbers:
                    old_n = random.choice(numbers)
                    new_n = str(random.choice(LOOKBACKS))
                    if old_n != new_n:
                        expr = expr.replace(old_n, new_n, 1)

            elif mutation == "price":
                old_p = random.choice(PRICES)
                new_p = random.choice(PRICES)
                if old_p in expr and old_p != new_p:
                    expr = expr.replace(old_p, new_p, 1)

            elif mutation == "wrap":
                op = random.choice(["rank", "zscore", "sigmoid"])
                expr = f"{op}({expr})"

            elif mutation == "sign":
                if expr.startswith("-"):
                    expr = expr[1:]
                else:
                    expr = f"-({expr})"

            elif mutation == "group":
                g = random.choice(GROUPS)
                expr = f"group_neutralize({expr}, {g})"

            self._add_if_new(
                expr, results,
                theme=parent_theme,
                mutation_type=f"seed_{mutation}",
                family=seed_family,
            )
            if len(results) >= n:
                break

        return results[:n]

    # =========================================================
    # Strategy 6: Level-5 Economic Intuition (rare-operator alpha)
    # =========================================================
    def generate_level5_intuition(self, n: int = 10) -> List[AlphaCandidate]:
        """
        Build economic-intuition expressions using rare operators and
        explicit regime/turnover controls for lower-correlation alpha mining.
        """
        results = []
        attempts = 0
        max_attempts = n * 12

        templates = [
            lambda d1, d2, g: f"trade_when(group_neutralize(rank(ts_delta(close, {d1})), {g}), rank(volume / adv20) > 0.6, 0)",
            lambda d1, d2, g: f"hump(group_neutralize(rank(ts_zscore(returns, {d2})), {g}), hump=0.01)",
            lambda d1, d2, g: f"vector_neut(rank(ts_delta(close, {d1})), rank(ts_mean(returns, {d2})))",
            lambda d1, d2, g: f"group_neutralize(winsorize(ts_quantile(returns, {d2}, driver='gaussian'), std=3), {g})",
            lambda d1, d2, g: f"trade_when(rank(ts_av_diff(close, {d1})), ts_std_dev(returns, {d2}) > ts_mean(ts_std_dev(returns, {d2}), {max(d2*2, 20)}), 0)",
            lambda d1, d2, g: f"jump_decay(group_neutralize(rank(ts_delta(eps, {d2})), {g}), {d2}, sensitivity=0.5, force=0.1)",
            lambda d1, d2, g: f"group_neutralize(rank(days_from_last_change(eps)) * rank(ts_delta(close, {d1})), {g})",
        ]

        while len(results) < n and attempts < max_attempts:
            attempts += 1
            d1 = self._get_biased_lookback("short")
            d2 = self._get_biased_lookback("mid")
            g = random.choice(["sector", "industry", "subindustry"])
            try:
                expr = random.choice(templates)(d1, d2, g)
                self._add_if_new(expr, results, theme="level5_intuition", mutation_type="level5_rareop")
            except Exception:
                continue

        return results

    # =========================================================
    # Public API
    # =========================================================
    def generate_batch(self, n: int = 50, use_rag: bool = True) -> List[AlphaCandidate]:
        """
        Generate n expressions using all 5 strategies + RAG.
        Distribution (quality-first):
          20% RAG Mutator (highest value, if enabled)
          25% themes (best hit rate)
          20% composites (signal × filter)
          15% group-aware (diversity)
          15% mutations (volume)
          5% seeds  (variety)
        """
        if self.generation_mode == "hypothesis_driven":
            return self.generate_hypothesis_driven(n=n)

        results = []
        
        llm_ratio = 0.10
        if getattr(self, "rag_mutator", None):
            llm_ratio = compute_llm_budget_ratio(
                baseline_ratio=0.10,
                llm_error_rate=float(getattr(self.rag_mutator, "last_error_rate", 0.0) or 0.0),
                submit_fail_rate=float(getattr(self, "runtime_submit_fail_rate", 0.0) or 0.0),
                has_api_key=bool(getattr(self.rag_mutator, "api_key", "")),
            )

        # 1. RAG Mutation (budget-controlled)
        if use_rag and getattr(self, 'rag_mutator', None):
            n_rag = int(n * llm_ratio)
            if n_rag > 0:
                cands = self.rag_mutator.generate_f1_alphas(batch_size=n_rag)
                for c in cands:
                    h = self._hash(c.expression)
                    if h not in self._seen:
                        self._seen.add(h)
                        results.append(c)

        # 2. Re-distribute remaining quota
        remaining = max(0, n - len(results))
        n_l5 = int(remaining * 0.25) if self.mining_level >= 5 else 0
        post_l5 = remaining - n_l5
        n_themes   = int(post_l5 * 0.35)
        n_comp     = int(post_l5 * 0.25)
        n_group    = int(post_l5 * 0.20)
        n_mut      = int(post_l5 * 0.15)
        n_seed     = post_l5 - n_themes - n_comp - n_group - n_mut

        if n_l5 > 0:
            results.extend(self.generate_level5_intuition(n_l5))
        results.extend(self.generate_from_themes(n_themes))
        results.extend(self.generate_composites(n_comp))
        results.extend(self.generate_group_aware(n_group))
        results.extend(self.generate_mutations(n_mut))

        # Seed sampling with slight mutations for diversity
        seeds = get_random_seeds(n_seed * 2)
        for seed in seeds:
            if len([r for r in results if r.expression == seed]) == 0:
                self._add_if_new(seed, results, theme="seed", mutation_type="raw_seed")
            if len(results) >= n:
                break

        random.shuffle(results)
        return results[:n]

    @property
    def total_generated(self) -> int:
        return len(self._seen)


# ============================================================
# Quick test
# ============================================================
if __name__ == "__main__":
    gen = AlphaGenerator()

    print("=== Theme-Driven (10) ===")
    for c in gen.generate_from_themes(10):
        print(f"  [{c.theme}] {c.expression}")

    print("\n=== Composite (5) ===")
    for c in gen.generate_composites(5):
        print(f"  [{c.theme}] {c.expression}")

    print("\n=== Group-Aware (5) ===")
    for c in gen.generate_group_aware(5):
        print(f"  [{c.theme}] {c.expression}")

    print("\n=== Full Batch (50) ===")
    batch = gen.generate_batch(50)
    print(f"  Generated: {len(batch)} unique candidates")
    print(f"  Total unique so far: {gen.total_generated}")
    from collections import Counter
    themes = Counter(c.theme for c in batch)
    print(f"  Theme distribution: {dict(themes)}")
    print(f"\n  Sample:")
    for c in batch[:5]:
        print(f"    [{c.theme}/{c.mutation_type}] {c.expression}")
