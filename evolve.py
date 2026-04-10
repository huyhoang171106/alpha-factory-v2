"""
evolve.py — Genetic Evolution for Alpha Expressions
Takes passing alphas → mutates → creates better variants.
Reference: worldquant-miner/generation_two/evolution/
"""

import re
import random
from typing import List
from generator import AlphaGenerator
from alpha_seeds import PRICES, LOOKBACKS, GROUPS


class AlphaEvolver:
    """Evolve passing alphas into better variants"""

    def __init__(self):
        self.generator = AlphaGenerator()

    def grid_search_mutations(self, expr: str) -> List[str]:
        """Systematically generate variants across common lookbacks and groups to find the Sharpe peak."""
        variants = set()
        
        # Grid Search Lookbacks
        numbers = re.findall(r'\b(\d+)\b', expr)
        if numbers:
            old = numbers[0]  # Just take the first common number as anchor
            for new_l in [5, 10, 15, 20, 40, 60]:
                if str(new_l) != old:
                    variants.add(expr.replace(old, str(new_l), 1))
                    
        # Grid Search Neutralizations
        for g in GROUPS:
            if g in expr:
                for new_g in GROUPS:
                    if new_g != g:
                        variants.add(expr.replace(g, new_g))
                        
        return list(variants)

    def mutate_lookback(self, expr: str) -> str:
        """Change a random lookback period"""
        numbers = re.findall(r'\b(\d+)\b', expr)
        if not numbers:
            return expr
        old = random.choice(numbers)
        new = str(random.choice(LOOKBACKS))
        return expr.replace(old, new, 1)

    def swap_price(self, expr: str) -> str:
        """Swap a price field for another"""
        for p in random.sample(PRICES, len(PRICES)):
            if p in expr:
                new_p = random.choice([x for x in PRICES if x != p])
                return expr.replace(p, new_p, 1)
        return expr

    def swap_group(self, expr: str) -> str:
        """Swap group neutralization level"""
        for g in GROUPS:
            if g in expr:
                new_g = random.choice([x for x in GROUPS if x != g])
                return expr.replace(g, new_g)
        return expr

    def add_decay(self, expr: str) -> str:
        """Wrap expression with ts_decay_linear"""
        d = random.choice([3, 5, 7, 10])
        return f"ts_decay_linear({expr}, {d})"

    def add_neutralize(self, expr: str) -> str:
        """Wrap with group_neutralize"""
        g = random.choice(GROUPS)
        return f"group_neutralize({expr}, {g})"

    def flip_sign(self, expr: str) -> str:
        """Flip the sign of the expression"""
        if expr.startswith("-"):
            return expr[1:].strip()
        if expr.startswith("(-"):
            return expr[2:].rstrip(")")
        return f"-({expr})"

    def add_regime_condition(self, expr: str) -> str:
        """
        Condition signal on volatility regime.
        Low vol → trend; High vol → mean revert.
        """
        d = random.choice([5, 10, 20])
        d2 = d * 4
        mode = random.choice(["trend", "revert"])
        if mode == "trend":
            # Signal works when vol is LOW (trending regime)
            return f"(-rank(ts_std_dev(returns, {d}))) * ({expr})"
        else:
            # Signal works when vol is HIGH (mean-revert regime)
            return f"rank(ts_std_dev(returns, {d})) * ({expr})"

    def add_quality_filter(self, expr: str) -> str:
        """
        Multiply signal by quality proxy (Sharpe-like).
        Strong alpha only for high-quality stocks.
        """
        d = random.choice([20, 40, 60])
        return f"({expr}) * rank(ts_mean(returns, {d}) / (ts_std_dev(returns, {d}) + 0.001))"

    def volume_confirm(self, expr: str) -> str:
        """
        Confirm signal with volume expansion.
        Only take position when unusual volume confirms.
        """
        d = random.choice([10, 20])
        direction = random.choice([1, -1])
        vol_sig = "rank(volume / adv20)" if direction > 0 else "-rank(volume / adv20)"
        return f"({expr}) * {vol_sig}"

    def crossover(self, expr1: str, expr2: str) -> str:
        """Combine two expressions using AST or operators"""
        from alpha_ast import tree_crossover
        op = random.choice([
            lambda a, b: f"(rank({a}) * rank({b}))",
            lambda a, b: tree_crossover(a, b),
            lambda a, b: tree_crossover(b, a),
            lambda a, b: f"rank({a} + {b})",
        ])
        return op(expr1, expr2)

    def evolve_single(self, expr: str, n_variants: int = 5) -> List[str]:
        """Generate n variants of a single expression"""
        # Weight financial-aware mutations higher
        mutations = [
            self.mutate_lookback,
            self.mutate_lookback,  # 2x weight (most safe)
            self.swap_price,
            self.swap_group,
            self.add_decay,
            self.add_neutralize,
            self.flip_sign,
            self.add_regime_condition,  # NEW
            self.add_quality_filter,    # NEW
            self.volume_confirm,        # NEW
        ]

        variants = set()
        for _ in range(n_variants * 4):
            mutator = random.choice(mutations)
            try:
                variant = mutator(expr)
            except Exception:
                continue
            if variant != expr and len(variant) < 800:  # stricter limit
                variants.add(variant)
            if len(variants) >= n_variants:
                break

        return list(variants)

    def evolve_batch(
        self,
        passing_exprs: List[str],
        n_per_alpha: int = 3,
        n_crossovers: int = 5,
    ) -> List[str]:
        """
        Evolve a batch of passing alphas.
        
        Args:
            passing_exprs: List of expressions that already passed
            n_per_alpha: Number of mutations per alpha
            n_crossovers: Number of crossover attempts
            
        Returns:
            List of new variant expressions
        """
        all_variants = []

        # 1. Mutate each passing alpha
        for expr in passing_exprs:
            grid_variants = self.grid_search_mutations(expr)
            variants = self.evolve_single(expr, n_per_alpha)
            all_variants.extend(grid_variants)
            all_variants.extend(variants)

        # 2. Crossover between top alphas
        if len(passing_exprs) >= 2:
            for _ in range(n_crossovers):
                e1, e2 = random.sample(passing_exprs, 2)
                cross = self.crossover(e1, e2)
                if len(cross) < 1024:
                    all_variants.append(cross)

        return all_variants


# ============================================================
# Quick test
# ============================================================
if __name__ == "__main__":
    evolver = AlphaEvolver()

    seed = "-rank(ts_delta(close, 5))"
    print(f"Seed: {seed}")
    print(f"\nMutations:")
    for v in evolver.evolve_single(seed, 5):
        print(f"  {v}")

    seeds = [
        "-rank(ts_delta(close, 5))",
        "rank(volume / ts_mean(volume, 20))",
    ]
    print(f"\nBatch evolution ({len(seeds)} seeds):")
    batch = evolver.evolve_batch(seeds)
    for v in batch:
        print(f"  {v}")
