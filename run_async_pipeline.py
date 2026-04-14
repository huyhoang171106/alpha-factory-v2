#!/usr/bin/env python3
"""
run_async_pipeline.py - Professional High-Throughput Async Alpha Factory
=======================================================================
Implements a streaming pipeline:
Generator -> Ranker -> Simulator -> ResultHandler/Evolver

Workflow:
1. Generator (Producer): Constantly generates new candidates.
2. Ranker (Filter): Scores candidates and checks for structural duplicates.
3. Simulator (Consumer): Calls WQ Brain API in batches (but streams results).
4. Handler (Storage/Feedback): Saves to DB and updates DNA weights.
"""

import asyncio
import logging
import re
import signal
import time
import os
import random
import hashlib
from types import SimpleNamespace
from typing import Dict, Tuple

from generator import AlphaGenerator
from alpha_ranker import score_expression, passes_complexity_check, regime_sensitivity_score, ic_decay_probability, expression_simplicity_score, turnover_prediction, score_with_meta_model, compute_similarity_penalty, apply_family_crowding_penalty
from alpha_policy import estimate_self_corr_risk, reduce_self_correlation
from tracker import AlphaTracker
from wq_client import WQClient
from alpha_policy import estimate_competition_priority, passes_quality_gate_v2, should_simulate_candidate, pre_submission_gate_from_result, pre_submission_gate, update_self_corr_weights
from alpha_candidate import AlphaCandidate
from alpha_ast import parameter_agnostic_signature
from submit_governor import SubmitGovernor
from quality_diversity import QualityDiversityArchive
from budget_allocator import BudgetAllocator, RegimeAwareArmSelector

# v2 new modules integration (loaded after logger is ready)
_NEW_MODULES_OK = False
try:
    from robustness_lab import BiasDetector, ICStabilityScorer, WalkForwardValidator  # noqa: E402
    from lineage_decay_tracker import LineageDecayTracker  # noqa: E402
    from portfolio_constructor import CorrelationBasedEnsembleBuilder, PreSubmissionGate  # noqa: E402
    from hypothesis_engine import MarketRegimeDetector, HypothesisDrivenGenerator  # noqa: E402
    _NEW_MODULES_OK = True
except ImportError:
    pass  # v2 modules optional — pipeline works without them

# Configuration
GEN_QUEUE_SIZE = int(os.getenv("ASYNC_GEN_QUEUE_SIZE", "600"))
SIM_QUEUE_SIZE = int(os.getenv("ASYNC_SIM_QUEUE_SIZE", "300"))
RANKER_WORKERS = max(1, int(os.getenv("ASYNC_RANKER_WORKERS", "2")))
SIMULATOR_WORKERS = max(1, int(os.getenv("ASYNC_SIMULATOR_WORKERS", "1")))  # WQ API has global concurrency limits
BATCH_SIZE = max(1, int(os.getenv("ASYNC_BATCH_SIZE", "12")))
PRE_RANK_THRESHOLD = 50.0
NOVELTY_WINDOW = 12000
NOVELTY_MIN = float(os.getenv("ASYNC_NOVELTY_MIN", "0.28"))
MIN_CRITIC_SCORE = float(os.getenv("ASYNC_MIN_CRITIC_SCORE", "0.38"))
GEN_BATCH_SIZE = max(5, int(os.getenv("ASYNC_GEN_BATCH_SIZE", "20")))
ASYNC_USE_RAG = os.getenv("ASYNC_USE_RAG", "0").strip().lower() in ("1", "true", "yes")
TIER1_MIN_QUALITY = float(os.getenv("ASYNC_TIER1_MIN_QUALITY", "0.50"))
TIER2_MIN_EV = float(os.getenv("ASYNC_TIER2_MIN_EV", "0.34"))
ADAPTIVE_GATES = os.getenv("ASYNC_ADAPTIVE_GATES", "1").strip().lower() in ("1", "true", "yes")
ADAPTIVE_MIN_PRE_RANK = float(os.getenv("ASYNC_ADAPTIVE_MIN_PRE_RANK", "42.0"))
ADAPTIVE_MAX_PRE_RANK = float(os.getenv("ASYNC_ADAPTIVE_MAX_PRE_RANK", "62.0"))
ADAPTIVE_MIN_QUALITY = float(os.getenv("ASYNC_ADAPTIVE_MIN_QUALITY", "0.44"))
ADAPTIVE_MAX_QUALITY = float(os.getenv("ASYNC_ADAPTIVE_MAX_QUALITY", "0.58"))
ADAPTIVE_MIN_NOVELTY = float(os.getenv("ASYNC_ADAPTIVE_MIN_NOVELTY", "0.18"))
ADAPTIVE_MAX_NOVELTY = float(os.getenv("ASYNC_ADAPTIVE_MAX_NOVELTY", "0.36"))
ADAPTIVE_MIN_EV = float(os.getenv("ASYNC_ADAPTIVE_MIN_EV", "0.30"))
ADAPTIVE_MAX_EV = float(os.getenv("ASYNC_ADAPTIVE_MAX_EV", "0.44"))
ENABLE_D0 = os.getenv("ASYNC_ENABLE_D0", "1").strip().lower() in ("1", "true", "yes")
D1_SHARE = max(0.0, min(1.0, float(os.getenv("ASYNC_D1_SHARE", "0.80"))))
SIM_BATCH_TIMEOUT = max(30, int(os.getenv("ASYNC_SIM_BATCH_TIMEOUT", "400")))
LOCAL_BT_ENABLED = os.getenv("ASYNC_LOCAL_BT", "0").strip().lower() in ("1", "true", "yes")
LOCAL_BT_MIN_SCORE = float(os.getenv("ASYNC_LOCAL_BT_MIN_SCORE", "45.0"))
LOCAL_BT_MIN_SHARPE = float(os.getenv("ASYNC_LOCAL_BT_MIN_SHARPE", "0.20"))
RUN_ID = os.getenv("RUN_ID", "")


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("alpha_async_factory.log")
    ]
)
logger = logging.getLogger("AsyncPipeline")


def safe_expr_ref(expression: str) -> str:
    """Return a non-reversible expression reference for logs."""
    return hashlib.sha256((expression or "").strip().encode("utf-8")).hexdigest()[:12]


def _quick_similarity_score(expr: str, recent: list[str], threshold: float = 0.75) -> float:
    """
    Fast Jaccard token overlap check against a list of recent expressions.
    Returns the highest overlap ratio found (0.0 = no similarity).
    """
    tokens = set(re.findall(r'[a-zA-Z_]+', expr.lower()))
    if not tokens:
        return 0.0
    best = 0.0
    for prev in recent:
        prev_tokens = set(re.findall(r'[a-zA-Z_]+', prev.lower()))
        if not prev_tokens:
            continue
        overlap = len(tokens & prev_tokens) / max(len(tokens | prev_tokens), 1)
        if overlap >= threshold:
            best = max(best, overlap)
    return best


class AsyncAlphaFactory:
    def __init__(
        self,
        candidates_target=500,
        pre_rank_score=PRE_RANK_THRESHOLD,
        tracker=None,
        client=None,
        generator=None,
        governor=None,
    ):
        self.target = candidates_target
        self.pre_rank_score = pre_rank_score
        self.base_pre_rank_score = float(pre_rank_score)
        self.dynamic_pre_rank_score = float(pre_rank_score)
        
        # Use PriorityQueue for sim_queue to ensure top-ranked candidates are simulated first.
        # Format: (priority, timestamp, AlphaCandidate). Lower priority value = higher priority.
        self.gen_queue = asyncio.Queue(maxsize=GEN_QUEUE_SIZE)
        self.sim_queue = asyncio.PriorityQueue(maxsize=SIM_QUEUE_SIZE)
        
        self.tracker = tracker if tracker is not None else AlphaTracker()
        self.client = client if client is not None else WQClient()
        self.generator = generator if generator is not None else AlphaGenerator(
            generation_mode=os.getenv("GENERATOR_MODE", "legacy")
        )
        self.governor = governor if governor is not None else SubmitGovernor(self.tracker, self.client)
        self.qd_archive = QualityDiversityArchive(max_recent=NOVELTY_WINDOW)
        self.pending_signatures: set[str] = set()
        self.allocator = BudgetAllocator(
            tier1_min_quality=TIER1_MIN_QUALITY,
            tier1_min_novelty=NOVELTY_MIN,
            min_expected_value=TIER2_MIN_EV,
        )

        # Global Simulation State
        self.simulation_pause_until = 0
        self._pause_lock = asyncio.Lock()
        self._local_bt = None
        self._local_bt_disabled = not LOCAL_BT_ENABLED
        self._last_gate_profile = ""
        self._batch_recent: list[str] = []  # rolling in-batch expression cache for similarity check

        for descriptor, expr, quality, novelty in self.tracker.load_qd_archive(limit=1800):
            self.qd_archive.restore_elite(descriptor, expr, quality, novelty)

        self.is_running = True
        self.stats = {
            "generated": 0,
            "filtered": 0,
            "simulated": 0,
            "simulated_d1": 0,
            "simulated_d0": 0,
            "passed": 0,
            "passed_d1": 0,
            "passed_d0": 0,
            "errors": 0,
            "queued": 0,
            "submitted": 0,
            "dead_lettered": 0,
            "rejected": 0,
            "gate_rejections": 0,
            "archive_updates": 0,
            "accepted_review": 0,
            "rejected_review": 0,
            "active_simulations": 0,
        }

    @staticmethod
    def _clamp(value: float, lower: float, upper: float) -> float:
        return max(lower, min(upper, float(value)))

    def adapt_runtime_gates(self, kpi: dict | None = None) -> str:
        """
        Nudge pre-simulation gates using live throughput and accepted/rejected truth.
        """
        if not ADAPTIVE_GATES:
            return "disabled"

        kpi = kpi or {}
        reason = "hold"
        sim_fill = self.sim_queue.qsize() / max(1, SIM_QUEUE_SIZE)
        generated = int(self.stats.get("generated", 0) or 0)
        simulated = int(self.stats.get("simulated", 0) or 0)
        resolved = int(kpi.get("accepted", 0) or 0) + int(kpi.get("rejected_after_submit", 0) or 0)
        true_accept_rate = float(kpi.get("true_accept_rate", 0.0) or 0.0)
        dlq_rate = float(kpi.get("dlq_rate", 0.0) or 0.0)
        queued = int(kpi.get("queued", 0) or 0)

        pre_rank_delta = quality_delta = novelty_delta = ev_delta = 0.0
        if queued >= 5 and dlq_rate >= 0.20:
            reason = "tighten_dlq"
            ev_delta = 0.02
        elif resolved >= 10 and true_accept_rate < 0.18:
            reason = "tighten_accept"
            pre_rank_delta = 1.5
            quality_delta = 0.02
            novelty_delta = 0.01
            ev_delta = 0.015
        elif sim_fill < 0.10 and generated >= max(40, GEN_BATCH_SIZE * 2):
            reason = "relax_starved"
            pre_rank_delta = -2.0
            quality_delta = -0.015
            novelty_delta = -0.015
            ev_delta = -0.01
        elif sim_fill > 0.75 and simulated >= max(5, SIMULATOR_WORKERS * 2):
            reason = "tighten_backlog"
            pre_rank_delta = 1.0
            quality_delta = 0.005

        self.dynamic_pre_rank_score = self._clamp(
            self.dynamic_pre_rank_score + pre_rank_delta,
            ADAPTIVE_MIN_PRE_RANK,
            ADAPTIVE_MAX_PRE_RANK,
        )
        self.allocator.tier1_min_quality = self._clamp(
            self.allocator.tier1_min_quality + quality_delta,
            ADAPTIVE_MIN_QUALITY,
            ADAPTIVE_MAX_QUALITY,
        )
        self.allocator.tier1_min_novelty = self._clamp(
            self.allocator.tier1_min_novelty + novelty_delta,
            ADAPTIVE_MIN_NOVELTY,
            ADAPTIVE_MAX_NOVELTY,
        )
        self.allocator.min_expected_value = self._clamp(
            self.allocator.min_expected_value + ev_delta,
            ADAPTIVE_MIN_EV,
            ADAPTIVE_MAX_EV,
        )

        profile = (
            f"{reason}:rank={self.dynamic_pre_rank_score:.1f} "
            f"q={self.allocator.tier1_min_quality:.2f} "
            f"nov={self.allocator.tier1_min_novelty:.2f} "
            f"ev={self.allocator.min_expected_value:.2f}"
        )
        if profile != self._last_gate_profile:
            logger.info("[AdaptiveGate] %s", profile)
            self._last_gate_profile = profile
        return reason

    def _arm_name(self, cand: AlphaCandidate) -> str:
        theme = getattr(cand, "theme", "unknown") or "unknown"
        mutation = getattr(cand, "mutation_type", "seed") or "seed"
        return f"{theme}:{mutation}"

    @staticmethod
    def _clamp01(value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    def _simulation_reward(self, res) -> float:
        """
        Continuous allocator reward from WQ simulation quality.

        This avoids treating near-miss positive Sharpe results the same as
        broken or strongly negative simulations, so the arm scheduler can learn
        from signal strength before final submit acceptance is available.
        """
        if getattr(res, "error", ""):
            return 0.0

        sharpe = float(getattr(res, "sharpe", 0.0) or 0.0)
        fitness = float(getattr(res, "fitness", 0.0) or 0.0)
        turnover = float(getattr(res, "turnover", 0.0) or 0.0)
        drawdown = abs(float(getattr(res, "drawdown", 0.0) or 0.0))
        sub_sharpe = float(getattr(res, "sub_sharpe", -1.0) or -1.0)

        sharpe_score = self._clamp01((sharpe + 0.5) / 2.5)
        fitness_score = self._clamp01((fitness + 0.2) / 1.8)
        if turnover <= 1.0 or turnover >= 85.0:
            turnover_score = 0.0
        elif 5.0 <= turnover <= 55.0:
            turnover_score = 1.0
        else:
            turnover_score = 0.55
        drawdown_score = self._clamp01(1.0 - (drawdown / 0.20))
        sub_score = 0.5 if sub_sharpe <= -0.99 else self._clamp01((sub_sharpe + 0.50) / 1.50)

        reward = (
            0.42 * sharpe_score
            + 0.28 * fitness_score
            + 0.14 * turnover_score
            + 0.10 * drawdown_score
            + 0.06 * sub_score
        )
        return self._clamp01(reward)

    def _passes_local_backtest(self, cand: AlphaCandidate) -> Tuple[bool, str]:
        if self._local_bt_disabled:
            return True, "local_bt_disabled"
        try:
            if self._local_bt is None:
                from local_backtest import LocalBacktester
                self._local_bt = LocalBacktester()
            res = self._local_bt.backtest_single(cand.expression)
        except Exception as exc:
            self._local_bt_disabled = True
            logger.warning("Local backtest disabled after initialization/eval failure: %s", exc)
            return True, "local_bt_unavailable"

        if getattr(res, "error", ""):
            return False, f"local_bt_error:{str(res.error)[:40]}"
        if float(getattr(res, "score", 0.0) or 0.0) < LOCAL_BT_MIN_SCORE:
            return False, "local_bt_low_score"
        if float(getattr(res, "sharpe", 0.0) or 0.0) < LOCAL_BT_MIN_SHARPE:
            return False, "local_bt_low_sharpe"
        return True, "local_bt_pass"

    def _assign_delay_lane(self, cand: AlphaCandidate) -> None:
        if not ENABLE_D0:
            cand.delay = 1
            return
        cand.delay = 1 if random.random() < D1_SHARE else 0

    async def should_accept_candidate(self, cand: AlphaCandidate) -> Tuple[bool, str, float]:
        """
        Fast gate for ranker worker. Returns (is_accepted, reason, score).
        """
        signature = parameter_agnostic_signature(cand.expression)
        if signature in self.pending_signatures or signature in self.qd_archive.signature_seen:
            return False, "duplicate_signature", 0.0
        if self.tracker.is_duplicate(cand.expression):
            return False, "duplicate_db", 0.0
        if self.tracker.is_collinear(cand.expression):
            return False, "collinear", 0.0

        # ---- Self-corr risk pre-filter ----
        risk = estimate_self_corr_risk(cand.expression)
        max_risk = float(os.getenv("ASYNC_MAX_SELF_CORR_RISK", "0.80"))
        if risk > max_risk:
            return False, f"self_corr_risk:{risk:.2f}", 0.0

        # ---- In-batch similarity penalty ----
        # Reject near-duplicates within the current batch (token Jaccard >= threshold)
        if self._batch_recent:
            sim = _quick_similarity_score(cand.expression, self._batch_recent)
            if sim > 0:
                return False, f"batch_similar:{sim:.0%}", 0.0

        if not should_simulate_candidate(cand.expression, min_critic_score=MIN_CRITIC_SCORE):
            return False, "low_critic_score", 0.0
        if not passes_complexity_check(cand.expression):
            return False, "complexity_check_failed", 0.0
        score, _ = score_expression(cand.expression)
        if score < self.dynamic_pre_rank_score:
            return False, "low_score", 0.0

        # ── v2 Gate: Bias Detection + Meta-Model Scoring ──────────────────────
        # These run BEFORE simulation to catch disqualifying patterns early.
        if _NEW_MODULES_OK:
            bias = BiasDetector()
            bias_passed, bias_reason = bias.passes_bias_check(cand.expression)
            if not bias_passed:
                return False, f"bias:{bias_reason}", 0.0

            # Meta-model composite score (expected Sharpe, decay risk, simplicity)
            meta = score_with_meta_model(cand.expression)
            # Penalise high decay risk or very complex expressions pre-sim
            if meta['decay_probability'] > 0.85:
                return False, "high_decay_risk", meta['composite_score']
            if meta['simplicity_score'] < 0.15:
                return False, "too_complex_pre_sim", meta['composite_score']

        local_ok, local_reason = self._passes_local_backtest(cand)
        if not local_ok:
            return False, local_reason, 0.0

        quality_norm = self.allocator.normalize_quality(score)
        novelty, _ = self.qd_archive.novelty_score(cand.expression)
        pass_tier1, tier1_reason = self.allocator.tier1_accept(quality_norm, novelty)
        if not pass_tier1:
            return False, tier1_reason, 0.0

        arm = self._arm_name(cand)
        pass_tier2, ev = self.allocator.tier2_accept(arm, quality_norm, novelty)
        if not pass_tier2:
            return False, f"tier2_low_ev:{ev:.2f}", 0.0
        return True, "accepted", float(score)

    async def producer_generator(self):
        """Task: Continuously generate alpha expressions"""
        logger.info("🚀 [Producer] Generation started.")
        loop = asyncio.get_running_loop()
        while self.is_running:
            try:
                if self.gen_queue.qsize() > GEN_QUEUE_SIZE * 0.75 or self.sim_queue.qsize() > SIM_QUEUE_SIZE * 0.75:
                    await asyncio.sleep(2)
                    continue
                batch = await loop.run_in_executor(
                    None,
                    lambda: self.generator.generate_batch(n=GEN_BATCH_SIZE, use_rag=ASYNC_USE_RAG),
                )
                for cand in batch:
                    if not self.is_running: break
                    self._assign_delay_lane(cand)
                    await self.gen_queue.put(cand)
                    self.stats["generated"] += 1
            except Exception as e:
                logger.error(f"❌ [Producer] Error: {e}")
                await asyncio.sleep(10)

    async def worker_ranker(self, worker_id):
        """Task: Filter and score expressions before simulation"""
        logger.info(f"🔍 [Ranker-{worker_id}] Filter worker ready.")
        loop = asyncio.get_running_loop()
        while self.is_running:
            cand: AlphaCandidate = await self.gen_queue.get()
            try:
                accepted, reason, score = await self.should_accept_candidate(cand)
                if not accepted:
                    self.stats["filtered"] += 1
                    if self.stats["filtered"] % 1000 == 0:
                        logger.info("Ranker high filter rate: %s rejected. Last reason: %s | expr_id=%s", self.stats["filtered"], reason, safe_expr_ref(cand.expression))
                    continue

                # 3. Pass to Priority Simulation Queue
                # Priority = 100 - score (so higher score gets lower value -> higher priority)
                priority = max(0, int(100 - score))
                self.pending_signatures.add(parameter_agnostic_signature(cand.expression))
                # Track in batch cache for similarity dedup
                self._batch_recent.append(cand.expression)
                if len(self._batch_recent) > 200:
                    self._batch_recent = self._batch_recent[-200:]
                # Push a tuple to PriorityQueue
                await self.sim_queue.put((priority, time.time(), cand))
            except Exception as e:
                logger.error(f"❌ [Ranker-{worker_id}] Error: {e}")
            finally:
                try:
                    self.gen_queue.task_done()
                except ValueError:
                    pass

    async def worker_simulator(self, worker_id: int):
        """Task: Pure Async Streaming Simulator Worker"""
        # Staggered start to avoid hitting WQ API burst limits all at once
        await asyncio.sleep(worker_id * 0.5)
        logger.info("⚡ [Simulator-%s] Independent streaming worker started.", worker_id)
        loop = asyncio.get_running_loop()

        while self.is_running:
            cand = None
            try:
                # 1. Fetch next highest priority task
                # Use wait_for to check is_running periodically
                try:
                    priority, ts, cand = await asyncio.wait_for(self.sim_queue.get(), timeout=2.0)
                except asyncio.TimeoutError:
                    continue
                
                # Check for global backoff/pause
                if time.time() < self.simulation_pause_until:
                    wait_time = self.simulation_pause_until - time.time()
                    await asyncio.sleep(wait_time)

                logger.info("[Simulator-%s] Processing priority=%s | expr_id=%s", worker_id, priority, safe_expr_ref(cand.expression))
                self.stats["active_simulations"] += 1
                delay = int(getattr(cand, "delay", 1) or 1)
                
                # run block-prone simulation in executor
                res = await asyncio.wait_for(
                    loop.run_in_executor(
                        None,
                        lambda: self.client.simulate(
                            cand.expression,
                            delay=delay
                        )
                    ),
                    timeout=SIM_BATCH_TIMEOUT
                )

                # 2. Handle Result Immediately
                await self._handle_simulation_result(res, cand)

                # 3. Adaptive Backoff: If we hit a concurrency limit, pause the whole factory
                if res.error and ("rate_limited" in res.error.lower() or "CONCURRENT_SIMULATION_LIMIT" in res.error.upper()):
                    async with self._pause_lock:
                        if time.time() > self.simulation_pause_until:
                            self.simulation_pause_until = time.time() + 60
                            logger.warning(f"🐢 [Backoff] WQ limit hit by W-{worker_id}. Pausing ALL simulations for 60s.")

            except asyncio.TimeoutError:
                logger.warning(f"⏰ [Simulator-{worker_id}] Global timeout or task wait exceeded.")
                self.stats["errors"] += 1
            except Exception as e:
                logger.error(f"⚠️ [Simulator-{worker_id}] Critical failure: {e}")
                self.stats["errors"] += 1
            finally:
                if cand is not None:
                    try:
                        self.stats["active_simulations"] = max(0, int(self.stats.get("active_simulations", 0) or 0) - 1)
                        self.pending_signatures.discard(parameter_agnostic_signature(cand.expression))
                        self.sim_queue.task_done()
                    except ValueError:
                        pass
                    cand = None

                # Optional: slight heart-beat sleep
                await asyncio.sleep(0.1)

    async def _handle_simulation_result(self, res, cand: AlphaCandidate):
        """Streaming Result Handler - Updates stats, DB, and Evolution Archive"""
        self.stats["simulated"] += 1
        if getattr(res, "delay", 1) == 1:
            self.stats["simulated_d1"] += 1
        else:
            self.stats["simulated_d0"] += 1

        # Save to DB
        row_id = self.tracker.save_result(res, candidate=cand, run_id=RUN_ID)

        arm = self._arm_name(cand)

        # Calculate scores for DNA/Ranker feedback
        quality_score, _ = score_expression(res.expression)
        quality_norm = self.allocator.normalize_quality(quality_score)
        novelty, descriptor = self.qd_archive.novelty_score(res.expression)

        if res.error:
            self.tracker.mark_rejected_by_id(row_id, reason=f"error:{res.error[:50]}")
            self.stats["rejected"] += 1
            self.allocator.update(arm, 0.0) # Penalty for crashing
            return

        # Tier-1 pre-submission gate
        reward = self._simulation_reward(res)
        gate = pre_submission_gate_from_result(res)
        if not gate["passed"]:
            self.tracker.mark_rejected_by_id(row_id, reason=f"gate:{gate['stage']}:{gate['reason']}")
            self.stats["rejected"] += 1
            self.stats["gate_rejections"] += 1
            self.allocator.update(arm, reward)
            logger.info("[GATE] D%s rejected=%s | expr_id=%s", res.delay, gate["stage"], safe_expr_ref(res.expression))
            return

        if passes_quality_gate_v2(res):
            self.tracker.mark_gated_by_id(row_id)
            logger.info("[SUCCESS] D%s Sharpe=%.2f | expr_id=%s", res.delay, res.sharpe, safe_expr_ref(res.expression))
            self.stats["passed"] += 1
            if res.delay == 1:
                self.stats["passed_d1"] += 1
            else:
                self.stats["passed_d0"] += 1

            self.allocator.update(arm, max(reward, 0.85)) # Reward successful discovery

            # High-performance feedback loop: Update Evolution Archive
            q_val = (0.70 * float(res.sharpe or 0.0) + 0.30 * float(res.fitness or 0.0))
            if self.qd_archive.maybe_update_archive(res.expression, quality=q_val, novelty=novelty, descriptor=descriptor):
                self.stats["archive_updates"] += 1
                self.tracker.upsert_qd_archive(descriptor=descriptor, expression=res.expression, quality_score=q_val, novelty_score=novelty)

            # Push to Submission Governor immediately
            res.competition_priority = estimate_competition_priority(res)
            queued = self.governor.enqueue([res], candidates_map={res.expression: cand})
            if queued:
                self.stats["queued"] += queued
                # Flush to WQ Brain - Review Submission
                flush = self.governor.flush_once(limit=1)
                self.stats["submitted"] += int(flush.get("submitted", 0))
                self.stats["dead_lettered"] += int(flush.get("dead_lettered", 0))
        else:
            self.tracker.mark_rejected_by_id(row_id, reason="quality_gate_failed")
            self.stats["rejected"] += 1
            self.allocator.update(arm, reward)

        # Periodically sync review status (Opportunistic)
        if self.stats["simulated"] % 10 == 0:
            review_sync = self.governor.reconcile_submitted(limit=5)
            self.stats["accepted_review"] += int(review_sync.get("accepted", 0))
            self.stats["rejected_review"] += int(review_sync.get("rejected", 0))

        # Acceptance-attributed feedback loop:
        # Every 50 simulations, re-read WQ true acceptance rates per arm
        # and update the allocator's EV priors so the budget shifts
        # toward generator strategies with real WQ acceptance signal.
        if self.stats["simulated"] % 50 == 0 and self.stats["simulated"] > 0:
            try:
                acceptance_rates = self.tracker.acceptance_rate_by_arm()
                if acceptance_rates:
                    self.allocator.update_acceptance_priors(acceptance_rates)
                    logger.debug(
                        "♻️ [Allocator] Updated priors for %d arms: %s",
                        len(acceptance_rates),
                        {arm: f"{v['p_accept']:.2f}({v['resolved']})" for arm, v in acceptance_rates.items()},
                    )
            except Exception as e:
                logger.warning("⚠️ [Allocator] Failed to update acceptance priors: %s", e)

            # Also refresh self-corr learned weights every 50 sims
            try:
                update_self_corr_weights()
            except Exception as e:
                logger.debug("Self-corr weights update skipped: %s", e)


        if self.target > 0 and self.stats["simulated"] >= self.target:
            self.is_running = False

    async def status_monitor(self):
        """Task: Print status updates periodically"""
        start_time = time.time()
        while self.is_running:
            elapsed = time.time() - start_time
            throughput = (self.stats["simulated"] / (elapsed / 3600)) if elapsed > 0 else 0
            kpi = self.tracker.minute_kpis(lookback_minutes=60)
            self.adapt_runtime_gates(kpi)
            qd_stats = self.qd_archive.stats()
            
            logger.info(
                f"📊 [Monitor] Sim: {self.stats['simulated']} | Passed: {self.stats['passed']} | "
                f"Filtered: {self.stats['filtered']} | ActiveSim: {self.stats['active_simulations']} | "
                f"D1Sim: {self.stats['simulated_d1']} | D0Sim: {self.stats['simulated_d0']} | "
                f"D1Pass: {self.stats['passed_d1']} | D0Pass: {self.stats['passed_d0']} | "
                f"Queued: {self.stats['queued']} | Submitted: {self.stats['submitted']} | "
                f"Accepted: {self.stats['accepted_review']} | ReviewReject: {self.stats['rejected_review']} | "
                f"Rejected: {self.stats['rejected']} (gate:{self.stats['gate_rejections']}) | DLQ: {self.stats['dead_lettered']} | "
                f"QD cells: {qd_stats['elite_cells']} (+{self.stats['archive_updates']}) | "
                f"Rate: {throughput:.1f} alpha/h | gate={kpi['gate_pass_rate']:.1%} "
                f"submit={kpi['submit_success_rate']:.1%} submit_ok={kpi['submit_ok_rate']:.1%} "
                f"accept={kpi['true_accept_rate']:.1%} dlq={kpi['dlq_rate']:.1%} "
                f"rank_min={self.dynamic_pre_rank_score:.1f} ev_min={self.allocator.min_expected_value:.2f} "
                f"| Q1: {self.gen_queue.qsize()} | Q2: {self.sim_queue.qsize()}"
            )
            if self.target > 0 and self.stats["simulated"] >= self.target:
                break
            await asyncio.sleep(30)

    async def run(self):
        # Handle SIGINT/SIGTERM
        try:
            loop = asyncio.get_running_loop()
            for s in (signal.SIGINT, signal.SIGTERM):
                loop.add_signal_handler(s, lambda: asyncio.create_task(self.shutdown()))
        except NotImplementedError:
            # Signal handlers not supported on some Windows event loops
            pass

        workers = [
            asyncio.create_task(self.producer_generator()),
            asyncio.create_task(self.status_monitor()),
        ]

        for i in range(SIMULATOR_WORKERS):
            workers.append(asyncio.create_task(self.worker_simulator(i)))
        
        for i in range(RANKER_WORKERS):
            workers.append(asyncio.create_task(self.worker_ranker(i)))

        try:
            await asyncio.gather(*workers)
        except asyncio.CancelledError:
            logger.info("🛑 Pipeline shutting down...")
        finally:
            await self.shutdown()

    async def shutdown(self):
        if not self.is_running and self.stats["simulated"] > 0:
             logger.info("🚦 Work completed.")
        else:
             logger.info("🚦 Shutdown signal received.")
        
        self.is_running = False
        self.tracker.close()
        # Cancel all tasks
        current = asyncio.current_task()
        tasks = [t for t in asyncio.all_tasks() if t is not current]
        for t in tasks: t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        logger.info("✅ Pipeline halted.")

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="Stop after N simulations")
    parser.add_argument("--score", type=float, default=PRE_RANK_THRESHOLD, help="Min pre-rank score")
    args = parser.parse_args()

    factory = AsyncAlphaFactory(candidates_target=args.limit, pre_rank_score=args.score)
    try:
        asyncio.run(factory.run())
    except (KeyboardInterrupt, SystemExit):
        pass
