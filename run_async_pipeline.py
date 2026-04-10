#!/usr/bin/env python3
"""
run_async_pipeline.py — Professional High-Throughput Async Alpha Factory
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
import signal
import time
import os
from typing import Dict, Tuple

from generator import AlphaGenerator
from alpha_ranker import score_expression
from tracker import AlphaTracker
from wq_client import WQClient
from alpha_policy import passes_quality_gate
from alpha_candidate import AlphaCandidate
from alpha_ast import parameter_agnostic_signature
from submit_governor import SubmitGovernor
from quality_diversity import QualityDiversityArchive
from budget_allocator import BudgetAllocator

# Configuration
QUEUE_SIZE = 1000
RANKER_WORKERS = max(1, int(os.getenv("ASYNC_RANKER_WORKERS", "2")))
SIMULATOR_WORKERS = max(1, int(os.getenv("ASYNC_SIMULATOR_WORKERS", "1")))  # WQ API has global concurrency limits
BATCH_SIZE = 20        # Simulation batch size
PRE_RANK_THRESHOLD = 50.0
NOVELTY_WINDOW = 12000
NOVELTY_MIN = float(os.getenv("ASYNC_NOVELTY_MIN", "0.28"))
ASYNC_USE_RAG = os.getenv("ASYNC_USE_RAG", "0").strip().lower() in ("1", "true", "yes")
TIER1_MIN_QUALITY = float(os.getenv("ASYNC_TIER1_MIN_QUALITY", "0.50"))
TIER2_MIN_EV = float(os.getenv("ASYNC_TIER2_MIN_EV", "0.34"))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("alpha_async_factory.log")
    ]
)
logger = logging.getLogger("AsyncPipeline")

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
        
        self.gen_queue = asyncio.Queue(maxsize=QUEUE_SIZE)
        self.sim_queue = asyncio.Queue(maxsize=QUEUE_SIZE // 2)
        
        self.tracker = tracker if tracker is not None else AlphaTracker()
        self.client = client if client is not None else WQClient()
        self.generator = generator if generator is not None else AlphaGenerator()
        self.governor = governor if governor is not None else SubmitGovernor(self.tracker, self.client)
        self.qd_archive = QualityDiversityArchive(max_recent=NOVELTY_WINDOW)
        self.pending_signatures: set[str] = set()
        self.allocator = BudgetAllocator(
            tier1_min_quality=TIER1_MIN_QUALITY,
            tier1_min_novelty=NOVELTY_MIN,
            min_expected_value=TIER2_MIN_EV,
        )

        for descriptor, expr, quality, novelty in self.tracker.load_qd_archive(limit=1800):
            self.qd_archive.restore_elite(descriptor, expr, quality, novelty)

        self.is_running = True
        self.stats = {
            "generated": 0,
            "filtered": 0,
            "simulated": 0,
            "passed": 0,
            "errors": 0,
            "queued": 0,
            "submitted": 0,
            "dead_lettered": 0,
            "rejected": 0,
            "archive_updates": 0,
            "accepted_review": 0,
            "rejected_review": 0,
        }

    def _arm_name(self, cand: AlphaCandidate) -> str:
        theme = getattr(cand, "theme", "unknown") or "unknown"
        mutation = getattr(cand, "mutation_type", "seed") or "seed"
        return f"{theme}:{mutation}"

    def should_accept_candidate(self, cand: AlphaCandidate) -> Tuple[bool, str]:
        """
        Fast gate for ranker worker. Returns (is_accepted, reason).
        """
        signature = parameter_agnostic_signature(cand.expression)
        if signature in self.pending_signatures or signature in self.qd_archive.signature_seen:
            return False, "duplicate_signature"
        if self.tracker.is_duplicate(cand.expression):
            return False, "duplicate_db"
        if self.tracker.is_collinear(cand.expression):
            return False, "collinear"
        score, _ = score_expression(cand.expression)
        if score < self.pre_rank_score:  # cheap pre-ranker
            return False, "low_score"

        quality_norm = self.allocator.normalize_quality(score)
        novelty, _ = self.qd_archive.novelty_score(cand.expression)
        pass_tier1, tier1_reason = self.allocator.tier1_accept(quality_norm, novelty)
        if not pass_tier1:
            return False, tier1_reason

        arm = self._arm_name(cand)
        pass_tier2, ev = self.allocator.tier2_accept(arm, quality_norm, novelty)
        if not pass_tier2:
            return False, f"tier2_low_ev:{ev:.2f}"
        return True, "accepted"

    async def producer_generator(self):
        """Task: Continuously generate alpha expressions"""
        logger.info("🚀 [Producer] Generation started.")
        loop = asyncio.get_running_loop()
        while self.is_running:
            try:
                # Generate a small batch to keep the queue fresh
                # Run CPU-heavy generation off the event loop to keep workers responsive.
                batch = await loop.run_in_executor(
                    None,
                    lambda: self.generator.generate_batch(n=50, use_rag=ASYNC_USE_RAG),
                )
                for cand in batch:
                    if not self.is_running: break
                    await self.gen_queue.put(cand)
                    self.stats["generated"] += 1
                
                # Dynamic pacing
                if self.gen_queue.qsize() > QUEUE_SIZE * 0.8:
                    await asyncio.sleep(2)
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
                # run CPU/database-heavy gate out of event loop
                accepted, reason = await loop.run_in_executor(None, self.should_accept_candidate, cand)
                if not accepted:
                    self.stats["filtered"] += 1
                    continue

                # 3. Pass to Simulator
                self.pending_signatures.add(parameter_agnostic_signature(cand.expression))
                await self.sim_queue.put(cand)
            except Exception as e:
                logger.error(f"❌ [Ranker-{worker_id}] Error: {e}")
            finally:
                self.gen_queue.task_done()

    async def worker_simulator(self, worker_id: int):
        """Task: Consume candidates and run WQ Simulations"""
        logger.info("⚡ [Simulator-%s] API simulation worker started.", worker_id)
        cooldown = 1
        while self.is_running:
            batch = []
            try:
                while len(batch) < BATCH_SIZE:
                    try:
                        cand = await asyncio.wait_for(self.sim_queue.get(), timeout=2.0)
                        batch.append(cand)
                    except asyncio.TimeoutError:
                        if batch: break
                        continue
                
                if not batch: continue

                logger.info(f"📡 [Simulator-{worker_id}] Simulating batch of {len(batch)} alphas...")
                batch_map: Dict[str, AlphaCandidate] = {}
                for cand in batch:
                    # keep first candidate for duplicated expressions
                    batch_map.setdefault(cand.expression, cand)
                
                loop = asyncio.get_event_loop()
                results = await loop.run_in_executor(
                    None, 
                    self.client.simulate_batch, 
                    [b.expression for b in batch]
                )

                # Adaptive rate limiting based on client stats
                if getattr(self.client, "last_batch_stats", {}).get("rate_limited", 0) > 0:
                    cooldown = min(cooldown * 2, 60)
                    logger.warning(f"🐢 [RateLimit] 429 detected. Increasing cooldown to {cooldown}s")
                else:
                    cooldown = max(1, cooldown // 2)

                passed_results = []
                for res in results:
                    cand = batch_map.get(res.expression)
                    if cand is None:
                        cand = AlphaCandidate(expression=res.expression, theme="unknown", mutation_type="async_fallback")
                    row_id = self.tracker.save_result(res, candidate=cand)
                    self.stats["simulated"] += 1
                    arm = self._arm_name(cand)
                    quality_score, _ = score_expression(res.expression)
                    quality_norm = self.allocator.normalize_quality(quality_score)
                    novelty, descriptor = self.qd_archive.novelty_score(res.expression)
                    self.pending_signatures.discard(parameter_agnostic_signature(res.expression))
                    
                    if passes_quality_gate(res):
                        self.tracker.mark_gated_by_id(row_id)
                        logger.info(f"💎 [SUCCESS] Sharpe={res.sharpe:.2f} | {res.expression[:60]}")
                        self.stats["passed"] += 1
                        self.allocator.update(arm, 1.0)
                        if self.qd_archive.maybe_update_archive(
                            res.expression,
                            quality=(0.70 * float(res.sharpe or 0.0) + 0.30 * float(res.fitness or 0.0)),
                            novelty=novelty,
                            descriptor=descriptor,
                        ):
                            self.stats["archive_updates"] += 1
                            self.tracker.upsert_qd_archive(
                                descriptor=descriptor,
                                expression=res.expression,
                                quality_score=(0.70 * float(res.sharpe or 0.0) + 0.30 * float(res.fitness or 0.0)),
                                novelty_score=novelty,
                            )
                        passed_results.append(res)
                    else:
                        self.tracker.mark_rejected_by_id(row_id, reason=getattr(res, "error", "") or "quality_gate_failed")
                        self.stats["rejected"] += 1
                        self.allocator.update(arm, 0.0)
                        # still register explored behavior to avoid repeating known bad shapes
                        self.qd_archive.maybe_update_archive(
                            res.expression,
                            quality=(0.40 * quality_norm),
                            novelty=novelty,
                            descriptor=descriptor,
                        )

                if passed_results:
                    queued = self.governor.enqueue(passed_results, candidates_map=batch_map)
                    if queued:
                        self.stats["queued"] += queued
                        flush = self.governor.flush_once(limit=min(queued, 4))
                        self.stats["submitted"] += int(flush.get("submitted", 0))
                        self.stats["dead_lettered"] += int(flush.get("dead_lettered", 0))
                review_sync = self.governor.reconcile_submitted(limit=10)
                self.stats["accepted_review"] += int(review_sync.get("accepted", 0))
                self.stats["rejected_review"] += int(review_sync.get("rejected", 0))
                
                if self.stats["simulated"] >= self.target and self.target > 0:
                    logger.info(f"🎯 Target reached: {self.stats['simulated']} alphas simulated.")
                    self.is_running = False

                await asyncio.sleep(cooldown)

            except Exception as e:
                logger.error(f"⚠️ [Simulator-{worker_id}] Batch failed: {e}")
                self.stats["errors"] += 1
                await asyncio.sleep(30)
            finally:
                for _ in range(len(batch)):
                    self.sim_queue.task_done()

    async def status_monitor(self):
        """Task: Print status updates periodically"""
        start_time = time.time()
        while self.is_running:
            elapsed = time.time() - start_time
            throughput = (self.stats["simulated"] / (elapsed / 3600)) if elapsed > 0 else 0
            kpi = self.tracker.minute_kpis(lookback_minutes=60)
            qd_stats = self.qd_archive.stats()
            
            logger.info(
                f"📊 [Monitor] Sim: {self.stats['simulated']} | Passed: {self.stats['passed']} | "
                f"Queued: {self.stats['queued']} | Submitted: {self.stats['submitted']} | "
                f"Accepted: {self.stats['accepted_review']} | ReviewReject: {self.stats['rejected_review']} | "
                f"Rejected: {self.stats['rejected']} | DLQ: {self.stats['dead_lettered']} | "
                f"QD cells: {qd_stats['elite_cells']} (+{self.stats['archive_updates']}) | "
                f"Rate: {throughput:.1f} alpha/h | gate={kpi['gate_pass_rate']:.1%} "
                f"submit={kpi['submit_success_rate']:.1%} submit_ok={kpi['submit_ok_rate']:.1%} "
                f"accept={kpi['true_accept_rate']:.1%} dlq={kpi['dlq_rate']:.1%} "
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
