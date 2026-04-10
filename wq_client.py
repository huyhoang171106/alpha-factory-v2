"""
wq_client.py - WorldQuant Brain API Client
Handles authentication, simulation, and submission of alpha expressions.
Reference: WQ-Brain/main.py + worldquant-miner/generation_two/core/simulator_tester.py
"""

import os
import time
import json
import logging
import requests
import threading
from datetime import datetime, timedelta
from typing import Optional
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

API_BASE = "https://api.worldquantbrain.com"
MAX_CONCURRENT = int(os.getenv("WQ_MAX_CONCURRENT", "4"))   # 4 concurrent sims for speed
POLL_INTERVAL = int(os.getenv("WQ_POLL_INTERVAL", "10"))    # 10s polling for faster results
MAX_WAIT_TIME = int(os.getenv("WQ_MAX_WAIT_TIME", "600"))   # 10 min max wait — WQ server can be slow
SIM_SUBMIT_RETRIES = max(0, int(os.getenv("WQ_SIM_SUBMIT_RETRIES", "3")))
SIM_SUBMIT_BACKOFF = max(3, int(os.getenv("WQ_SIM_SUBMIT_BACKOFF", "8")))
SIM_SUBMIT_FAIL_WINDOW = int(os.getenv("WQ_SIM_FAIL_WINDOW", "20"))
SIM_CIRCUIT_BREAKER_THRESHOLD = float(os.getenv("WQ_SIM_BREAKER_THRESHOLD", "0.80"))
SIM_CIRCUIT_BREAKER_COOLDOWN = int(os.getenv("WQ_SIM_BREAKER_COOLDOWN", "90"))


@dataclass
class SimResult:
    """Result from a WQ Brain simulation"""
    expression: str
    sharpe: float = 0.0
    fitness: float = 0.0
    turnover: float = 0.0
    returns: float = 0.0
    drawdown: float = 0.0
    passed_checks: int = 0
    total_checks: int = 0
    all_passed: bool = False
    alpha_id: str = ""
    alpha_url: str = ""
    error: str = ""
    sub_sharpe: float = -1.0
    # Settings used
    region: str = "USA"
    universe: str = "TOP3000"
    delay: int = 1
    decay: int = 6
    neutralization: str = "SUBINDUSTRY"

    @property
    def is_submittable(self) -> bool:
        """Check if alpha meets minimum submission criteria"""
        return (
            self.sharpe > 1.5
            and self.fitness >= 1.0
            and 1 < self.turnover < 70
            and not self.error
        )


class WQClient:
    """WorldQuant Brain API client with session management"""

    def __init__(self, email: str = None, password: str = None):
        self.email = email or os.getenv("WQ_EMAIL")
        self.password = password or os.getenv("WQ_PASSWORD")
        if not self.email or not self.password:
            raise ValueError("WQ_EMAIL and WQ_PASSWORD required in .env")

        self.session = requests.Session()
        self.last_batch_stats = {}
        self._auth_lock = threading.Lock()
        self.interactive_auth = os.getenv("WQ_INTERACTIVE_AUTH", "0").strip().lower() in ("1", "true", "yes")
        self._recent_submit_failures: list[bool] = []
        self._circuit_open_until: Optional[datetime] = None
        
        # Optimize connection pool for high concurrency
        adapter = requests.adapters.HTTPAdapter(pool_connections=100, pool_maxsize=100)
        self.session.mount('https://', adapter)
        self.session.mount('http://', adapter)
        
        self._login()

    def _login(self):
        """Authenticate with WQ Brain API"""
        logger.info(f"Logging in as {self.email}...")
        self.session.auth = (self.email, self.password)
        r = self.session.post(f"{API_BASE}/authentication")
        data = r.json()

        if 'user' not in data:
            if 'inquiry' in data:
                logger.warning("Biometric auth required! Complete it in browser first.")
                if self.interactive_auth:
                    input(f"Complete auth at {r.url}/persona?inquiry={data['inquiry']}, then press Enter...")
                    self.session.post(f"{r.url}/persona", json=data)
                else:
                    raise ConnectionError(
                        "Login requires biometric flow. Set WQ_INTERACTIVE_AUTH=1 for manual continuation."
                    )
            else:
                raise ConnectionError(f"Login failed: {data}")

        logger.info("OK: Logged in to WQ Brain!")

    def _api_request(self, method: str, url: str, max_retries: int = 5, **kwargs):
        """Make API request with auto re-login on 401 and rate-limit handling"""
        last_response = None
        for attempt in range(max_retries + 1):
            try:
                r = getattr(self.session, method)(url, **kwargs)
                last_response = r
                if r.status_code == 401:
                    logger.warning("Session expired, re-authenticating...")
                    with self._auth_lock:
                        self._login()
                    time.sleep(1)
                    continue
                if r.status_code == 429:
                    # Snappy backoff: 5, 10, 20, 30
                    wait = min(30, 5 * (2 ** attempt))
                    # logger.debug to mute terminal spam while keeping WQ Brain happy
                    logger.debug(f"⏳ Rate limited (429), waiting {wait}s... (attempt {attempt+1}/{max_retries})")
                    time.sleep(wait)
                    continue
                return r
            except requests.exceptions.RequestException as e:
                if attempt < max_retries:
                    wait = min(15, 3 * (attempt + 1))
                    logger.debug(f"Request failed, retry in {wait}s...")
                    time.sleep(wait)
                else:
                    raise
        return last_response

    @staticmethod
    def _classify_submit_error(response: Optional[requests.Response], exception: Optional[Exception] = None) -> str:
        if exception is not None:
            if isinstance(exception, requests.exceptions.Timeout):
                return "submit_network_timeout"
            if isinstance(exception, requests.exceptions.ConnectionError):
                return "submit_network_connection"
            return "submit_network_exception"
        if response is None:
            return "submit_no_response"
        code = response.status_code
        if code == 401:
            return "submit_auth_401"
        if code == 403:
            return "submit_auth_403"
        if code == 429:
            return "submit_rate_limited"
        if 500 <= code < 600:
            return "submit_server_5xx"
        if code >= 400:
            return "submit_payload_4xx"
        if "Location" not in response.headers:
            return "submit_missing_location"
        return ""

    def _record_submit_outcome(self, is_failure: bool):
        self._recent_submit_failures.append(is_failure)
        if len(self._recent_submit_failures) > SIM_SUBMIT_FAIL_WINDOW:
            self._recent_submit_failures = self._recent_submit_failures[-SIM_SUBMIT_FAIL_WINDOW:]
        if len(self._recent_submit_failures) >= 8:
            fail_ratio = sum(1 for v in self._recent_submit_failures if v) / len(self._recent_submit_failures)
            if fail_ratio >= SIM_CIRCUIT_BREAKER_THRESHOLD:
                self._circuit_open_until = datetime.utcnow() + timedelta(seconds=SIM_CIRCUIT_BREAKER_COOLDOWN)

    def _circuit_breaker_open(self) -> bool:
        if self._circuit_open_until is None:
            return False
        if datetime.utcnow() >= self._circuit_open_until:
            self._circuit_open_until = None
            return False
        return True

    @staticmethod
    def _is_concurrent_sim_limit(response: Optional[requests.Response]) -> bool:
        if response is None:
            return False
        try:
            payload = response.json()
            detail = str(payload.get("detail", "")).upper() if isinstance(payload, dict) else ""
        except Exception:
            detail = (response.text or "").upper()
        return "CONCURRENT_SIMULATION_LIMIT_EXCEEDED" in detail

    def simulate(
        self,
        expression: str,
        region: str = "USA",
        universe: str = "TOP3000",
        delay: int = 1,
        decay: int = 6,
        neutralization: str = "SUBINDUSTRY",
        truncation: float = 0.08,
    ) -> SimResult:
        """
        Submit an alpha expression for simulation and wait for results.
        
        Returns SimResult with metrics or error.
        """
        result = SimResult(
            expression=expression,
            region=region,
            universe=universe,
            delay=delay,
            decay=decay,
            neutralization=neutralization,
        )

        if self._circuit_breaker_open():
            result.error = "Simulation paused: circuit breaker open after repeated submit failures"
            return result

        # 1. Submit simulation
        payload = {
            "regular": expression,
            "type": "REGULAR",
            "settings": {
                "nanHandling": "OFF",
                "instrumentType": "EQUITY",
                "delay": delay,
                "universe": universe,
                "truncation": truncation,
                "unitHandling": "VERIFY",
                "pasteurization": "ON",
                "region": region,
                "language": "FASTEXPR",
                "decay": decay,
                "neutralization": neutralization,
                "visualization": False,
            }
        }

        r = None
        for attempt in range(SIM_SUBMIT_RETRIES + 1):
            try:
                r = self._api_request("post", f"{API_BASE}/simulations", json=payload)
                error_code = self._classify_submit_error(r)
                hit_sim_limit = self._is_concurrent_sim_limit(r)
                if (error_code == "submit_rate_limited" or hit_sim_limit) and attempt < SIM_SUBMIT_RETRIES:
                    wait = min(90, SIM_SUBMIT_BACKOFF * (2 ** attempt))
                    logger.warning(
                        "⏳ Simulation slot full/rate-limited, retry in %ss (attempt %s/%s)",
                        wait,
                        attempt + 1,
                        SIM_SUBMIT_RETRIES + 1,
                    )
                    time.sleep(wait)
                    continue
                if error_code:
                    try:
                        err_detail = r.json() if r is not None else {}
                    except Exception:
                        err_detail = (r.text if r is not None else "") or "(empty body)"
                    result.error = f"{error_code}: {err_detail}"
                    self._record_submit_outcome(True)
                    logger.error(f"❌ Submit failed for: {expression[:50]}... [{error_code}]")
                    return result
                break
            except Exception as e:
                if attempt < SIM_SUBMIT_RETRIES:
                    wait = min(60, SIM_SUBMIT_BACKOFF * (2 ** attempt))
                    time.sleep(wait)
                    continue
                error_code = self._classify_submit_error(None, e)
                result.error = f"{error_code}: {e}"
                self._record_submit_outcome(True)
                return result

        self._record_submit_outcome(False)

        progress_url = r.headers['Location']
        logger.info(f"📤 Submitted: {expression[:60]}... → polling...")

        # 2. Poll for completion
        start_time = time.time()
        while time.time() - start_time < MAX_WAIT_TIME:
            try:
                r = self._api_request("get", progress_url)
                if r is None:
                    result.error = "poll_no_response"
                    return result
                data = r.json()

                # Check for intermediate results to fail-fast
                is_data = data.get('is')
                if is_data:
                    self._parse_is_metrics(is_data, result)
                    # Fail-fast: if metrics are already below minimum thresholds, stop polling.
                    # We use 1.25/1.0 as standard "useful" floors. 
                    if result.sharpe < 1.0 or result.fitness < 0.6:
                        logger.info(f"  [FAIL-FAST] Sharpe={result.sharpe:.3f}, Fitness={result.fitness:.2f} (Incomplete)")
                        if 'alpha' in data:
                            result.alpha_id = data['alpha']
                            result.alpha_url = f"https://platform.worldquantbrain.com/alpha/{result.alpha_id}"
                        return result

                if 'alpha' in data:
                    # Simulation complete → fetch alpha details
                    alpha_id = data['alpha']
                    return self._fetch_alpha_details(alpha_id, result)

                # Still running
                progress = data.get('progress', 0)
                logger.debug(f"  [POLL] Progress: {int(progress * 100)}%")

            except Exception as e:
                logger.warning(f"  Poll error: {e}")

            time.sleep(POLL_INTERVAL)

        result.error = "Simulation timeout"
        logger.warning(f"⏰ Timeout for: {expression[:50]}...")
        return result

    def _parse_is_metrics(self, is_data: dict, result: SimResult):
        """Helper to extract IS metrics from alpha or simulation payload"""
        result.sharpe = is_data.get('sharpe', 0)
        result.fitness = is_data.get('fitness', 0)
        result.turnover = round(is_data.get('turnover', 0) * 100, 2)
        result.returns = is_data.get('returns', 0)
        result.drawdown = is_data.get('drawdown', 0)

        # Check pass/fail for specific checks if available
        checks = is_data.get('checks', [])
        if checks:
            result.total_checks = len(checks)
            result.passed_checks = sum(1 for c in checks if c.get('result') == 'PASS')
            result.all_passed = (result.passed_checks == result.total_checks)
            
            # Parse sub_universe_sharpe
            for check in checks:
                name = check.get('name', '').lower()
                if 'sub_universe' in name and 'sharpe' in name:
                    raw = check.get('value', None)
                    if raw is not None:
                        result.sub_sharpe = float(raw)
                        break

    def _fetch_alpha_details(self, alpha_id: str, result: SimResult) -> SimResult:
        """Fetch detailed results for a completed simulation"""
        r = self._api_request("get", f"{API_BASE}/alphas/{alpha_id}")
        data = r.json()

        result.alpha_id = alpha_id
        result.alpha_url = f"https://platform.worldquantbrain.com/alpha/{alpha_id}"

        try:
            is_data = data.get('is', {})
            self._parse_is_metrics(is_data, result)
            
            status = "[PASS]" if result.all_passed else f"[FAIL] {result.passed_checks}/{result.total_checks}"
            logger.info(
                f"  {status} | Sharpe: {result.sharpe:.3f} | "
                f"Fitness: {result.fitness:.2f} | Turnover: {result.turnover:.1f}% | "
                f"{result.alpha_url}"
            )

        except Exception as e:
            result.error = f"Parse error: {e}"
            logger.error(f"  ❌ Parse error: {e}")

        return result

    def simulate_batch(
        self,
        expressions: list[str],
        region: str = "USA",
        universe: str = "TOP3000",
        delay: int = 1,
        decay: int = 6,
        neutralization: str = "SUBINDUSTRY",
        max_workers: int = MAX_CONCURRENT,
        progress_callback=None,
    ) -> list[SimResult]:
        """
        Simulate multiple expressions using a true ThreadPoolExecutor.
        This safely scales parallel window frames independently of blocking 429 timeouts.
        """
        total = len(expressions)
        logger.info(f"🔄 True Threaded Batch simulating {total} expressions (max_concurrent={max_workers})...")

        results = []
        stats = {
            "total": total,
            "success": 0,
            "errors": 0,
            "timeouts": 0,
            "submit_failed": 0,
            "parse_failed": 0,
            "submittable": 0,
            "auth_failed": 0,
            "rate_limited": 0,
            "circuit_breaker": 0,
        }
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_expr = {
                executor.submit(
                    self.simulate, expr, region, universe, delay, decay, neutralization, 0.08
                ): expr for expr in expressions
            }
            
            for i, future in enumerate(as_completed(future_to_expr), 1):
                try:
                    res = future.result()
                    results.append(res)
                    if progress_callback:
                        try:
                            progress_callback(res, i, total)
                        except Exception:
                            pass
                    if res.error:
                        stats["errors"] += 1
                        err_l = (res.error or "").lower()
                        if "timeout" in err_l:
                            stats["timeouts"] += 1
                        if "submit_" in err_l or "submit failed" in err_l:
                            stats["submit_failed"] += 1
                        if "auth" in err_l:
                            stats["auth_failed"] += 1
                        if "rate_limited" in err_l:
                            stats["rate_limited"] += 1
                        if "circuit breaker" in err_l:
                            stats["circuit_breaker"] += 1
                        if "parse error" in err_l:
                            stats["parse_failed"] += 1
                        logger.warning(f"  [ERROR] [{i}/{total}] Failed: {res.expression[:50]}... Error: {res.error}")
                    else:
                        stats["success"] += 1
                        status = "[PASS]" if res.all_passed else f"[FAIL]"
                        logger.info(f"  {status} [{i}/{total}] {res.expression[:40]}... (Sharpe: {res.sharpe:.2f})")
                except Exception as exc:
                    logger.error(f"  ❌ Exception during threaded sim: {exc}")
                    
        passed_count = sum(1 for r in results if r.is_submittable)
        stats["submittable"] = passed_count
        self.last_batch_stats = stats
        logger.info(f"📊 Batch done: {passed_count}/{total} submittable")
        return results

    def submit_alpha(self, alpha_id: str) -> bool:
        """Submit a passing alpha for review"""
        try:
            r = self._api_request(
                "post",
                f"{API_BASE}/alphas/{alpha_id}/submit"
            )
            if r and r.status_code in (200, 201, 202):
                logger.info(f"🚀 Submitted alpha: {alpha_id}")
                return True
            else:
                logger.warning(f"Submit response: {r.status_code if r else 'None'}")
                return False
        except Exception as e:
            logger.error(f"Submit error: {e}")
            return False

    def submit_alpha_detailed(self, alpha_id: str) -> tuple[bool, str]:
        """
        Submit alpha and return (success, error_class) for retry policy.
        """
        try:
            r = self._api_request("post", f"{API_BASE}/alphas/{alpha_id}/submit")
            if r is None:
                return False, "submit_no_response"
            if r.status_code in (200, 201, 202):
                logger.info(f"🚀 Submitted alpha: {alpha_id}")
                return True, ""
            if r.status_code == 429:
                return False, "submit_rate_limited_429"
            if r.status_code in (401, 403):
                return False, f"submit_auth_{r.status_code}"
            if 500 <= r.status_code < 600:
                return False, f"submit_server_5xx_{r.status_code}"
            if 400 <= r.status_code < 500:
                return False, f"submit_semantic_4xx_{r.status_code}"
            return False, f"submit_http_{r.status_code}"
        except requests.exceptions.Timeout:
            return False, "submit_network_timeout"
        except requests.exceptions.ConnectionError:
            return False, "submit_network_connection"
        except Exception as e:
            logger.error(f"Submit error: {e}")
            return False, "submit_network_exception"

    @staticmethod
    def _extract_submission_state(data: dict) -> str:
        """
        Best-effort parser for post-submit review state.
        Returns one of: accepted, rejected, submitted, unknown.
        """
        if not isinstance(data, dict):
            return "unknown"

        candidate_values = []
        for key in (
            "submissionStatus",
            "submission_status",
            "submission",
            "status",
            "state",
            "reviewStatus",
            "review_status",
        ):
            if key in data:
                candidate_values.append(str(data.get(key, "")))

        for node_key in ("submission", "review", "status"):
            node = data.get(node_key)
            if isinstance(node, dict):
                for inner_key in ("status", "state", "result", "decision"):
                    if inner_key in node:
                        candidate_values.append(str(node.get(inner_key, "")))

        flattened = " ".join(candidate_values).lower()
        blob = json.dumps(data).lower()
        scan = f"{flattened} {blob}"

        if any(token in scan for token in ("accepted", "pass", "approved")):
            return "accepted"
        if any(token in scan for token in ("rejected", "fail", "denied")):
            return "rejected"
        if any(token in scan for token in ("submitted", "pending", "review", "in progress", "queued")):
            return "submitted"
        return "unknown"

    def get_submission_decision(self, alpha_id: str) -> tuple[str, str, str]:
        """
        Poll alpha details and infer review decision.
        Returns: (decision, error_class, detail)
        decision: accepted | rejected | submitted | unknown
        """
        try:
            r = self._api_request("get", f"{API_BASE}/alphas/{alpha_id}")
            if r is None:
                return "unknown", "submit_status_no_response", "no response"
            if r.status_code in (401, 403):
                return "unknown", f"submit_status_auth_{r.status_code}", ""
            if r.status_code == 429:
                return "submitted", "", "rate limited while polling"
            if r.status_code >= 500:
                return "unknown", f"submit_status_server_{r.status_code}", ""
            if r.status_code >= 400:
                return "unknown", f"submit_status_payload_{r.status_code}", ""
            data = r.json()
            state = self._extract_submission_state(data)
            return state, "", ""
        except requests.exceptions.Timeout:
            return "unknown", "submit_status_timeout", ""
        except requests.exceptions.ConnectionError:
            return "unknown", "submit_status_connection", ""
        except Exception as e:
            return "unknown", "submit_status_exception", str(e)[:200]


# ============================================================
# Quick test
# ============================================================
if __name__ == "__main__":
    client = WQClient()
    
    # Test with a simple alpha
    result = client.simulate("-rank(ts_delta(close, 5))")
    print(f"\nResult: Sharpe={result.sharpe}, Fitness={result.fitness}, "
          f"Turnover={result.turnover}%, Submittable={result.is_submittable}")
    print(f"URL: {result.alpha_url}")
