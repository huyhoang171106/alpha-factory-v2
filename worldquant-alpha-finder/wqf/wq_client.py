"""wqf/wq_client.py — Minimal WQ Brain API client using httpx."""
import os
import time
import logging
import httpx
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)
API_BASE = "https://api.worldquantbrain.com"


class WQClient:
    def __init__(self):
        self.email = os.getenv("WQ_EMAIL", "")
        self.password = os.getenv("WQ_PASSWORD", "")
        if not self.email or not self.password:
            raise RuntimeError(
                "Set WQ_EMAIL and WQ_PASSWORD in .env\n"
                "  cp .env.example .env  →  edit .env"
            )
        self.token = self._auth()
        self.client = httpx.Client(
            base_url=API_BASE,
            headers={"Authorization": f"Bearer {self.token}"},
            timeout=60.0,
        )

    def _auth(self) -> str:
        resp = httpx.post(
            f"{API_BASE}/authentication",
            json={"email": self.email, "password": self.password},
        )
        resp.raise_for_status()
        return resp.json()["token"]

    def simulate(
        self,
        expression: str,
        region: str = "USA",
        universe: str = "TOP3000",
        delay: int = 1,
        decay: int = 6,
        neutralization: str = "SUBINDUSTRY",
    ) -> dict:
        payload = {
            "expression": expression,
            "settings": {
                "region": region,
                "universe": universe,
                "delay": delay,
                "decay": decay,
                "neutralization": {"target": neutralization},
            },
        }
        resp = self.client.post("/alphas/me/simulate", json=payload)
        if resp.status_code == 429:
            logger.warning("Rate limited — waiting 30s...")
            time.sleep(30)
            resp = self.client.post("/alphas/me/simulate", json=payload)
        resp.raise_for_status()
        data = resp.json()
        status_url = data.get("links", {}).get("self", "")
        for _ in range(60):
            r = self.client.get(status_url) if status_url else resp
            state = r.json().get("state", "")
            if state == "completed":
                return r.json().get("data", {})
            elif state in ("failed", "error"):
                return {"error": state}
            time.sleep(5)
        return {"error": "timeout"}

    def get_status(self) -> dict:
        try:
            r = self.client.get("/alphas/me")
            if r.status_code == 200:
                return r.json()
            return {"error": f"HTTP {r.status_code}"}
        except Exception as e:
            return {"error": str(e)}
