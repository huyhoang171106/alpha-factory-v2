import os
import sys
import json
import logging

# Ensure root directory is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from alpha_seeds import get_all_seeds
from community_harvester import CommunityHarvester

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    # 1. Create directory if not exists
    extract_path = "data/elite_alphas"
    if not os.path.exists(extract_path):
        os.makedirs(extract_path, exist_ok=True)
        logger.info(f"Created directory: {extract_path}")

    elite_seeds = []

    # 2. Harvest from CommunityHarvester (Local DB + API)
    logger.info("Harvesting from CommunityHarvester...")
    harvester = CommunityHarvester()
    # Note: harvest() tries DB first, then API. 
    # Since I might not have API access right now, it will likely only get from DB.
    harvested = harvester.harvest(min_sharpe=1.25)
    
    for h in harvested:
        elite_seeds.append({
            "expression": h.expression,
            "sharpe": h.sharpe,
            "fitness": h.fitness,
            "turnover": h.turnover,
            "source": h.source
        })
    
    logger.info(f"Harvested {len(harvested)} alphas from community/local.")

    # 3. Harvest from internal library alpha_seeds.py
    logger.info("Harvesting from alpha_seeds.py libraries...")
    internal_seeds = get_all_seeds()
    
    # We'll assign a baseline sharpe of 1.25 to internal seeds to ensure they are used by RAG
    # unless we already have them from the harvester with real stats.
    seen_expressions = {s['expression'] for s in elite_seeds}
    
    for expr in internal_seeds:
        if expr not in seen_expressions:
            elite_seeds.append({
                "expression": expr,
                "sharpe": 1.25, # Baseline for RAG usage
                "fitness": 1.0,
                "turnover": 50,
                "source": "internal_seeds"
            })
            seen_expressions.add(expr)
            
    logger.info(f"Total elite seeds combined: {len(elite_seeds)}")

    # 4. Save to JSON
    output_file = os.path.join(extract_path, "elite_seed.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(elite_seeds, f, indent=2)
    
    logger.info(f"Successfully saved {len(elite_seeds)} seeds to {output_file}")

if __name__ == "__main__":
    main()
