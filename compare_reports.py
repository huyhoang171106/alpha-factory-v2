import sqlite3
import pandas as pd
import argparse
import os

def compare_runs(run_id_a, run_id_b, db_path='alpha_results.db'):
    if not os.path.exists(db_path):
        print(f"❌ Database not found: {db_path}")
        return

    conn = sqlite3.connect(db_path)
    
    query = "SELECT sharpe, fitness, turnover, theme, mutation_type, expression, run_id FROM alphas WHERE run_id IN (?, ?)"
    df = pd.read_sql_query(query, conn, params=(run_id_a, run_id_b))
    conn.close()

    if df.empty:
        print(f"❌ No data found for run_ids: {run_id_a}, {run_id_b}")
        return

    print("====================================================")
    print("      A/B COMPARISON: NORMAL vs RAG")
    print("====================================================\n")

    report = []
    for rid in [run_id_a, run_id_b]:
        sub = df[df['run_id'] == rid]
        if sub.empty:
            report.append({'run_id': rid, 'count': 0, 'gate_pass': 0, 'avg_sharpe': 0, 'unique_themes': 0})
            continue

        gate_pass = len(sub[sub['sharpe'] >= 1.25])
        avg_sharpe = sub['sharpe'].mean()
        unique_themes = sub['theme'].nunique()
        
        # Diversity: unique operator sets (simplified proxy)
        unique_exprs = sub['expression'].nunique()

        report.append({
            'run_id': rid,
            'count': len(sub),
            'gate_pass': gate_pass,
            'gate_pass_rate': gate_pass / len(sub) if len(sub) > 0 else 0,
            'avg_sharpe': avg_sharpe,
            'unique_themes': unique_themes,
            'unique_expressions': unique_exprs
        })

    # Display comparison
    rdf = pd.DataFrame(report)
    print(rdf.to_string(index=False))
    
    print("\n--- Theme Distribution ---")
    theme_dist = df.groupby(['run_id', 'theme']).size().unstack(fill_value=0)
    print(theme_dist)

    print("\n--- Detailed Metrics (Top 5 Sharpe per Leg) ---")
    for rid in [run_id_a, run_id_b]:
        print(f"\nTop 5 for {rid}:")
        top = df[df['run_id'] == rid].sort_values('sharpe', ascending=False).head(5)
        print(top[['sharpe', 'fitness', 'theme', 'mutation_type']])

    print("\n====================================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", required=True, help="Run ID for Leg A (Normal)")
    parser.add_argument("--b", required=True, help="Run ID for Leg B (RAG)")
    parser.add_argument("--db", default="alpha_results.db")
    args = parser.parse_args()
    
    compare_runs(args.a, args.b, db_path=args.db)
