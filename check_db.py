import sqlite3
import pandas as pd
from datetime import datetime, timedelta

def check_submissions():
    conn = sqlite3.connect('alpha_results.db')
    
    # Check total alphas and submit_state counts
    print("--- Tổng quan Alphas (Submit State) ---")
    df_counts = pd.read_sql_query("SELECT submit_state, count(*) as count FROM alphas GROUP BY submit_state", conn)
    print(df_counts)
    
    # Check last 10 alphas
    print("\n--- 10 Alpha mới nhất ---")
    df_recent = pd.read_sql_query("SELECT id, expression, sharpe, fitness, submit_state, submitted_at, alpha_id FROM alphas ORDER BY id DESC LIMIT 10", conn)
    print(df_recent)
    
    # Check successful submissions
    print(f"\n--- Submissions thành công ---")
    df_submitted = pd.read_sql_query("SELECT id, expression, sharpe, alpha_id, submitted_at, submit_state FROM alphas WHERE submit_state IN ('submitted', 'accepted', 'rejected_review') OR submitted = 1", conn)
    print(df_submitted)
    
    conn.close()

if __name__ == "__main__":
    check_submissions()
