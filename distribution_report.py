import sqlite3
import pandas as pd
import numpy as np

def generate_distribution_report():
    conn = sqlite3.connect('alpha_results.db')
    
    # Lấy toàn bộ alpha đã mô phỏng (bao gồm cả bị reject)
    query = """
    SELECT sharpe, fitness, turnover, sub_sharpe, theme, mutation_type, expression 
    FROM alphas 
    WHERE error = '' OR error IS NULL
    """
    df = pd.read_sql_query(query, conn)
    conn.close()

    if df.empty:
        print("❌ Chưa có dữ liệu alpha nào để phân tích.")
        return

    print("====================================================")
    print("      BÁO CÁO PHÂN PHỐI CHẤT LƯỢNG ALPHA (RAW)")
    print("====================================================\n")

    # Thống kê tổng quát
    stats = df[['sharpe', 'fitness', 'turnover']].describe()
    print("--- Thống kê mô tả (Sharpe/Fitness/Turnover) ---")
    print(stats)
    print("\n")

    # Phân phối theo Theme
    print("--- Hiệu suất trung bình theo Theme (Chiến thuật) ---")
    theme_stats = df.groupby('theme')[['sharpe', 'fitness']].agg(['mean', 'std', 'max', 'count'])
    print(theme_stats)
    print("\n")

    # Kiểm tra "Vùng chết"
    bins = [-np.inf, 0, 0.5, 0.8, 1.0, 1.25, 1.5, np.inf]
    labels = ['<0 (Loss)', '0-0.5 (Noise)', '0.5-0.8 (Weak)', '0.8-1.0 (Potential)', '1.0-1.25 (Good)', '1.25-1.5 (Elite)', '>1.5 (God)']
    df['sharpe_bucket'] = pd.cut(df['sharpe'], bins=bins, labels=labels)
    
    print("--- Phân bổ Sharpe theo các nhóm ---")
    bucket_counts = df['sharpe_bucket'].value_counts().sort_index()
    for label, count in bucket_counts.items():
        percentage = (count / len(df)) * 100
        bar = "█" * int(percentage / 2)
        print(f"{label:20} | {count:4} ({percentage:5.1f}%) {bar}")

    # Top 3 'Quặng' tốt nhất hiện có
    print("\n--- Top 3 Alpha 'tiềm năng nhất' (bất kể bị reject) ---")
    top_3 = df.sort_values(by='sharpe', ascending=False).head(3)
    for _, row in top_3.iterrows():
        print(f"Sharpe: {row['sharpe']:.3f} | Theme: {row['theme']} | Expr: {row['expression'][:80]}...")

    print("\n====================================================")

if __name__ == "__main__":
    generate_distribution_report()
