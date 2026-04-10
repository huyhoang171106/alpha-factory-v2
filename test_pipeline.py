from pipeline import AlphaFactory
import os
import sys

def main():
    print("🚀 Khởi chạy VIP Pipeline (Dry Run Test)...")
    try:
        # Use a dummy email/password since we are dry-running
        factory = AlphaFactory("dummy@email.com", "dummy_pass")
        stats = factory.run_daily(
            target_alphas=5,
            max_candidates=50,
            max_simulations=10,
            auto_submit=False,
            evolve=True,
            harvest=False,  # Skip harvest in dry run unless we have DB data
            learn=True,
            cleanup=False,
            dry_run=True
        )
        print("\n✅ DRY RUN SUCCESSFUL!")
        
        # Test dashboard
        print("\n📊 Kiểm tra Dashboard (Lineage + Theme Analytics):")
        factory.show_dashboard()
        
    except Exception as e:
        print(f"\n❌ LỖI TRONG PIPELINE: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
