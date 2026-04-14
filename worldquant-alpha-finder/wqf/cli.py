"""wqf/cli.py — Terminal UI for alpha discovery."""
import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from .ranker import score_expression, rank_candidates
from .hypothesis import HypothesisEngine
from .db import AlphaDB
from .wq_client import WQClient

console = Console()


# ── Shared: ensure .env exists ─────────────────────────────────
def _check_env():
    env_path = __import__('os').path.join(__import__('os').path.dirname(__file__), '..', '.env')
    example = __import__('os').path.join(__import__('os').path.dirname(__file__), '..', '.env.example')
    if not __import__('os').path.exists(env_path):
        if __import__('os').path.exists(example):
            import shutil
            shutil.copy(example, env_path)
        console.print("[yellow].env not found — created from .env.example[/yellow]")
        console.print("[yellow]Fill in WQ_EMAIL and WQ_PASSWORD, then re-run.[/yellow]")


# ── Commands ────────────────────────────────────────────────────
@click.group()
@click.version_option(version="1.0.0")
def cli():
    """WorldQuant Alpha Finder — discover winning alphas from hypotheses."""
    pass


@cli.command()
@click.option("--count", default=20, help="Number of candidates to generate")
@click.option("--min-score", default=40.0, help="Minimum ranker score")
@click.option("--simulate", "do_sim", is_flag=True, help="Also run WQ Brain simulation")
@click.option("--top-n", default=5, help="Simulate only top N candidates")
def run(count, min_score, do_sim, top_n):
    """Generate candidates, rank them, optionally simulate on WQ Brain."""
    engine = HypothesisEngine()
    db = AlphaDB()

    console.print(f"\n[dim]Generating {count} candidates from hypothesis engine...[/dim]")
    raw = engine.generate(count)
    exprs = [expr for expr, _, _ in raw]

    console.print(f"[dim]Scoring {len(exprs)} expressions...[/dim]")
    ranked = rank_candidates(exprs, min_score=min_score)

    # Build display table
    table = Table(title=f"Ranked Candidates  (showing {len(ranked)}/{len(exprs)})")
    table.add_column("#", style="dim", width=3)
    table.add_column("Score", width=6)
    table.add_column("Hypothesis", width=26)
    table.add_column("Expression", width=68)

    for i, (expr, score, hypo, reasons) in enumerate(ranked, 1):
        bar = "█" * int(score / 5)
        table.add_row(str(i), f"[green]{score:.1f}[/green] {bar}", hypo, expr[:68])
        db.save_candidate(expr, score, hypo, reasons)

    console.print(table)

    # Hypothesis breakdown
    hypo_counts = {}
    for _, score, hypo, _ in ranked:
        hypo_counts[hypo] = hypo_counts.get(hypo, 0) + 1
    console.print(f"\n[dim]Hypothesis distribution: {dict(sorted(hypo_counts.items(), key=lambda x: -x[1]))}[/dim]")

    if do_sim:
        if not ranked:
            console.print("[yellow]No candidates above threshold — nothing to simulate.[/yellow]")
            db.close()
            return

        try:
            client = WQClient()
        except RuntimeError as e:
            console.print(f"[red]Auth error: {e}[/red]")
            db.close()
            return

        to_sim = ranked[:top_n]
        console.print(f"\n[yellow]Simulating top {len(to_sim)} on WQ Brain...[/yellow]")
        for expr, score, hypo, _ in to_sim:
            console.print(f"\n  [{score:.1f}] {hypo}")
            console.print(f"  → {expr[:75]}")
            try:
                result = client.simulate(expr)
                db.save_result(expr, result)
                if result.get("error"):
                    console.print(f"    [red]Error: {result['error']}[/red]")
                else:
                    s = result.get("sharpe", 0)
                    f = result.get("fitness", 0)
                    t = result.get("turnover", 0)
                    color = "[green]" if s >= 1.25 else "[yellow]" if s >= 0.5 else "[red]"
                    console.print(f"    {color}Sharpe={s:.2f}  Fitness={f:.2f}  Turnover={t:.0f}%[/]")
            except Exception as e:
                console.print(f"    [red]Exception: {e}[/red]")

    db.close()


@cli.command()
@click.argument("file", type=click.Path(exists=True))
@click.option("--min-score", default=40.0)
def rank(file, min_score):
    """Score expressions from a file (one per line, # = comment)."""
    exprs = [l.strip() for l in open(file, encoding="utf-8")
             if l.strip() and not l.strip().startswith("#")]
    console.print(f"[dim]Scoring {len(exprs)} expressions from {file}...[/dim]")
    ranked = rank_candidates(exprs, min_score=min_score)

    table = Table(title=f"Ranked  {len(ranked)}/{len(exprs)} passed (min={min_score})")
    table.add_column("#", style="dim", width=3)
    table.add_column("Score", width=28)
    table.add_column("Hypothesis", width=24)
    table.add_column("Expression", width=65)

    for i, (expr, score, hypo, reasons) in enumerate(ranked, 1):
        bar = "█" * int(score / 5)
        table.add_row(str(i), f"[green]{score:.1f}[/green] {bar}", hypo, expr[:65])

    console.print(table)


@cli.command()
@click.option("--top", default=10, help="Show top N performers")
def analyze(top):
    """Show portfolio stats, top performers, hypothesis breakdown."""
    db = AlphaDB()
    stats = db.get_stats()
    top_rows = db.get_top(top)
    breakdown = db.get_hypothesis_breakdown()
    db.close()

    console.print(Panel(f"[bold]Portfolio Summary[/bold]\n"
                         f"  Total simulated : {stats['total']}\n"
                         f"  Win rate (S>=1.25): {stats['win_rate']:.1%}\n"
                         f"  Elite (S>=2.5)   : {stats['elite']}\n"
                         f"  Avg Sharpe      : {stats['avg_sharpe']:.2f}",
                         title="[bold]Alpha Finder[/bold]", border_style="blue"))

    if top_rows:
        t = Table(title=f"Top {len(top_rows)} Performers")
        t.add_column("Sharpe", width=8)
        t.add_column("Fitness", width=8)
        t.add_column("T/O%", width=6)
        t.add_column("Hypothesis", width=26)
        t.add_column("Expression", width=60)
        for r in top_rows:
            s = r["sharpe"]
            color = "[green]" if s >= 1.25 else "[yellow]" if s >= 0.5 else "[red]"
            t.add_row(
                f"{color}{s:.2f}[/]",
                f"{r['fitness']:.2f}",
                f"{r['turnover']:.0f}%",
                r.get("hypothesis", "?")[:26],
                r["expression"][:60],
            )
        console.print(t)

    if breakdown:
        t2 = Table(title="Hypothesis Win Rates")
        t2.add_column("Hypothesis", width=26)
        t2.add_column("N", width=5)
        t2.add_column("Win Rate", width=10)
        t2.add_column("Avg Sharpe", width=12)
        for name, vals in sorted(breakdown.items(), key=lambda x: -(x[1].get("win_rate") or 0)):
            wr = vals["win_rate"] or 0
            t2.add_row(name, str(vals["total"]),
                       f"{wr:.0%}", f"{vals['avg_sharpe'] or 0:.2f}")
        console.print(t2)


@cli.command()
def status():
    """Show current WQ Brain submission status."""
    try:
        client = WQClient()
        info = client.get_status()
        console.print("[bold]WQ Brain Status[/bold]")
        if isinstance(info, dict) and "error" in info:
            console.print(f"[red]{info['error']}[/red]")
        elif isinstance(info, list):
            console.print(f"[green]Connected — {len(info)} alpha(s) found[/green]")
        else:
            console.print(info)
    except RuntimeError as e:
        console.print(f"[red]Auth error: {e}[/red]")
        console.print("[yellow]Tip: cp .env.example .env  →  edit .env[/yellow]")


@cli.command()
def hypotheses():
    """List all hypothesis categories."""
    engine = HypothesisEngine()
    items = engine.list_hypotheses()
    t = Table(title="Hypothesis Categories (weighted sampling)")
    t.add_column("ID", width=16)
    t.add_column("Name", width=26)
    t.add_column("Weight", width=8)
    t.add_column("Description", width=60)
    for hid, name, weight, desc in items:
        t.add_row(hid, name, f"{weight:.0%}", desc)
    console.print(t)


if __name__ == "__main__":
    cli()
