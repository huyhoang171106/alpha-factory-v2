"""
WorldQuant Alpha Finder — Clean terminal entry point
Run: python main.py <command>
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from wqf.cli import cli
raise SystemExit(cli())
