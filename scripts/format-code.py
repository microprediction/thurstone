#!/usr/bin/env python3
"""
Format code to exactly match CI expectations using ruff.

This script ensures your local formatting matches CI exactly by using
the same ruff version and configuration.

Usage:
    python scripts/format-code.py        # Format all code
    python scripts/format-code.py --check # Check formatting only
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        print(f"✅ {description} passed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed!")
        if e.stdout:
            print("STDOUT:", e.stdout)
        if e.stderr:
            print("STDERR:", e.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(description="Format code to match CI using ruff")
    parser.add_argument(
        "--check", action="store_true", help="Check formatting only, don't modify files"
    )
    args = parser.parse_args()

    root = Path(__file__).parent.parent
    print(f"📁 Working in: {root}")

    success = True

    if args.check:
        # Check formatting
        format_cmd = ["ruff", "format", "--check", "--diff", str(root)]
        success &= run_command(format_cmd, "Ruff format check")

        # Check linting
        lint_cmd = ["ruff", "check", str(root)]
        success &= run_command(lint_cmd, "Ruff lint check")
    else:
        # Format code
        format_cmd = ["ruff", "format", str(root)]
        success &= run_command(format_cmd, "Ruff formatting")

        # Check linting (don't auto-fix in format mode)
        lint_cmd = ["ruff", "check", str(root)]
        success &= run_command(lint_cmd, "Ruff lint check")

    if success:
        action = "check passed" if args.check else "formatting complete"
        print(f"\n🎉 All {action}! Your code matches CI expectations.")
    else:
        action = "checks failed" if args.check else "formatting had errors"
        print(f"\n💥 Some {action}. Please fix the issues above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
