#!/usr/bin/env python3
"""
ADE3x3 Pipeline Orchestrator

This script provides a convenient way to run the entire ADE3x3 pipeline
or specific steps in sequence.

Usage:
    python scripts/run_pipeline.py              # Run all steps
    python scripts/run_pipeline.py 1-10         # Run steps 1 through 10
    python scripts/run_pipeline.py 5            # Run step 5 only
    python scripts/run_pipeline.py 5,7,9        # Run specific steps
"""

import sys
import os
import importlib
import argparse
from pathlib import Path

# Add src directory to path
SRC_DIR = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))


def get_available_steps():
    """Discover all available step modules."""
    steps_dir = SRC_DIR / "ade3x3" / "steps"
    if not steps_dir.exists():
        return []

    step_files = sorted(steps_dir.glob("ade3x3_step*.py"))
    steps = []

    for step_file in step_files:
        # Extract step number/identifier from filename
        name = step_file.stem
        if "step" in name:
            steps.append(name)

    return steps


def parse_step_range(range_str):
    """Parse step range string like '1-10', '5', or '1,3,5'."""
    if '-' in range_str:
        start, end = range_str.split('-')
        return list(range(int(start), int(end) + 1))
    elif ',' in range_str:
        return [int(x.strip()) for x in range_str.split(',')]
    else:
        return [int(range_str)]


def run_step(step_name):
    """Run a single step module."""
    try:
        module_path = f"ade3x3.steps.{step_name}"
        print(f"\n{'='*60}")
        print(f"Running: {step_name}")
        print(f"{'='*60}\n")

        module = importlib.import_module(module_path)

        # If the module has a main() function, call it
        if hasattr(module, 'main'):
            module.main()

        print(f"\n✓ Completed: {step_name}\n")
        return True

    except Exception as e:
        print(f"\n✗ Error in {step_name}: {e}\n")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Run ADE3x3 pipeline steps",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_pipeline.py              # Run all steps
  python scripts/run_pipeline.py 1-10         # Run steps 1-10
  python scripts/run_pipeline.py 5            # Run step 5
  python scripts/run_pipeline.py 1,3,5        # Run steps 1, 3, 5
  python scripts/run_pipeline.py --list       # List available steps
        """
    )

    parser.add_argument(
        'steps',
        nargs='?',
        default='all',
        help='Step range to run (e.g., "1-10", "5", "1,3,5", or "all")'
    )

    parser.add_argument(
        '--list',
        action='store_true',
        help='List all available steps'
    )

    parser.add_argument(
        '--stop-on-error',
        action='store_true',
        help='Stop pipeline if any step fails'
    )

    args = parser.parse_args()

    # Get available steps
    available_steps = get_available_steps()

    if args.list:
        print("Available steps:")
        for step in available_steps:
            print(f"  - {step}")
        return

    # Determine which steps to run
    if args.steps == 'all':
        steps_to_run = available_steps
    else:
        try:
            step_numbers = parse_step_range(args.steps)
            # Find matching step modules
            steps_to_run = [
                s for s in available_steps
                if any(f"step{n}" in s or f"step0{n}" in s for n in step_numbers)
            ]
        except ValueError as e:
            print(f"Error parsing step range: {e}")
            return

    if not steps_to_run:
        print(f"No steps found matching: {args.steps}")
        return

    print(f"Running {len(steps_to_run)} step(s)...")

    # Run steps
    success_count = 0
    fail_count = 0

    for step in steps_to_run:
        success = run_step(step)
        if success:
            success_count += 1
        else:
            fail_count += 1
            if args.stop_on_error:
                print("Stopping pipeline due to error.")
                break

    # Summary
    print("\n" + "="*60)
    print("Pipeline Summary")
    print("="*60)
    print(f"Total steps: {len(steps_to_run)}")
    print(f"✓ Successful: {success_count}")
    print(f"✗ Failed: {fail_count}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
