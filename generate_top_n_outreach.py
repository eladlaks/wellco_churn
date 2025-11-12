#!/usr/bin/env python3
"""
Generate a CSV of the top N members for outreach.

Produces a CSV with columns:
- member_id
- prioritization_score
- rank

Usage:
    python generate_top_n_outreach.py --input members.csv --n 50 --output top_50_members.csv
"""
from __future__ import annotations
import csv
import argparse
from typing import List, Dict, Any


def read_members(path: str, id_col: str = "member_id", score_col: str = "score") -> List[Dict[str, Any]]:
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows = []
        for r in reader:
            if id_col not in r:
                raise KeyError(f"ID column '{id_col}' not found in input file.")
            # Try to find a suitable score column if the provided one is missing
            if score_col not in r:
                # common alternative names
                for alt in ("prioritization_score", "priority_score", "score"):
                    if alt in r:
                        score_col = alt
                        break
                else:
                    raise KeyError(f"Score column '{score_col}' not found in input file.")
            try:
                score = float(r.get(score_col, "") or 0.0)
            except ValueError:
                # Skip rows with non-numeric score
                continue
            rows.append({"member_id": r[id_col], "prioritization_score": score})
        return rows


def write_top_n(members: List[Dict[str, Any]], n: int, output_path: str) -> None:
    # Sort descending by prioritization_score
    sorted_members = sorted(members, key=lambda m: m["prioritization_score"], reverse=True)
    top_n = sorted_members[:n]
    with open(output_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["member_id", "prioritization_score", "rank"])
        for idx, m in enumerate(top_n, start=1):
            writer.writerow([m["member_id"], f"{m['prioritization_score']:.6f}", idx])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate CSV of top N members for outreach.")
    p.add_argument("--input", "-i", required=True, help="Input CSV file with at least member_id and a score column.")
    p.add_argument("--output", "-o", default="top_n_members.csv", help="Output CSV file path.")
    p.add_argument("--n", "-n", type=int, default=100, help="Number of top members to include.")
    p.add_argument("--id-column", default="member_id", help="Name of the member id column in the input file.")
    p.add_argument("--score-column", default="score", help="Name of the score column in the input file.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    members = read_members(args.input, id_col=args.id_column, score_col=args.score_column)
    if not members:
        raise SystemExit("No valid members found in input file.")
    write_top_n(members, args.n, args.output)


if __name__ == "__main__":
    main()