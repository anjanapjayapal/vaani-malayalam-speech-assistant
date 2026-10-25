"""Compute screening metrics for the 10-example QA pilot. Not a semantic evaluator."""
import argparse
import unicodedata
import pandas as pd

def norm(x):
    return " ".join(unicodedata.normalize("NFC", str(x)).casefold().strip().split())

def main():
    p = argparse.ArgumentParser()
    p.add_argument("input_csv", help="qa_seed_comparison.csv")
    p.add_argument("--output", default="qa_pilot_metrics.csv")
    args = p.parse_args()
    df = pd.read_csv(args.input_csv)
    df["baseline_match"] = df.apply(
        lambda r: norm(r["expected_answer"]) in norm(r["baseline_answer"]), axis=1
    )
    df["context_match"] = df.apply(
        lambda r: norm(r["expected_answer"]) in norm(r["context_answer"]), axis=1
    )
    metrics = pd.DataFrame([
        {"condition": "baseline_no_context", "n": len(df),
         "exact_substring_match_rate": df["baseline_match"].mean(),
         "mean_latency_s": df["baseline_latency_s"].mean()},
        {"condition": "context_assisted", "n": len(df),
         "exact_substring_match_rate": df["context_match"].mean(),
         "mean_latency_s": df["context_latency_s"].mean()}
    ])
    metrics.to_csv(args.output, index=False)
    print(metrics.to_string(index=False))
    print("Caution: substring match is a screening metric, not semantic correctness.")

if __name__ == "__main__":
    main()
