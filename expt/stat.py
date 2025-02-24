import json
import argparse


def compute_statistics(file_path):
    stats = {
        "total_entries": 0,
        # Aggregated raw values
        "original_token_total": 0,
        "improved_token_total": 0,
        "original_step_total": 0,
        "improved_step_total": 0,
        "original_acc_total": 0.0,
        "improved_acc_total": 0.0,
        "ps_chain_total": 0.0,
        "total_equiv_check": 0,
        "total_rollout_calls": 0,
        # Calculated comparisons
        "token_summary": "",
        "step_summary": "",
        "acc_summary": "",
        "ps_summary": "",
        "api_summary": ""
    }

    with open(file_path, "r") as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                metrics = data.get("metrics", {})
                stats["total_entries"] += 1

                # Aggregate values
                stats["original_token_total"] += metrics.get("original_token_length", 0)
                stats["improved_token_total"] += metrics.get("token_length", 0)
                stats["original_step_total"] += metrics.get("original_step_length", 0)
                stats["improved_step_total"] += metrics.get("step_length", 0)
                stats["original_acc_total"] += metrics.get("original_accuracy", 0.0)
                stats["improved_acc_total"] += metrics.get("accuracy", 0.0)
                stats["ps_chain_total"] += metrics.get("PS(chain)", 0.0)
                stats["total_equiv_check"] += metrics.get("equiv_check_calls", 0)
                stats["total_rollout_calls"] += metrics.get("total_rollout_calls", 0)

            except json.JSONDecodeError:
                continue

    if stats["total_entries"] > 0:
        n = stats["total_entries"]

        # Token length comparison
        orig_token = stats["original_token_total"] / n
        impr_token = stats["improved_token_total"] / n
        token_pct = 100 * (orig_token - impr_token) / orig_token if orig_token else 0
        stats["token_summary"] = (
            f"{orig_token:.1f} → {impr_token:.1f} ({token_pct:.1f}% reduction)"
        )

        # Step length comparison
        orig_step = stats["original_step_total"] / n
        impr_step = stats["improved_step_total"] / n
        step_pct = 100 * (orig_step - impr_step) / orig_step if orig_step else 0
        stats["step_summary"] = (
            f"{orig_step:.1f} → {impr_step:.1f} ({step_pct:.1f}% reduction)"
        )

        # Accuracy comparison
        orig_acc = 100 * stats["original_acc_total"] / n
        impr_acc = 100 * stats["improved_acc_total"] / n
        acc_pct = impr_acc - orig_acc
        stats["acc_summary"] = (
            f"{orig_acc:.1f}% → {impr_acc:.1f}% ({acc_pct:+.1f}% change)"
        )

        # PS(chain) summary
        ps_chain = 100 * stats["ps_chain_total"] / n
        stats["ps_summary"] = f"{ps_chain:.1f}%"

        # API call summary (averages)
        avg_equiv = stats["total_equiv_check"] / n
        avg_rollout = stats["total_rollout_calls"] / n
        avg_total = avg_equiv + avg_rollout
        stats["api_summary"] = (
            f"Equiv checks: {avg_equiv:.1f} + "
            f"Rollouts: {avg_rollout:.1f} = "
            f"Total: {avg_total:.1f}"
        )

    return stats


def main():
    parser = argparse.ArgumentParser(description="Analyze optimization metrics")
    parser.add_argument("file", help="JSONL file containing metrics")
    args = parser.parse_args()

    stats = compute_statistics(args.file)

    print(f"Analysis Results ({stats['total_entries']} entries):")
    print(f"Token Length: {stats['token_summary']}")
    print(f"Step Length:  {stats['step_summary']}")
    print(f"Accuracy:     {stats['acc_summary']}")
    print(f"PS(chain):    {stats['ps_summary']}")
    print(f"API Calls:    {stats['api_summary']} (avg per entry)")


if __name__ == "__main__":
    main()