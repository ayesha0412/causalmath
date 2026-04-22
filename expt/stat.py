import json
import argparse


def _empty_stats():
    return {
        "total_entries":        0,
        "original_token_total": 0,
        "improved_token_total": 0,
        "original_step_total":  0,
        "improved_step_total":  0,
        "original_acc_total":   0.0,
        "improved_acc_total":   0.0,
        "ps_chain_total":       0.0,
        "total_equiv_check":    0,
        "total_rollout_calls":  0,
        "pn_sum_total":         0.0,
        "pn_step_count":        0,
        "pn_global_max":        None,
        "pn_global_min":        None,
        "schema":               "unknown",
    }


def _accumulate(stats, metrics):
    stats["total_entries"] += 1
    stats["original_token_total"] += metrics.get("original_token_length", 0)
    stats["original_step_total"]  += metrics.get("original_step_length",  0)
    stats["original_acc_total"]   += metrics.get("original_accuracy",     0.0)
    stats["improved_token_total"] += metrics.get("token_length",  0)
    stats["improved_step_total"]  += metrics.get("step_length",   0)
    stats["improved_acc_total"]   += metrics.get("accuracy",      0.0)
    stats["ps_chain_total"]       += metrics.get("PS(chain)",     0.0)
    stats["total_equiv_check"]    += metrics.get("equiv_check_calls",   0)
    stats["total_rollout_calls"]  += metrics.get("total_rollout_calls",  0)
    pn_per_step = metrics.get("pn_per_step", [])
    if pn_per_step:
        stats["pn_sum_total"]  += sum(pn_per_step)
        stats["pn_step_count"] += len(pn_per_step)
        step_max = max(pn_per_step)
        step_min = min(pn_per_step)
        if stats["pn_global_max"] is None or step_max > stats["pn_global_max"]:
            stats["pn_global_max"] = step_max
        if stats["pn_global_min"] is None or step_min < stats["pn_global_min"]:
            stats["pn_global_min"] = step_min
    if "token_length" in metrics and "PS(chain)" in metrics:
        stats["schema"] = "full optimization schema"
    elif "avg_PN(steps)" in metrics:
        stats["schema"] = "PN measurement only"


def _print_stats(stats, label):
    n = stats["total_entries"]
    if n == 0:
        print(f"\n{label}: no entries.")
        return

    orig_tok  = stats["original_token_total"] / n
    impr_tok  = stats["improved_token_total"]  / n
    orig_stp  = stats["original_step_total"]   / n
    impr_stp  = stats["improved_step_total"]   / n
    orig_acc  = 100.0 * stats["original_acc_total"]  / n
    impr_acc  = 100.0 * stats["improved_acc_total"]  / n
    ps_chain  = 100.0 * stats["ps_chain_total"]      / n
    pn_n      = stats["pn_step_count"]
    avg_pn    = stats["pn_sum_total"] / pn_n if pn_n else None
    max_pn    = stats["pn_global_max"]
    min_pn    = stats["pn_global_min"]

    tok_pct   = 100.0 * (orig_tok - impr_tok) / orig_tok if orig_tok else 0
    stp_pct   = 100.0 * (orig_stp - impr_stp) / orig_stp if orig_stp else 0
    acc_delta = impr_acc - orig_acc

    avg_equiv   = stats["total_equiv_check"]   / n
    avg_rollout = stats["total_rollout_calls"]  / n
    avg_total   = avg_equiv + avg_rollout

    print(f"\n{label} ({n} entries):")
    print(f"  Token Length: {orig_tok:.1f} → {impr_tok:.1f} ({tok_pct:.1f}% reduction)")
    print(f"  Step Length:  {orig_stp:.1f} → {impr_stp:.1f} ({stp_pct:.1f}% reduction)")
    print(f"  Accuracy:     {orig_acc:.1f}% → {impr_acc:.1f}% ({acc_delta:+.1f}% change)")
    print(f"  PS(chain):    {ps_chain:.1f}%")
    if avg_pn is not None:
        print(f"  PN(steps):    avg={avg_pn:.3f} (over {pn_n} steps), max={max_pn:.3f}, min={min_pn:.3f}")
    print(f"  API Calls:    Equiv: {avg_equiv:.1f} + Rollouts: {avg_rollout:.1f} = Total: {avg_total:.1f} avg/entry")
    print(f"  Schema:       {stats['schema']}")


def compute_statistics(file_path):
    all_stats  = _empty_stats()   # all entries
    ps1_stats  = _empty_stats()   # entries where base model was correct (PS=1)
    ps0_stats  = _empty_stats()   # entries where base model was wrong  (PS=0)

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data    = json.loads(line)
                metrics = data.get("metrics", {})
                _accumulate(all_stats, metrics)
                if metrics.get("original_accuracy", 0) == 1:
                    _accumulate(ps1_stats, metrics)
                else:
                    _accumulate(ps0_stats, metrics)
            except json.JSONDecodeError:
                continue

    if all_stats["total_entries"] == 0:
        print("No entries found.")
        return

    # --- overall (matches paper Table 1 denominator) ---
    _print_stats(all_stats, "ALL entries (paper Table 1 denominator)")

    # --- PS=1 subset: the only subset where PNS optimization runs ---
    # Accuracy improvement here directly reflects the algorithm's effect.
    _print_stats(ps1_stats, "PS=1 subset (base model correct — PNS optimization applied)")

    # --- PS=0 subset: algorithm skips these; accuracy stays 0 ---
    _print_stats(ps0_stats, "PS=0 subset (base model wrong  — algorithm skipped)")


def main():
    parser = argparse.ArgumentParser(description="Analyze optimization metrics")
    parser.add_argument("file", help="JSONL file containing metrics")
    args = parser.parse_args()
    compute_statistics(args.file)


if __name__ == "__main__":
    main()