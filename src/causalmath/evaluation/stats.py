import json
import argparse
import os


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
        mx = max(pn_per_step)
        mn = min(pn_per_step)
        if stats["pn_global_max"] is None or mx > stats["pn_global_max"]:
            stats["pn_global_max"] = mx
        if stats["pn_global_min"] is None or mn < stats["pn_global_min"]:
            stats["pn_global_min"] = mn


def _derive(stats):
    n = stats["total_entries"]
    if n == 0:
        return None
    orig_tok = stats["original_token_total"] / n
    impr_tok = stats["improved_token_total"]  / n
    orig_stp = stats["original_step_total"]   / n
    impr_stp = stats["improved_step_total"]   / n
    orig_acc = 100.0 * stats["original_acc_total"] / n
    impr_acc = 100.0 * stats["improved_acc_total"] / n
    ps_pct   = 100.0 * stats["ps_chain_total"]     / n
    pn_n     = stats["pn_step_count"]
    avg_pn   = stats["pn_sum_total"] / pn_n if pn_n else 0.0
    tok_red  = 100.0 * (orig_tok - impr_tok) / orig_tok if orig_tok else 0.0
    stp_red  = 100.0 * (orig_stp - impr_stp) / orig_stp if orig_stp else 0.0
    return dict(
        n=n,
        orig_tok=orig_tok, impr_tok=impr_tok, tok_red=tok_red,
        orig_stp=orig_stp, impr_stp=impr_stp, stp_red=stp_red,
        orig_acc=orig_acc, impr_acc=impr_acc, acc_delta=impr_acc - orig_acc,
        ps_pct=ps_pct, avg_pn=avg_pn,
        pn_max=stats["pn_global_max"] or 0.0,
        pn_min=stats["pn_global_min"] or 0.0,
        avg_equiv=stats["total_equiv_check"] / n,
        avg_rollout=stats["total_rollout_calls"] / n,
    )


def compute_statistics(file_path):
    all_stats = _empty_stats()
    ps1_stats = _empty_stats()
    ps0_stats = _empty_stats()

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

    a  = _derive(all_stats)
    p1 = _derive(ps1_stats)
    p0 = _derive(ps0_stats)

    dataset = os.path.basename(file_path)

    W = 72
    print(f"\n{'='*W}")
    print(f"  {dataset}")
    print(f"{'='*W}")

    def section(label, d, note=""):
        if d is None:
            return
        sign = "+" if d["acc_delta"] >= 0 else ""
        print(f"\n  {label}  (N={d['n']}){('  ' + note) if note else ''}")
        print(f"  {'─'*60}")
        print(f"  Token Length : {d['orig_tok']:6.1f}  →  {d['impr_tok']:6.1f}  ({d['tok_red']:.1f}% reduction)")
        print(f"  Step Length  : {d['orig_stp']:6.1f}  →  {d['impr_stp']:6.1f}  ({d['stp_red']:.1f}% reduction)")
        print(f"  Accuracy     : {d['orig_acc']:5.1f}%  →  {d['impr_acc']:5.1f}%  ({sign}{d['acc_delta']:.1f}%)")
        print(f"  PS(chain)    : {d['ps_pct']:.1f}%")
        print(f"  PN(steps)    : avg={d['avg_pn']:.3f}  max={d['pn_max']:.3f}  min={d['pn_min']:.3f}")
        print(f"  API Calls    : Equiv {d['avg_equiv']:.1f} + Rollouts {d['avg_rollout']:.1f} = {d['avg_equiv']+d['avg_rollout']:.1f} avg/entry")

    section("ALL   ", a, "← Table 1 denominator")
    section("PS=1  ", p1, "← base model correct, PNS applied")
    section("PS=0  ", p0, "← base model wrong, skipped")

    print(f"\n{'='*W}\n")


def main():
    parser = argparse.ArgumentParser(description="Print Table 1 stats from a PNS output JSONL")
    parser.add_argument("file", help="JSONL file with metrics")
    args = parser.parse_args()
    compute_statistics(args.file)


if __name__ == "__main__":
    main()
