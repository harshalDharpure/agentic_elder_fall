#!/usr/bin/env python3
"""Recompute descriptive paired fold-bootstrap intervals without compilation.

SisFall uses supplied main-ablation confusion counts, before metric rounding.
KFall uses the rounded per-fold source table. Five overlapping training folds
are not independent experiments: these intervals are descriptive, not evidence
of population-level significance or clinical validation.
"""
import random
import re
from pathlib import Path
from check_reported_results import grouped_metrics, load_counts

R = 10000
SEED = 42
ALPHA = 0.05


def _phi(z):
    """Standard normal CDF via erf."""
    import math
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _phi_inv(p):
    """Acklam's rational approximation to the normal quantile."""
    import math
    if p <= 0.0:
        return -8.0
    if p >= 1.0:
        return 8.0
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    pl = 0.02425
    if p < pl:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > 1 - pl:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def mean(xs):
    return sum(xs) / len(xs)


def bootstrap(deltas, reps=R, seed=SEED, alpha=ALPHA):
    """Percentile and BCa intervals for the mean of paired deltas."""
    rng = random.Random(seed)
    n = len(deltas)
    theta = mean(deltas)
    boots = []
    for _ in range(reps):
        boots.append(mean([deltas[rng.randrange(n)] for _ in range(n)]))
    boots.sort()

    def q(pp):
        i = min(max(int(round(pp * (reps - 1))), 0), reps - 1)
        return boots[i]

    pct = (q(alpha / 2), q(1 - alpha / 2))

    # bias correction
    n_less = sum(1 for b in boots if b < theta)
    z0 = _phi_inv(max(min(n_less / reps, 1 - 1e-9), 1e-9))
    # jackknife acceleration
    jack = [mean(deltas[:i] + deltas[i + 1:]) for i in range(n)]
    jbar = mean(jack)
    num = sum((jbar - x) ** 3 for x in jack)
    den = 6.0 * (sum((jbar - x) ** 2 for x in jack) ** 1.5)
    a = num / den if den else 0.0

    def adj(z_alpha):
        zz = z0 + z_alpha
        return _phi(z0 + zz / (1 - a * zz))

    lo = adj(_phi_inv(alpha / 2))
    hi = adj(_phi_inv(1 - alpha / 2))
    bca = (q(lo), q(hi))
    return theta, pct, bca, z0, a


def sisfall_pair():
    groups = grouped_metrics(load_counts())
    return tuple({metric: [row[metric] for row in groups[('main_ablation', method)]]
                  for metric in ('f1', 'recall', 'cost')}
                 for method in ('tier1_only', 'gate_knn_llm'))


def kfall_pair():
    text = (Path(__file__).resolve().parent / 'tables/tab_kfall_perfold.tex').read_text(encoding='utf-8')
    rows = []
    for line in text.splitlines():
        if re.match(r'^[0-4]\s*&', line):
            # Columns: fold, n, Tier-1 F1/recall/cost, stack F1/recall/cost.
            rows.append([float(cell.strip()) for cell in line.split('&')[:8]])
    assert len(rows) == 5, 'KFall requires five source rows'
    return tuple({metric: [row[offset + i] for row in rows]
                  for i, metric in enumerate(('f1', 'recall', 'cost'))}
                 for offset in (2, 5))


def run(name, base, stack):
    print("=" * 72)
    print("%s  (paired, n=5 folds, %d resamples, seed=%d)" % (name, R, SEED))
    print("=" * 72)
    for key, label in (("f1", "F1"), ("recall", "Recall"),
                       ("cost", "Cost/1000")):
        d = [s - b for s, b in zip(stack[key], base[key])]
        theta, pct, bca, z0, a = bootstrap(d)
        pos = sum(1 for x in d if (x < 0 if key == "cost" else x > 0))
        fmt = "%.3f" if key != "cost" else "%.1f"
        print("\n  %s" % label)
        print("    per-fold deltas : %s"
              % ", ".join(fmt % x for x in d))
        print("    mean delta      : " + fmt % theta)
        print("    percentile 95%%  : [" + fmt % pct[0] + ", " + fmt % pct[1] + "]")
        print("    BCa 95%%         : [" + fmt % bca[0] + ", " + fmt % bca[1] + "]"
              + "   (z0=%.3f, a=%.3f)" % (z0, a))
        print("    improved on     : %d/5 folds" % pos)


if __name__ == "__main__":
    run("SisFall main protocol", *sisfall_pair())
    print()
    run("KFall binary external", *kfall_pair())
