#!/usr/bin/env python3
"""Recover exact confusion matrices from the reported metrics.

For each fold we know, to three decimals, recall and F1, and we know the
clinical cost EXACTLY, because cost = 10*FN + FP is an integer and both
FN and FP are integers. That integer constraint is strong enough to pin
the confusion matrix down. This script brute-forces all (P, FN, FP) with
P + N = n that reproduce every reported figure, and reports whether the
solution is unique.

If unique, precision / specificity / FPR are not estimates -- they are
determined by the numbers already in the paper.
"""
import itertools

N = 1000

# fold -> (recall, f1, cost) for each mode, from tables/tab_perfold.tex
SISFALL = {
    "tier1_only": {
        0: (0.781, 0.774, 1042),
        1: (0.748, 0.737, 1223),
        2: (0.783, 0.755, 1066),
        3: (0.732, 0.766, 1248),
        4: (0.749, 0.760, 1175),
    },
    "gate_knn_llm": {
        0: (0.986, 0.892, 156),
        1: (0.911, 0.851, 490),
        2: (0.998, 0.874, 134),
        3: (0.995, 0.930, 83),
        4: (0.993, 0.887, 136),
    },
}

KFALL = {
    "tier1_only": {
        0: (0.998, 0.992, 17),
        1: (0.996, 0.987, 31),
        2: (0.994, 0.987, 40),
        3: (0.990, 0.992, 53),
        4: (0.979, 0.980, 109),
    },
    "gate_knn_llm": {
        0: (1.000, 0.993, 7),
        1: (0.996, 0.986, 32),
        2: (0.996, 0.989, 29),
        3: (0.996, 0.995, 23),
        4: (0.994, 0.989, 38),
    },
}

BETA = 10
TOL = 5e-4          # a printed 3-decimal value rounds from +/- 0.0005


def solve(recall, f1, cost, n=N, pmin=200, pmax=800):
    """All (P, TP, FP, FN, TN) consistent with the reported metrics."""
    out = []
    for fn in range(0, cost // BETA + 1):
        fp = cost - BETA * fn
        if fp < 0 or fp > n:
            continue
        for p in range(pmin, pmax + 1):
            tp = p - fn
            if tp < 0:
                continue
            neg = n - p
            tn = neg - fp
            if tn < 0:
                continue
            if abs(tp / p - recall) >= TOL:
                continue
            denom = 2 * tp + fp + fn
            if denom == 0:
                continue
            if abs(2 * tp / denom - f1) >= TOL:
                continue
            out.append((p, tp, fp, fn, tn))
    return out


def joint_solve(table, n=N):
    """Solve both modes per fold under the shared-slice constraint.

    tier1_only and gate_knn_llm are evaluated on the SAME stratified test
    slice, so the positive count P must be identical for both. Requiring
    that collapses most of the per-mode ambiguity.
    """
    folds = sorted(table["tier1_only"])
    out = {}
    for f in folds:
        cands = {}
        for mode in ("tier1_only", "gate_knn_llm"):
            rec, f1, cost = table[mode][f]
            byp = {}
            for sol in solve(rec, f1, cost, n=n):
                byp.setdefault(sol[0], []).append(sol)
            cands[mode] = byp
        shared = sorted(set(cands["tier1_only"]) & set(cands["gate_knn_llm"]))
        out[f] = (shared, cands)
    return out


def report_joint(name, table):
    print("=" * 78)
    print("%s -- joint solve under shared-slice constraint" % name)
    print("=" * 78)
    res = joint_solve(table)
    rows = {"tier1_only": [], "gate_knn_llm": []}
    for f, (shared, cands) in sorted(res.items()):
        if not shared:
            print("  fold %d: NO shared P -- modes disagree on the slice" % f)
            continue
        status = "UNIQUE P" if len(shared) == 1 else "P in %s" % shared
        print("\n  fold %d: %s" % (f, status))
        p = shared[len(shared) // 2]
        for mode in ("tier1_only", "gate_knn_llm"):
            sols = cands[mode][p]
            _, tp, fp, fn, tn = sols[0]
            prec, spec = tp / (tp + fp), tn / (tn + fp)
            amb = "" if len(sols) == 1 else "  (%d ties at this P)" % len(sols)
            print("    %-14s P=%d TP=%d FP=%d FN=%d TN=%d | "
                  "prec %.3f  spec %.3f%s"
                  % (mode, p, tp, fp, fn, tn, prec, spec, amb))
            rows[mode].append((f, p, tp, fp, fn, tn, prec, spec))
    print("\n  five-fold means")
    for mode in ("tier1_only", "gate_knn_llm"):
        r = rows[mode]
        if not r:
            continue
        k = len(r)
        mp = sum(x[6] for x in r) / k
        ms = sum(x[7] for x in r) / k
        sd = lambda xs: (sum((x - sum(xs) / len(xs)) ** 2
                             for x in xs) / (len(xs) - 1)) ** 0.5
        print("    %-14s precision %.3f+/-%.3f   specificity %.3f+/-%.3f"
              % (mode, mp, sd([x[6] for x in r]), ms, sd([x[7] for x in r])))
    print("    prevalence P/n: %s"
          % [x[1] for x in rows["tier1_only"]])
    return rows


def report(name, table):
    print("=" * 74)
    print(name)
    print("=" * 74)
    summary = {}
    for mode, folds in table.items():
        print("\n  %s" % mode)
        print("  fold   P    TP   FP   FN   TN  | prec.  spec.   FPR   solutions")
        print("  " + "-" * 68)
        rows = []
        for f, (rec, f1, cost) in sorted(folds.items()):
            sols = solve(rec, f1, cost)
            if not sols:
                print("  %4d   NO SOLUTION -- reported metrics are inconsistent" % f)
                continue
            # If several, they are near-identical; take the modal P.
            p, tp, fp, fn, tn = sols[0]
            prec = tp / (tp + fp)
            spec = tn / (tn + fp)
            fpr = fp / (tn + fp)
            rows.append((f, p, tp, fp, fn, tn, prec, spec, fpr))
            flag = "UNIQUE" if len(sols) == 1 else "%d cands" % len(sols)
            print("  %4d %4d %5d %4d %4d %4d | %.3f  %.3f  %.3f   %s"
                  % (f, p, tp, fp, fn, tn, prec, spec, fpr, flag))
        if rows:
            k = len(rows)
            mp = sum(r[6] for r in rows) / k
            ms = sum(r[7] for r in rows) / k
            mf = sum(r[8] for r in rows) / k
            sd = lambda xs: (sum((x - sum(xs) / len(xs)) ** 2
                                 for x in xs) / (len(xs) - 1)) ** 0.5
            print("  mean                          | %.3f  %.3f  %.3f"
                  % (mp, ms, mf))
            print("  std                           | %.3f  %.3f  %.3f"
                  % (sd([r[6] for r in rows]), sd([r[7] for r in rows]),
                     sd([r[8] for r in rows])))
            summary[mode] = rows
    return summary


if __name__ == "__main__":
    report_joint("SisFall main protocol (n=1000/fold)", SISFALL)
    print()
    report_joint("KFall binary external (n=1000/fold)", KFALL)
    print()
    s = report("SisFall main protocol (n=1000/fold)", SISFALL)
    report("KFall binary external (n=1000/fold)", KFALL)

    # Prevalence consistency across modes is an independent sanity check:
    # the two modes see the SAME test slice, so P must agree per fold.
    print("\n" + "=" * 74)
    print("Cross-mode prevalence agreement (same slice -> same P)")
    print("=" * 74)
    if "tier1_only" in s and "gate_knn_llm" in s:
        a = {r[0]: r[1] for r in s["tier1_only"]}
        b = {r[0]: r[1] for r in s["gate_knn_llm"]}
        for f in sorted(a):
            mark = "agree" if a[f] == b.get(f) else "DIFFER"
            print("  fold %d: tier1 P=%d, stack P=%d  -> %s"
                  % (f, a[f], b.get(f, -1), mark))
