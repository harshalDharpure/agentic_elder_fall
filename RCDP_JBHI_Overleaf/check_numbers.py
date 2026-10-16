#!/usr/bin/env python3
"""Numeric consistency check for the RCDP result tables.

For every table in tables/ that has per-fold rows followed by a
"Mean" row, recompute the column means (and sample std where the Mean
row is written as "a$\\pm$b") and compare against what is printed.

This is the check that catches transcription drift between a re-run and
the manuscript. It is deliberately independent of the pipeline: it reads
only the .tex the reviewer will read.

Usage:  python check_numbers.py
"""
import io
import os
import re
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tables")

# Absolute tolerance per column, chosen from the printed precision.
def tol_for(printed):
    if "." not in printed:
        return 0.75            # integers: allow rounding of the mean
    return 10 ** (-len(printed.split(".")[1])) * 0.85


NUM = re.compile(r"[+-]?\d+(?:\.\d+)?$")
EMPTY = ("", "---", "--", "-")


def _clean(tok):
    """Strip formatting wrappers so only the numeral remains."""
    for junk in (r"\textbf", r"\mathbf", r"\emph", r"\textit",
                 r"\,", r"\;", r"\!", r"$", "{", "}", "~", " "):
        tok = tok.replace(junk, "")
    return tok.strip()


def parse_cell(cell):
    """Return (value, printed_repr, std_or_None), or None if not a number.

    Strict: the whole cell must be a numeral, optionally "a$\\pm$b".
    Anything containing words (a header like "Cost/1k") is rejected, which
    is what keeps header rows out of the per-fold average.
    """
    cell = cell.strip()
    if _clean(cell) in EMPTY:
        return None
    parts = cell.split(r"\pm")
    if len(parts) > 2:
        return None
    head = _clean(parts[0])
    if not NUM.match(head):
        return None
    std = None
    if len(parts) == 2:
        tail = _clean(parts[1])
        if not NUM.match(tail):
            return None
        std = float(tail)
    return float(head), head, std


def is_data_row(cells):
    """A data row: every cell is numeric, empty, or the row label."""
    numeric = 0
    for c in cells[1:]:
        if _clean(c) in EMPTY:
            continue
        if parse_cell(c) is None:
            return False
        numeric += 1
    return numeric >= 1


def mean(xs):
    return sum(xs) / len(xs)


def sample_std(xs):
    if len(xs) < 2:
        return 0.0
    mu = mean(xs)
    return (sum((x - mu) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


RULE_RE = re.compile(
    r"\\(?:top|mid|bottom)rule|"
    r"\\cmidrule\s*(?:\([^)]*\))?\s*\{[^}]*\}|"
    r"\\addlinespace(?:\[[^\]]*\])?|"
    r"\\renewcommand\{[^}]*\}\{[^}]*\}")


def logical_rows(body):
    """Split a tabular body into logical rows (a row may wrap lines).

    Horizontal rules are stripped rather than treated as rows, since they
    sit on their own lines and would otherwise be glued onto the next row.
    """
    rows, pending = [], ""
    for line in body.split("\n"):
        line = RULE_RE.sub("", line.split("%")[0]).strip()
        if not line:
            continue
        pending = (pending + " " + line).strip()
        if pending.endswith(r"\\"):
            rows.append(pending[:-2].strip())
            pending = ""
    return rows


SKIP = (r"\multicolumn", r"\multirow")

problems, checked = [], 0


def check_block(fn, data, mrow, ncol):
    """Compare one Mean row against the data rows above it."""
    n = 0
    for c in range(1, ncol):
        stated = parse_cell(mrow[c])
        if stated is None:
            continue
        vals = [parse_cell(d[c]) for d in data]
        if any(v is None for v in vals):
            continue
        vals = [v[0] for v in vals]
        n += 1

        got, printed, stated_std = mean(vals), stated[1], stated[2]
        if abs(got - stated[0]) > tol_for(printed):
            problems.append(
                "%s col %d: printed mean %s but %d rows average %.4f"
                % (fn, c + 1, printed, len(vals), got))
        if stated_std is not None:
            gs = sample_std(vals)
            # tolerance from the std's own printed precision
            st = tol_for(_clean(mrow[c].split(r"\pm")[1]))
            if abs(gs - stated_std) > st:
                problems.append(
                    "%s col %d: printed std %.4f but sample std is %.4f"
                    % (fn, c + 1, stated_std, gs))
    return n


for fn in sorted(os.listdir(ROOT)):
    if not fn.endswith(".tex"):
        continue
    text = io.open(os.path.join(ROOT, fn), encoding="utf-8").read()
    m = re.search(r"\\begin\{tabular\}.*?\n(.*?)\\end\{tabular\}",
                  text, re.S)
    if not m:
        continue

    rows = [r for r in logical_rows(m.group(1))
            if r and not any(x in r for x in SKIP)]

    # A table may hold several blocks, each ending in its own Mean row.
    mean_idxs = [i for i, r in enumerate(rows)
                 if re.search(r"\bMean\b", r.split("&")[0], re.I)]
    blocks, start = [], 0
    for mi in mean_idxs:
        if mi > start:
            blocks.append((start, mi))
        start = mi + 1
    if not blocks:
        continue

    for lo, mean_idx in blocks:
        mrow = rows[mean_idx].split("&")
        ncol = len(mrow)
        data = [r.split("&") for r in rows[lo:mean_idx]]
        data = [d for d in data if len(d) == ncol and is_data_row(d)]
        if len(data) < 2:
            continue
        checked += check_block(fn, data, mrow, ncol)


print("checked %d mean/std cells across tables/" % checked)
if problems:
    print("\nFAILED -- %d mismatch(es):" % len(problems))
    for p in problems:
        print("  * " + p)
    sys.exit(1)
print("OK -- every printed Mean row agrees with its per-fold rows.")
