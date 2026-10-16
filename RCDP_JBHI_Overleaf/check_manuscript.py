#!/usr/bin/env python3
r"""Structural pre-flight check for the RCDP manuscript.

Runs without a LaTeX installation. Verifies:
  1. every \input target exists
  2. every \ref / \eqref has a matching \label
  3. every \cite key exists in refs.bib (and lists unused entries)
  4. brace and math-mode delimiters balance per file
  5. tabular column counts match the declared column spec
  6. duplicate labels

Usage:  python check_manuscript.py
"""
import io
import os
import re
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.abspath(__file__))
MAIN = os.path.join(ROOT, "main.tex")

problems = []
notes = []


def read(path):
    with io.open(path, encoding="utf-8") as fh:
        return fh.read()


def is_escaped(text, i):
    """True if text[i] is preceded by an odd number of backslashes."""
    n = 0
    j = i - 1
    while j >= 0 and text[j] == "\\":
        n += 1
        j -= 1
    return n % 2 == 1


def strip_comments(text):
    """Drop LaTeX comments but keep escaped percent signs."""
    out = []
    for line in text.split("\n"):
        idx = None
        for i, ch in enumerate(line):
            if ch == "%" and not is_escaped(line, i):
                idx = i
                break
        out.append(line if idx is None else line[:idx])
    return "\n".join(out)


# --------------------------------------------------------------- collect files
def resolve_inputs(path, seen):
    r"""Recursively expand \input targets, returning list of (path, text)."""
    if path in seen:
        return []
    seen.add(path)
    raw = strip_comments(read(path))
    collected = [(path, raw)]
    for target in re.findall(r"\\input\{([^}]+)\}", raw):
        cand = os.path.join(ROOT, target)
        if not cand.endswith(".tex"):
            cand += ".tex"
        if not os.path.exists(cand):
            problems.append("MISSING \\input target: %s (from %s)"
                            % (target, os.path.basename(path)))
            continue
        collected.extend(resolve_inputs(cand, seen))
    return collected


files = resolve_inputs(MAIN, set())
corpus = "\n".join(text for _, text in files)
print("expanded %d source files" % len(files))

# ------------------------------------------------------------- graphics files
for inc in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", corpus):
    cand = os.path.join(ROOT, inc)
    if not (os.path.exists(cand) or os.path.exists(cand + ".pdf")
            or os.path.exists(cand + ".png")):
        problems.append("MISSING graphic: %s" % inc)

# -------------------------------------------------------------- labels / refs
labels = re.findall(r"\\label\{([^}]+)\}", corpus)
dupes = [k for k, v in Counter(labels).items() if v > 1]
for d in dupes:
    problems.append("DUPLICATE \\label: %s" % d)
labelset = set(labels)

refs = set()
for macro in ("ref", "eqref", "autoref"):
    refs |= set(re.findall(r"\\%s\{([^}]+)\}" % macro, corpus))
for r in sorted(refs - labelset):
    problems.append("DANGLING \\ref: %s" % r)

unused_labels = sorted(labelset - refs)
if unused_labels:
    notes.append("labels never referenced (harmless): %s"
                 % ", ".join(unused_labels))

# ---------------------------------------------------------------- bib keys
bib = read(os.path.join(ROOT, "refs.bib"))
bibkeys = set(re.findall(r"@\w+\s*\{\s*([^,\s]+)\s*,", bib))
cited = set()
for group in re.findall(r"\\cite\{([^}]+)\}", corpus):
    cited |= {k.strip() for k in group.split(",") if k.strip()}

for k in sorted(cited - bibkeys):
    problems.append("CITED BUT NOT IN refs.bib: %s" % k)
unused_bib = sorted(bibkeys - cited)
if unused_bib:
    problems.append("IN refs.bib BUT NEVER CITED (will not print): %s"
                    % ", ".join(unused_bib))

print("labels=%d  refs=%d  bib entries=%d  cited=%d"
      % (len(labelset), len(refs), len(bibkeys), len(cited)))

# ------------------------------------------------------------ delimiter check
for path, text in files:
    name = os.path.relpath(path, ROOT)
    depth = 0
    for i, ch in enumerate(text):
        if ch in "{}" and not is_escaped(text, i):
            depth += 1 if ch == "{" else -1
            if depth < 0:
                problems.append("%s: unbalanced '}' at offset %d" % (name, i))
                break
    if depth > 0:
        problems.append("%s: %d unclosed '{'" % (name, depth))

    dollars = len([1 for i, ch in enumerate(text)
                   if ch == "$" and not is_escaped(text, i)])
    if dollars % 2:
        problems.append("%s: odd number of '$' (%d)" % (name, dollars))

    for env in ("table", "table*", "figure", "figure*", "tabular",
                "algorithm", "algorithmic", "tikzpicture", "itemize",
                "enumerate", "equation", "multline"):
        esc = env.replace("*", r"\*")
        b = len(re.findall(r"\\begin\{%s\}" % esc, text))
        e = len(re.findall(r"\\end\{%s\}" % esc, text))
        if b != e:
            problems.append("%s: \\begin{%s}=%d but \\end{%s}=%d"
                            % (name, env, b, env, e))

# ------------------------------------------------------------- tabular arity
COLSPEC = re.compile(r"\\begin\{tabular\}\s*(?:\[[^\]]*\])?\s*\{(.+?)\}\s*\n",
                     re.S)


def count_cols(spec):
    """Count column slots in a tabular preamble."""
    spec = re.sub(r"@\{[^}]*\}", "", spec)
    spec = re.sub(r"p\{[^}]*\}", "p", spec)
    spec = re.sub(r"[|\s]", "", spec)
    return len([c for c in spec if c in "lcrp"])


for path, text in files:
    name = os.path.relpath(path, ROOT)
    for m in COLSPEC.finditer(text):
        ncol = count_cols(m.group(1))
        body = text[m.end():]
        end = body.find(r"\end{tabular}")
        body = body[:end if end != -1 else len(body)]
        # A logical row may span several physical lines; accumulate to '\\'.
        pending = ""
        for line in body.split("\n"):
            pending = (pending + " " + line.strip()).strip()
            if not pending.endswith(r"\\"):
                continue
            s, pending = pending, ""
            if any(t in s for t in (r"\multicolumn", r"\cmidrule", r"\midrule",
                                    r"\toprule", r"\bottomrule", r"\multirow",
                                    r"\addlinespace")):
                continue
            cells = s[:-2].count("&") + 1
            if cells != ncol:
                problems.append("%s: row has %d cells, spec declares %d: %s"
                                % (name, cells, ncol, s[:70]))

# ------------------------------------------------------------------- report
print()
if notes:
    for n in notes:
        print("note: " + n)
    print()
if problems:
    print("FAILED -- %d issue(s):" % len(problems))
    for p in problems:
        print("  * " + p)
    sys.exit(1)
print("OK -- no structural issues found.")
