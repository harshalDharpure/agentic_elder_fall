#!/usr/bin/env python3
"""Audit author-supplied confusion counts; never run experiments or write files.

Reports fold means/sample standard deviations, validates count totals and
within-evaluation label totals, and checks selected manuscript aggregate rows.
Counts were transcribed from the author's 2026-09-24 tables, not raw predictions.
"""
import csv
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parent
METRICS = ('f1', 'recall', 'precision', 'specificity', 'accuracy', 'cost')


def load_counts():
    with (ROOT / 'results_text/reported_counts.csv').open(encoding='utf-8', newline='') as handle:
        records = []
        for row in csv.DictReader(handle):
            records.append({key: int(value) if key in ('fold', 'n', 'tp', 'tn', 'fp', 'fn') else value
                            for key, value in row.items()})
    return records


def metrics(row):
    tp, tn, fp, fn, n = (row[key] for key in ('tp', 'tn', 'fp', 'fn', 'n'))
    return dict(zip(METRICS, (
        2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        tp / (tp + fn) if tp + fn else 0.0,
        tp / (tp + fp) if tp + fp else 0.0,
        tn / (tn + fp) if tn + fp else 0.0,
        (tp + tn) / n,
        1000 * (10 * fn + fp) / n,
    )))


def grouped_metrics(records):
    groups = defaultdict(list)
    for row in sorted(records, key=lambda item: item['fold']):
        groups[(row['evaluation'], row['method'])].append(metrics(row))
        # A constant prediction is determined by class totals: no model rerun.
        if row['evaluation'] == 'llm_shared' and row['method'] == 'mistral':
            constant = dict(row, tp=row['tp'] + row['fn'], fn=0,
                            fp=row['tn'] + row['fp'], tn=0)
            groups[('llm_shared', 'always_fall_derived')].append(metrics(constant))
    return groups


def aggregate(groups, key, metric):
    values = [row[metric] for row in groups[key]]
    return mean(values), stdev(values)


def check_latex_row(filename, label, expected):
    """Check a one-line row of numeric cells, optionally containing stds."""
    text = (ROOT / 'tables' / filename).read_text(encoding='utf-8')
    matches = [line for line in text.splitlines() if line.partition('&')[0].strip() == label]
    assert len(matches) == 1, (filename, label, 'row missing or duplicated')
    cells = matches[0].split('&')[1:]
    assert len(cells) == len(expected), (filename, label, 'column count')
    for cell, values in zip(cells, expected):
        numbers = re.findall(r'(?<![A-Za-z])\d+(?:\.\d+)?', cell)
        assert len(numbers) == len(values), (filename, label, cell)
        for printed, value in zip(numbers, values):
            decimals = len(printed.partition('.')[2])
            tolerance = 0.5 * 10 ** (-decimals) + 1e-10
            assert abs(float(printed) - value) <= tolerance, (filename, label, printed, value)


def main():
    records = load_counts()
    seen, class_totals = set(), {}
    for row in records:
        key = (row['evaluation'], row['method'], row['fold'])
        assert key not in seen, ('duplicate record', key)
        seen.add(key)
        assert row['n'] > 0 and all(row[c] >= 0 for c in ('tp', 'tn', 'fp', 'fn'))
        assert sum(row[c] for c in ('tp', 'tn', 'fp', 'fn')) == row['n'], key
        label_key = (row['evaluation'], row['fold'])
        totals = (row['tp'] + row['fn'], row['tn'] + row['fp'])
        assert class_totals.setdefault(label_key, totals) == totals, label_key
    groups = grouped_metrics(records)
    for key in sorted(groups):
        assert len(groups[key]) == 5, ('five folds required', key)
        print('/'.join(key) + ': ' + ' | '.join(
            f'{metric} {aggregate(groups, key, metric)[0]:.6f} +/- '
            f'{aggregate(groups, key, metric)[1]:.6f}' for metric in METRICS))
    labels = {'always_fall': 'Always fall', 'always_adl': 'Always ADL',
              'random': 'Random', 'majority_train': 'Training majority',
              'heuristic': 'Heuristic', 'logreg': 'Logistic regression', 'gbm': 'Gradient boosting'}
    for method, label in labels.items():
        expected = [aggregate(groups, ('g2_features', method), metric)
                    for metric in ('f1', 'recall', 'specificity', 'cost')]
        check_latex_row('tab_adjudicator.tex', label, expected)
    for method, label in [('always_fall_derived', 'Always fall (derived)'),
                          ('heuristic', 'Heuristic (LLM set)'),
                          ('mistral', 'Mistral-7B'), ('qwen_coder', 'Qwen2.5-Coder-7B')]:
        expected = [aggregate(groups, ('llm_shared', method), metric)
                    for metric in ('f1', 'recall', 'specificity', 'cost')]
        check_latex_row('tab_adjudicator.tex', label, expected)
    for method in ('tier1_only', 'gate_only', 'gate_knn', 'gate_llm', 'gate_knn_llm'):
        expected = [aggregate(groups, ('main_ablation', method), metric)
                    for metric in ('f1', 'recall', 'cost')]
        check_latex_row('tab_ablation_mean.tex', r'\texttt{' + method.replace('_', r'\_') + '}', expected)
    backbone_labels = {'cnn1d': 'CNN1D', 'lstm': 'LSTM', 'cnn_lstm': 'CNN--LSTM',
                       'cnn_lstm_attn': 'CNN--LSTM--Attention', 'tcn': 'TCN',
                       'modern_tcn': 'ModernTCN', 'tsmixer': 'TSMixer',
                       'patchtst': 'PatchTST', 'transformer': 'Transformer'}
    with (ROOT / 'results_text/backbone_f1.csv').open(encoding='utf-8', newline='') as handle:
        for row in csv.DictReader(handle):
            expected = []
            for phase in ('e1', 'retrained'):
                values = [float(row[f'{phase}_fold{i}']) for i in range(5)]
                expected.append((mean(values), stdev(values)))
            check_latex_row('tab_backbone.tex', backbone_labels[row['model']], expected)
    # Verify the displayed primary count rows, not just the mean summaries.
    text = (ROOT / 'tables/tab_confusion.tex').read_text(encoding='utf-8')
    displayed = [line.split('&') for line in text.splitlines() if re.match(r'^[0-4]\s*&', line)]
    expected_rows = [row for method in ('tier1_only', 'gate_knn_llm') for row in records
                     if row['evaluation'] == 'main_ablation' and row['method'] == method]
    assert len(displayed) == len(expected_rows) == 10
    for cells, row in zip(displayed, expected_rows):
        assert [int(c.strip()) for c in cells[:5]] == [row[k] for k in ('fold', 'tp', 'fp', 'fn', 'tn')]
    print(f'OK -- {len(records)} supplied count rows; 25 aggregate rows and 10 confusion rows checked.')
    print('Class-total agreement does not establish case identity or absence of leakage.')


if __name__ == '__main__':
    main()