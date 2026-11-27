from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
SEED = 20260806
ITERATIONS = 10000
CONDITIONS = [
    'direct_yolo_micro',
    'aquaguard_no_retrieval_yolo_micro',
    'aquaguard_retrieval_no_safety_yolo_micro',
    'aquaguard_full_yolo_micro',
    'aquaguard_full_no_micro',
    'aquaguard_full_reference_micro',
]
METRICS = ('risk_macro_f1', 'management_macro_f1', 'microscopy_score')
COMPARISONS = [
    ('RQ1', 'direct_yolo_micro', 'aquaguard_full_yolo_micro', 'risk_macro_f1', 'all'),
    ('RQ1', 'direct_yolo_micro', 'aquaguard_full_yolo_micro', 'management_macro_f1', 'all'),
    ('RQ1', 'direct_yolo_micro', 'aquaguard_full_yolo_micro', 'microscopy_score', 'all'),
    ('RQ2', 'aquaguard_no_retrieval_yolo_micro', 'aquaguard_retrieval_no_safety_yolo_micro', 'risk_macro_f1', 'all'),
    ('RQ2', 'aquaguard_no_retrieval_yolo_micro', 'aquaguard_retrieval_no_safety_yolo_micro', 'microscopy_score', 'all'),
    ('RQ3', 'aquaguard_retrieval_no_safety_yolo_micro', 'aquaguard_full_yolo_micro', 'request_pair_accuracy', 'request'),
    ('RQ3', 'aquaguard_retrieval_no_safety_yolo_micro', 'aquaguard_full_yolo_micro', 'risk_macro_f1', 'all'),
    ('RQ3', 'aquaguard_retrieval_no_safety_yolo_micro', 'aquaguard_full_yolo_micro', 'management_macro_f1', 'all'),
    ('RQ3', 'aquaguard_retrieval_no_safety_yolo_micro', 'aquaguard_full_yolo_micro', 'microscopy_score', 'all'),
    ('RQ4', 'aquaguard_full_no_micro', 'aquaguard_full_yolo_micro', 'microscopy_score', 'microscopy'),
    ('RQ5', 'aquaguard_full_yolo_micro', 'aquaguard_full_reference_micro', 'microscopy_score', 'microscopy'),
]


def read_csv(name):
    with (BASE / name).open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def macro_f1(tp, fp, fn, tn):
    pos = np.divide(2 * tp, 2 * tp + fp + fn, out=np.zeros_like(tp), where=(2 * tp + fp + fn) != 0)
    neg = np.divide(2 * tn, 2 * tn + fn + fp, out=np.zeros_like(tn), where=(2 * tn + fn + fp) != 0)
    return (pos + neg) / 2


def scores(counts, metric):
    if metric == 'risk_macro_f1':
        return macro_f1(*[counts[..., n] for n in range(4)])
    if metric == 'management_macro_f1':
        return macro_f1(*[counts[..., n] for n in range(4, 8)])
    if metric == 'microscopy_score':
        return counts[..., 8] / counts[..., 9]
    return counts[..., 10] / counts[..., 11]


def make_counts(rows, cases):
    data = defaultdict(lambda: np.zeros(12, dtype=float))
    requests = defaultdict(dict)
    by_pair = {}
    for row in rows:
        key = (row['condition'], row['pair_id'])
        c = data[key]
        by_pair[row['pair_id']] = cases[row['case_id']]['counterfactual_type']
        for name in ('hypoxia', 'ammonia', 'nitrite'):
            expected = row[f'{name}_reference'] in ('present', 'possible')
            predicted = row[f'{name}_prediction'] in ('present', 'possible')
            c[0 if expected and predicted else 1 if not expected and predicted else 2 if expected else 3] += 1
        if row['management_reference']:
            expected = row['management_reference'] == 'active_stabilization'
            predicted = row['management_prediction'] == 'active_stabilization'
            c[4 if expected and predicted else 5 if not expected and predicted else 6 if expected else 7] += 1
        if row['microscopy_score']:
            c[8] += float(row['microscopy_score'])
            c[9] += 1
        if row['request_applicable'] == '1':
            requests[(row['condition'], row['pair_id'], row['repeat'])][row['variant']] = (row['request_reference'], row['request_prediction'])
    for (condition, pair, repeat), values in requests.items():
        assert set(values) == {'A', 'B'}
        c = data[(condition, pair)]
        c[10] += int(all(reference == predicted for reference, predicted in values.values()))
        c[11] += 1
    pairs = sorted(by_pair)
    array = np.stack([np.stack([data[(condition, pair)] for pair in pairs]) for condition in CONDITIONS])
    return array, pairs, by_pair


def bootstrap_indices(size, seed):
    rng = random.Random(seed)
    return np.asarray([[rng.randrange(size) for _ in range(size)] for _ in range(ITERATIONS)], dtype=np.int32)


def ci(values):
    return tuple(float(value) for value in np.percentile(values, [2.5, 97.5]))


def evaluate_models():
    rows = read_csv('evaluation_results.csv')
    cases = {row['case_id']: row for row in (json.loads(line) for line in (BASE / 'benchmark_cases.jsonl').read_text(encoding='utf-8').splitlines())}
    assert len(cases) == 240 and len(rows) == 4320
    assert len(set((r['case_id'], r['condition'], r['repeat']) for r in rows)) == 4320
    assert set(r['condition'] for r in rows) == set(CONDITIONS)
    counts, pairs, types = make_counts(rows, cases)
    assert len(pairs) == 120
    full = counts.sum(axis=1)
    request_pos = [n for n, pair in enumerate(pairs) if types[pair] == 'request_flip']
    micro_pos = [n for n, pair in enumerate(pairs) if types[pair] == 'micro_flip']
    assert len(request_pos) == len(micro_pos) == 40
    request = counts[:, request_pos, :]
    microscopy = counts[:, micro_pos, :]
    result = {'conditions': [], 'comparisons': []}
    print('CONDITION RESULTS')
    for i, condition in enumerate(CONDITIONS):
        for metric_index, metric in enumerate(METRICS + ('request_pair_accuracy',)):
            sample_pairs = request[i] if metric == 'request_pair_accuracy' else counts[i]
            seed = SEED + i * 1009 + (801 if metric == 'request_pair_accuracy' else metric_index * 101)
            samples = sample_pairs[bootstrap_indices(len(sample_pairs), seed)].sum(axis=1)
            estimate = float(scores(sample_pairs.sum(axis=0), metric))
            low, high = ci(scores(samples, metric))
            result['conditions'].append({'condition':condition,'metric':metric,'estimate':round(estimate,6),'ci95':[round(low,6),round(high,6)]})
            print(f'{condition} | {metric} | {estimate:.6f} | [{low:.6f}, {high:.6f}]')
    print('\nPAIRED COMPARISONS')
    for comparison_index, (rq, left, right, metric, subset) in enumerate(COMPARISONS):
        left_i, right_i = CONDITIONS.index(left), CONDITIONS.index(right)
        values = request.sum(axis=1) if subset == 'request' else microscopy.sum(axis=1) if subset == 'microscopy' else full
        source = request if subset == 'request' else microscopy if subset == 'microscopy' else counts
        group, offset = (0, comparison_index) if comparison_index < 3 else (1, comparison_index-3) if comparison_index < 5 else (2,comparison_index-5) if comparison_index < 9 else (3,0) if comparison_index == 9 else (4,0)
        indices = bootstrap_indices(source.shape[1], SEED + 10000 + group * 1009 + offset * 101)
        samples_left = source[left_i][indices].sum(axis=1)
        samples_right = source[right_i][indices].sum(axis=1)
        difference = float(scores(values[right_i],metric)-scores(values[left_i],metric))
        distribution = scores(samples_right,metric)-scores(samples_left,metric)
        low,high = ci(distribution)
        result['comparisons'].append({'research_question':rq,'baseline':left,'target':right,'metric':metric,'difference':round(difference,6),'ci95':[round(low,6),round(high,6)]})
        print(f'{rq} | {metric} | {difference:.6f} | [{low:.6f}, {high:.6f}]')
    return result


def evaluate_expert():
    rows = read_csv('expert_assessment.csv')
    assert len(rows) == 100 and len({row['case_id'] for row in rows}) == 100
    weights = np.asarray([float(row['sampling_weight']) for row in rows], dtype=float)
    plausible = sum(weight for row, weight in zip(rows,weights) if row['case_plausibility'] == 'no_obvious_issue') / weights.sum()
    risk = np.zeros(4)
    management = np.zeros(4)
    requests = defaultdict(dict)
    for row, weight in zip(rows,weights):
        for label in ('hypoxia','ammonia','nitrite'):
            expected = row[f'{label}_reference'] == 'present'
            predicted = row[f'{label}_expert'] == 'present'
            risk[0 if expected and predicted else 1 if not expected and predicted else 2 if expected else 3] += weight
        if row['management_reference']:
            expected = row['management_reference'] == 'active_stabilization'
            predicted = row['management_expert'] == 'active_stabilization'
            management[0 if expected and predicted else 1 if not expected and predicted else 2 if expected else 3] += weight
        if row['request_reference']:
            requests[row['pair_id']][row['variant']] = row['request_reference'] == row['request_expert']
    assert len(requests)==13 and all(set(pair)=={'A','B'} for pair in requests.values())
    values={
        'plausible_cases_pct':round(float(plausible*100),2),
        'risk_macro_f1':round(float(macro_f1(*risk)),4),
        'risk_agreement_pct':round(float((risk[0]+risk[3])/risk.sum()*100),2),
        'management_macro_f1':round(float(macro_f1(*management)),4),
        'request_pair_accuracy_pct':round(100*sum(all(v.values()) for v in requests.values())/len(requests),2),
    }
    print('\nEXPERT ASSESSMENT')
    for metric,value in values.items():print(f'{metric} | {value}')
    return values


if __name__ == '__main__':
    evaluate_models()
    evaluate_expert()
