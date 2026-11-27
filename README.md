# AquaDecision-Bench

This dataset supports the controlled evaluation of AquaGuard and the expert assessment reported in the manuscript.

## Files

| File | Contents |
|---|---|
| `benchmark_cases.jsonl` | 240 semi-synthetic aquaculture cases in 120 counterfactual A/B pairs, including case inputs, microscopy summaries, image identifiers, and task reference labels. |
| `evaluation_results.csv` | 4,320 run-level evaluation records for six experimental conditions and three runs per case. |
| `expert_assessment.csv` | Expert judgments for 100 sampled cases (50 A/B pairs), with stratum weights and the corresponding reference labels. |
| `verify.py` | Recomputes evaluation metrics, paired comparisons, 95% confidence intervals, and expert-assessment metrics. |

## Evaluation fields

Each row of `evaluation_results.csv` identifies a case, experimental condition, and repeat. The risk fields contain reference and predicted stances for hypoxia, ammonia, and nitrite. The management fields contain reference and predicted management intensity; blank values indicate that the metric does not apply. `microscopy_score` is the fraction of applicable microscopy criteria satisfied for that run. Request-handling predictions are evaluated jointly for both sides of each request A/B pair.

Each row of `expert_assessment.csv` contains one case assessment. `sampling_weight` represents the stratum expansion weight (population count divided by sample count). Risk agreement and binary Macro-F1 pool the three water-quality judgments. Management-intensity Macro-F1 uses the 74 applicable cases. Request-handling pair accuracy uses 13 complete A/B pairs. The file contains the expert judgments used to compute the reported results.

## Reproduce the results

Use Python 3.10+ with NumPy installed:

```bash
python -m pip install numpy
python verify.py
```

The script reports the six-condition metrics, RQ1–RQ5 paired comparisons, pair-cluster bootstrap 95% confidence intervals (10,000 iterations; random seed 20260806), and the five expert-assessment values.

Image identifiers refer to the EMDS-7 test split. Image binaries and full generated response texts are not included; structured inputs and evaluation records needed for these reported scores are provided.
