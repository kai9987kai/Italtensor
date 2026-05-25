# Italtensor Experiment Report

Generated: 2026-05-25T07:21:15.581940+00:00

## Dataset
- Samples: 4
- Input dimension: 2
- Class counts: {'0': 2, '1': 2}

## Dataset Audit
- Imbalance ratio: 1.0000
- Duplicate rows: 1
- Duplicate groups: 1
- Label conflicts: 0
- Conflicting rows: 0
- Constant features: []
- High correlation pairs: 1
- Warnings: ['very small dataset', 'duplicate feature rows', 'highly correlated features']

## Model
- Config: {'hidden_layers': [16], 'learning_rate': 0.001, 'batch_size': 16, 'max_epochs': 3, 'patience': 5, 'random_seed': 42, 'feature_map': 'linear', 'rff_components': 64, 'rff_gamma': 1.0, 'l1_penalty': 0.0, 'feature_selection_k': None, 'lr_schedule': 'constant', 'gradient_clip': 0.0, 'backend': 'auto', 'mps_bond_dim': 8, 'mps_physical_dim': 4}
- Decision threshold: 0.4000

## Metrics
- f1: 0.7500
- threshold: 0.4000

## Uncertainty
- conformal_source: dedicated_calibration
- conformal_alpha: 0.1000
- conformal_quantile: 0.3500
- conformal_target_coverage: 0.9000
- conformal_coverage: 1.0000
- conformal_calibration_count: 8
- conformal_evaluation_count: 8
- conformal_singleton_rate: 0.7500

## Post-Hoc Conformal Diagnostics
- Split source: posthoc_stratified_split
- Calibration rows: 4
- Evaluation rows: 4
- Recommended alpha: 0.1000
- Target coverage: 0.9000
- Empirical coverage: 1.0000
- Mean set size: 1.2500
- Singleton rate: 0.7500
- Ambiguous rate: 0.2500
- Warning: none
- alpha=0.1000: target=0.9000, coverage=1.0000, gap=0.1000, mean_size=1.2500, singleton_acc=1.0000

## Post-Hoc Calibration Repair
- Split source: posthoc_stratified_split
- Calibration rows: 4
- Evaluation rows: 4
- Recommended method: platt
- Recommended Brier: 0.1000
- Recommended ECE: 0.0500
- Recommended log loss: 0.4000
- Brier improvement: 0.0800
- ECE improvement: 0.0300
- Warning: none
- raw: Brier=0.1800, ECE=0.0800, logloss=0.5000, dBrier=0.0000
- platt: Brier=0.1000, ECE=0.0500, logloss=0.4000, dBrier=0.0800

## Top Feature Importances
- Feature 0: importance=0.2500

## Ablation Diagnostics
- Base F1: 0.7500
- Top feature: x1
- Max F1 drop: 0.2500
- Max label flip rate: 0.2000
- High-reliance features: 1
- Label-proxy flags: 1
- Feature 0: drop=0.2500, perm_drop=0.2000, flip=0.2000, corr=0.9000, flags=label_proxy

## Model Response / Partial Dependence
- Top feature: 0
- Top response range: 0.5000
- Top direction: increasing
- Nonmonotonic features: 1
- High-impact features: 2
- Warning: none
- Feature 0: range=0.5000, change=0.4500, direction=increasing, min_at=-1.0000, max_at=1.0000, flags=high_impact

## Pairwise Feature Interactions
- Evaluated pairs: 1
- Top pair: x1:x2
- Top interaction strength: 0.5500
- Top max absolute interaction: 0.2200
- Strong pairs: 1
- Threshold-crossing pairs: 1
- Warning: none
- x1:x2: H=0.5500, max_abs=0.2200, mean_abs=0.1000, crossings=2, flags=strong_interaction

## Sample Review
- Label issues: 1
- Disagreements: 2
- Ambiguous rows: 1
- Mean loss: 0.3000
- Max loss: 1.2000
- label_issue row 2: label=0, pred=1, p=0.9500, loss=2.9000

## Threshold Tradeoffs
- Current threshold: 0.4000
- Best F1 threshold: 0.3000
- Best balanced-accuracy threshold: 0.3500
- Minimum-cost threshold: 0.2500
- Current cost: 0.5000
- Minimum cost: 0.2500
- best_f1: t=0.3000, F1=0.8000, precision=0.7500, recall=0.8500, cost=0.3000
- best_balanced_accuracy: t=0.3500, F1=0.7500, precision=0.7000, recall=0.8000, cost=0.4000
- min_cost: t=0.2500, F1=0.7000, precision=0.6500, recall=0.9000, cost=0.2500

## Decision Curve / Utility
- Prevalence: 0.5000
- Best threshold: 0.4000
- Best net benefit: 0.2500
- Max gain vs best default: 0.2000
- Useful threshold ranges: 0.2000-0.6000
- Current threshold: 0.4000
- Current net benefit: 0.2500
- Current gain vs best default: 0.2000
- Warning: none
- t=0.4000: model=0.2500, all=0.1000, none=0.0000, gain=0.1500

## Selective Prediction / Risk-Coverage
- Base risk: 0.5000
- Minimum selective risk: 0.0000
- Recommended cutoff: 0.2000
- Best selective accuracy: 1.0000
- Best selective coverage: 0.5000
- Error reduction: 0.5000
- Coverage at 10 pct risk: 0.5000
- AURC: 0.1000
- Warning: none
- cutoff=0.2000: coverage=0.5000, risk=0.0000, accuracy=1.0000, F1=1.0000

## Slice Diagnostics
- Base F1: 0.7500
- Slice count: 1
- Worst slice: x1[0, 1]
- Worst F1 delta: -0.2500
- Worst accuracy delta: -0.2500
- x1[0.0000, 1.0000]: n=2, F1=0.5000, delta=-0.2500

## Subgroup Disparity Diagnostics
- Evaluated features: 1
- Evaluated subgroups: 2
- Worst feature: 1
- Worst subgroup: x2=1
- Worst metric: false_negative_rate_gap
- Max disparity: 0.6000
- Max FNR gap: 0.6000
- Max FPR gap: 0.2000
- Max selection-rate gap: 0.3000
- Warning: Numeric feature slices are proxy subgroup diagnostics.
- x2=1: n=4, coverage=0.5000, gap=0.6000, metric=false_negative_rate_gap, flags=fnr_gap

## Robustness Stress Lab
- Base F1: 0.7500
- Worst F1: 0.5000
- Stress F1 ratio: 0.6667
- Max label flip rate: 0.2500
- Worst case: feature_dropout@0.25
- feature_dropout@0.2500: F1=0.5000, flip=0.2500

## Trial History
- Trial 1: map=rff, f1=0.7500, brier=0.1000, log_loss=0.3000
