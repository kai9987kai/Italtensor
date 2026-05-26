# Italtensor Experiment Report

Generated: 2026-05-26T18:43:59.585868+00:00

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

## Post-Hoc Permutation-Null Diagnostic
- Permutations: 80
- Seed: 42
- Verdict: strong_signal
- Observed F1: 0.9000
- Null mean F1: 0.4500
- F1 gap: 0.4500
- F1 z-score: 3.1000
- F1 p-value: 0.0100
- Accuracy p-value: 0.0200
- Warning: none
- f1: observed=0.9000, mean=0.4500, p95=0.7000, p=0.0100
- accuracy: observed=0.8500, mean=0.5000, p95=0.7500, p=0.0200
- balanced_accuracy: observed=0.8400, mean=0.5000, p95=0.7400, p=0.0300

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

## Population Drift Diagnostics
- Split source: row_order_first_reference_then_current
- Reference rows: 2
- Current rows: 2
- Top feature: x2
- Max PSI: 0.4000
- Max KS statistic: 0.5000
- Max mean shift: 1.2000
- Max outside-reference rate: 0.2500
- Drifted features: 1
- Label prevalence shift: 0.2500
- Warning: none
- x2: PSI=0.4000, KS=0.5000, mean_shift=1.2000, outside=0.2500, flags=major_psi_shift

## Adversarial Validation Diagnostics
- Split source: row_order_domain_classifier
- Reference rows: 2
- Current rows: 2
- Validation rows: 2
- Domain AUC: 0.8800
- Domain accuracy: 0.8000
- Detectability: 0.8800
- Verdict: strong_multivariate_shift
- Top feature: x2
- Important features: 1
- Label prevalence shift: 0.2500
- Warning: none
- x2: auc_drop=0.2000, accuracy_drop=0.1500, prob_shift=0.1200, flags=domain_auc_driver

## Chronological Holdout Diagnostics
- Split source: row_order_reference_then_current
- Reference rows: 3
- Reference evaluation rows: 1
- Current rows: 2
- Feature map: linear
- Threshold: -
- Reference F1: 0.9000
- Current F1: 0.5000
- F1 delta: -0.4000
- Accuracy delta: -0.3000
- Brier delta: 0.1500
- Log loss delta: 0.4000
- Mean probability delta: 0.2000
- Current ECE: -
- Label prevalence shift: 0.2500
- Top current reliance feature: x2
- Current-baseline F1 gain: 0.2000
- Verdict: severe_temporal_degradation_current_relearns
- Warning: none
- Current-baseline train rows: 4
- Current-baseline evaluation rows: 2
- Current-baseline F1: 0.7000
- Current-baseline F1 gain vs reference model: 0.2000
- x2: F1_drop=0.2000, logloss_increase=0.1000, prob_shift=0.1500, flags=current_f1_driver

## Dataset Cartography
- Samples: 4
- Threshold: 0.4000
- Median confidence: 0.7500
- Median variability: 0.0500
- Easy rows: 2
- Ambiguous rows: 1
- Hard rows: 1
- Overconfident wrong rows: 0
- ambiguous row 1: label=0, pred=1, conf=0.4500, var=0.2000

## Neighborhood Hardness
- Rows scanned: 4
- k: 3
- Leave-one-out accuracy: 0.7500
- Hard rows: 1
- Ambiguous rows: 1
- Label issue candidates: 1
- Locally easy rows: 2
- Top hard row: 2
- Warning: none
- row 2: label=0, vote=1, hardness=0.8000, opp_vote=1.0000, entropy=0.0000, flags=label_issue_candidate

## Feature Separability Lens
- Rows scanned: 4
- Input dimension: 2
- Top feature: x2
- Top AUC: 0.9500
- Top balanced accuracy: 0.9000
- Near-perfect features: 1
- Weak features: 1
- Redundant pairs: 1
- Warning: none
- x2: AUC=0.9500, bal_acc=0.9000, SMD=2.4000, direction=positive_high, flags=strong_single_feature
- redundant x1/x2: corr=0.9700, flags=redundant_features

## Prototype Audit
- Rows scanned: 4
- k: 3
- Prototypes: 2
- Boundary rows: 1
- Isolated rows: 1
- Possible label contradictions: 1
- Top boundary row: 2
- Top contradiction row: 2
- Warning: none
- prototype row 0: label=0, score=0.8000, opp_frac=0.0000, flags=class_prototype
- boundary row 2: label=1, boundary=0.6000, contradiction=0.7000, flags=class_boundary,possible_label_contradiction

## OOD Sentinel
- Model used: True
- Rows scanned: 4
- Top row: 3
- Max OOD score: 3.2000
- Max robust z: 4.1000
- Max nearest-neighbor distance: 2.5000
- Flagged rows: 1
- Warning: none
- row 3: score=3.2000, max_z=4.1000, nn=2.5000, loss=1.1000, p=0.9000, flags=robust_outlier

## Bootstrap Stability Diagnostics
- Models: 8
- Feature map: linear
- Threshold: 0.4000
- Ensemble F1: 0.8000
- Ensemble accuracy: 0.7500
- Mean probability std: 0.0800
- Max probability std: 0.2200
- Max disagreement: 0.5000
- Unstable rows: 1
- Top row: 2
- Warning: none
- row 2: instability=0.7000, std=0.2200, disagreement=0.5000, mean_p=0.4800, flags=committee_disagreement

## MPS Bond Sweep
- Input dimension: 2
- Physical dimension: 4
- Validation samples: 2
- Tested chi: [4, 8]
- Recommended chi: 8
- Recommended F1: 0.8000
- chi=8: F1=0.8000, accuracy=0.7500, Brier=0.2000, ECE=0.1000

## Trial History
- Trial 1: map=rff, f1=0.7500, brier=0.1000, log_loss=0.3000
