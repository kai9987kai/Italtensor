# Italtensor

Italtensor is a local desktop workbench for small binary-classification experiments. It is designed for quick iteration: load or create a dataset, run a model, inspect validation metrics, export a report, and save a reusable model or dataset preset.

The full `requirements.txt` install includes TensorFlow. The app code still has a NumPy fallback path, so local development can run without TensorFlow when only the lightweight package dependencies are installed.

## What It Does

- Desktop GUI built with PySimpleGUI.
- Manual JSON sample entry and CSV import.
- Built-in dataset presets for quick experiments:
  - Linear blobs
  - XOR pattern
  - Imbalanced blobs
  - Signal plus noise
  - Concentric rings
  - Two moons
  - Rare event signal
  - Overlapping margin
  - Noisy labels
  - Sparse interaction signal
  - Deployment drift probe
  - Active learning margin
  - Spurious shortcut
  - Subgroup blind spot
  - Cost-sensitive screening
  - Decision utility tradeoff
  - Selective abstention triage
  - Conformal coverage lab
  - Label audit traps
  - Proxy leakage lab
- Save and import reusable dataset presets.
- Train once or run random-search experiments.
- Batch prediction CSV export for scoring unlabeled rows, including active-learning query rank.
- Reviewed-label import from scored batch CSV files for closed-loop active learning.
- Counterfactual recourse: ask what small numeric changes would flip the current prediction.
- Sample review diagnostics for likely label issues, hard rows, and ambiguous rows.
- Feature ablation diagnostics for model reliance, proxy-feature risk, and feature-selection sanity checks.
- Slice diagnostics for finding raw feature ranges where the active model underperforms.
- Threshold tradeoff sweeps for operating-point, precision/recall, and cost-sensitive decisions.
- Decision-curve utility diagnostics for comparing the active model against treat-all and treat-none baselines.
- Post-hoc conformal prediction-set diagnostics for coverage, singleton, ambiguous, and empty-set behavior.
- Selective prediction risk-coverage sweeps for choosing when to abstain on low-confidence rows.
- Robustness stress lab for Gaussian noise, feature dropout, and single-feature shifts.
- Dataset audit for imbalance, duplicates, label conflicts, constant features, and correlated features.
- No-TensorFlow fallback trainer with linear, quadratic, and random Fourier feature maps.
- Validation-tuned decision threshold plus fixed-`0.5` baseline metrics.
- Class weighting for imbalanced binary datasets.
- Feature standardization fitted only on the training split.
- Permutation-style feature importance on validation data.
- Probability diagnostics including Brier score, log loss, calibration error, ROC-AUC, and average precision.
- Split-conformal and **APS** (Adaptive Prediction Sets) uncertainty diagnostics for abstention experiments.
- **MPS tensor-chain** binary classifier (`backend=mps`): ordered features as sites, bond dimension chi, soft site embeddings.
- Dataset **audit** (imbalance, duplicates, constant/correlated features) in the workbench and reports.
- **Learning curve** diagnostics (F1 vs. training fraction).
- Trial-history export for comparing auto-experiment runs.
- Explicit training backend control: `auto`, `numpy`, `mps`, or `keras` (when TensorFlow is installed).
- Multi-backend run queue: train NumPy, MPS, and Keras models in one sweep and store each in the registry.
- Model panel communication: per-model votes, consensus fusion (mean, median, vote, weighted, stacking), and disagreement scores.
- Persistable model registry (`.json`) for NumPy models and ensembles across sessions.
- JSON or Markdown experiment reports.
- Model save/load:
  - `.italtensor-model.json` for NumPy fallback models
  - `.italtensor-mps.json` for MPS chain models
  - `.italtensor-meta.json` sidecar metadata (no double `.json` suffix)
  - `.keras` for TensorFlow/Keras models when TensorFlow is installed

## Install

Editable install (recommended):

```powershell
python -m pip install -e .
```

Base install from requirements:

```powershell
python -m pip install -r requirements.txt
```

TensorFlow-only backend file:

```powershell
python -m pip install -r requirements-tensorflow.txt
```

TensorFlow is the default full-stack install path, but the fallback trainer can still train and predict without TensorFlow in lightweight environments.

## Run

```powershell
python main.py
```

## Quick Start

1. Choose a built-in dataset preset.
2. Keep `Map` as `rff` or choose `quadratic` for XOR-style patterns when using the default NumPy fallback.
3. Click `Load preset`.
4. Click `Train once` or `Run auto experiments`.
5. Enter a prediction vector such as `[0.2, 0.8]`.
6. Click `Predict`.
7. Optionally export batch predictions, save the model, export a report, or save the dataset as a preset.

## Input Formats

Manual training sample:

```json
[[0.1, 0.2, 0.3], 1]
```

Prediction vector:

```json
[0.1, 0.2, 0.3]
```

CSV datasets use the final column as the binary label:

```csv
f1,f2,label
0.1,0.2,0
0.9,0.7,1
```

Labels must be `0` or `1`. Feature values must be finite numbers. All rows must have the same number of features.

Batch prediction CSV files are unlabeled: every column is treated as a numeric feature and the column count must match the trained model's raw input dimension. The exported CSV adds probability, predicted label, conformal set, uncertainty score, active-learning query score/rank, drift score, max absolute z-score, OOD flag, review priority, and a blank `italtensor_review_label` column.

## Dataset Presets

Preset files wrap the normal dataset shape with metadata:

```json
{
  "kind": "italtensor.dataset_preset",
  "schema_version": 1,
  "name": "My preset",
  "description": "Reusable local dataset",
  "training_defaults": {
    "epochs": 50,
    "batch_size": 16,
    "trials": 8,
    "feature_map": "linear"
  },
  "recommended_feature_map": "linear",
  "feature_names": ["x1", "x2"],
  "label_names": {"0": "negative", "1": "positive"},
  "prediction_examples": [
    {"name": "Likely positive", "features": [0.9, 0.7], "expected_label": 1}
  ],
  "dataset": {
    "input_dim": 2,
    "samples": [
      {"features": [0.1, 0.2], "label": 0},
      {"features": [0.9, 0.7], "label": 1}
    ]
  }
}
```

Use `Save as preset` to turn the current dataset into a reusable preset. Built-in and imported presets can carry training defaults, a recommended feature map, feature names, label names, and prediction examples; the GUI applies core training defaults when loading presets.

## Experiment Notes

The NumPy fallback is intentionally lightweight. It trains a logistic classifier, but can expand features before fitting:

- `linear`: fastest, best for simple boundaries.
- `quadratic`: adds squared and interaction terms; useful for XOR-like structure.
- `rff`: random Fourier features inspired by random kitchen sinks, useful for nonlinear boundaries without adding heavy dependencies.

Feature maps are used by the NumPy fallback backend. If TensorFlow is installed and active, Keras models ignore the feature-map selector.

Auto experiments search model settings and feature maps, then rank runs by validation F1, accuracy, and validation loss.

Experiment reports include dataset availability, class counts when a dataset is loaded, the selected threshold, fixed-`0.5` baseline metrics, calibration diagnostics, conformal-style uncertainty diagnostics, post-hoc conformal prediction-set diagnostics, feature importance, feature ablation diagnostics, decision-curve utility diagnostics, selective prediction risk-coverage diagnostics, and trial history for auto experiments.

The uncertainty output is intended as an experimental local diagnostic. When each class has enough samples, Italtensor uses a separate calibration split to estimate a split-conformal-style quantile, then evaluates coverage on the validation split. Prediction displays a label set such as `{0}`, `{1}`, `{0,1}`, or `abstain`. Tiny datasets fall back to validation-reused uncertainty and mark that source in model metadata and reports.

Batch prediction exports rank rows for review using threshold-distance uncertainty, conformal ambiguity, and row-level drift diagnostics. Rows near the decision threshold, rows whose conformal set contains both labels, or rows whose standardized features cross the OOD flag threshold get higher review priority. The separate active-query score favors uncertain in-distribution rows so the file can double as a lightweight active-learning queue without mistaking every shifted row for a good labeling candidate.

After reviewing a scored batch file, fill `italtensor_review_label` with `0` or `1` on the rows you want to add, leave the rest blank, and click `Import reviewed labels`. Italtensor imports only reviewed rows, ignores the model-generated `italtensor_label`, appends the human labels to the training dataset, and invalidates the stale model so the next training run uses the expanded data.

Counterfactual recourse is available next to `Predict`. It runs a small model-agnostic search over the raw numeric input and reports the nearest found feature changes that cross the current decision threshold. This is useful for local debugging: it shows whether a prediction flips because of one dominant feature, many small feature moves, or no nearby move in the search budget.

Sample review diagnostics run the active model over the loaded dataset and rank rows for manual inspection: confident disagreements that may be label issues, high-loss hard examples, and near-threshold ambiguous rows. The `Label audit traps` preset is built to exercise this workflow. These diagnostics are model-assisted review cues, not automatic relabeling.

Feature ablation diagnostics run the active model on the loaded dataset, then neutralize and permute one raw feature at a time. The output ranks features by F1 drop, label-flip rate, probability movement, and label correlation. The `Proxy leakage lab` preset is built for this workflow: it makes a shortcut-like proxy easy to spot without claiming causal proof.

Slice diagnostics run the active model on the loaded dataset, split raw numeric features into quantile bins, and rank the ranges where F1 and accuracy fall most below the overall dataset score. This is meant for quick subgroup debugging: a model can look strong on average while failing in a narrow feature range or minority marker. Slice diagnostics are stored in reports after you run them.

Threshold tradeoff sweeps run the active model once, evaluate many probability cutoffs, and report best-F1, best-balanced-accuracy, minimum-cost, high-recall, and high-precision operating points. The default cost model treats false negatives as more expensive than false positives, which makes the `Cost-sensitive screening` preset useful for threshold experiments. These diagnostics are not applied automatically to the active model threshold; they are stored in reports and sidecars for explicit decision-making.

Decision-curve diagnostics compare the model's net benefit against acting on every row and acting on no rows across a threshold grid. This is useful when a false positive has a concrete action cost: the best F1 threshold may not be the best operating point. The `Decision utility tradeoff` preset creates rare positives and a broad gray zone to make the useful threshold range visible.

Post-hoc conformal diagnostics split the loaded dataset into calibration and evaluation rows, sweep alpha values, and report empirical coverage, coverage gap, mean prediction-set size, singleton rate, ambiguous `{0,1}` rate, empty-set rate, and class-specific coverage when observable. This complements the training-time uncertainty summary by making coverage/efficiency tradeoffs visible after a model is active. The `Conformal coverage lab` preset is built to show confident singleton regions and an overlapping shoulder where both-label sets are expected.

Selective prediction diagnostics sweep confidence cutoffs and report how much coverage is retained when low-confidence rows are abstained. The ranked cutoffs show covered accuracy, F1, error rate, abstention rate, and risk-coverage area. The `Selective abstention triage` preset is designed so ambiguous boundary rows are natural abstention candidates.

The robustness stress lab runs against the current dataset and active model. It perturbs raw features with Gaussian noise, replaces random cells with dataset means to simulate missingness, and shifts individual features by one raw standard deviation. The output reports worst-case F1, label-flip rate, probability movement, and the most damaging perturbation. Stress results stay separate from validation metrics and are included in exported reports after you run the stress test.

Dataset audits run from the desktop and are embedded in JSON/Markdown reports. They summarize class balance, duplicate feature rows, possible label conflicts, constant columns, highly correlated feature pairs, and compact warnings before you spend time tuning a model.

## Testing

```powershell
python -m pytest -q -rs
pytest -q -p no:cacheprovider
```

TensorFlow-specific tests skip when TensorFlow is not installed.

## Current Limits

- Italtensor is for local small-to-medium binary classification experiments, not production ML serving.
- The fallback trainer is not a neural network; nonlinear feature maps make it stronger but still lightweight.
- Existing `.keras` files require TensorFlow to load.
- Counterfactual recourse is a model-debugging heuristic. It does not know which features are actually actionable, causal, legal, or safe to change.
- Stress-lab results are perturbation diagnostics on the active dataset, not proof of future production robustness.
- Feature ablation diagnostics measure model reliance under simple median replacement and permutation. Correlated features can hide or spread reliance, so treat the output as inspection evidence rather than causal importance.
- Reviewed-label import trusts the reviewer column. Bad human labels will become training data, so keep the scored CSV as an audit trail.
- Slice diagnostics use simple one-feature quantile bins in v1. They are interpretable and fast, but they will not discover every multi-feature or semantic error slice.
- Threshold tradeoffs run on the active loaded dataset. Use held-out or reviewed data when you want deployment-grade operating-point evidence.
- Decision-curve results depend on the chosen threshold grid and only encode one simple harm ratio at a time through threshold odds. One-class datasets are allowed for debugging but marked low evidence.
- Post-hoc conformal diagnostics are strongest on held-out or newly reviewed data. If you run them on the same rows used to train the model, treat coverage as a local sanity check rather than a finite-sample guarantee.
- Selective prediction can hide weak subgroup performance if abstention falls unevenly across groups. Pair it with slice diagnostics when coverage fairness matters.
- Sample review can surface genuine label mistakes, ambiguous cases, or model blind spots. Treat flagged rows as a review queue, not ground truth.
- Reports are richest when the dataset is loaded in the same session as the model; model-only reports cannot reconstruct class counts.
- Validation metrics need enough examples from both classes. The app requires at least two samples per class for train/validation splitting.
- Conformal-style uncertainty is strongest when the dataset is large enough for the dedicated calibration split. On tiny datasets, it falls back to validation-reused diagnostics.

## Multi-Model Controls

- **Backend** selector on the training panel chooses `auto`, `numpy`, `mps`, or `keras`. `auto` uses Keras when TensorFlow is installed, otherwise NumPy.
- **Run Multi-Backend** trains one model per available backend with the same hyperparameters, stores each run in the registry, and activates the best validation F1.
- **Fusion** on the registry panel controls how ensembles combine member predictions: `mean`, `median`, `vote`, `weighted` (by slot F1), or `stacking` (linear meta-learner on validation probabilities).
- **Panel Predict** runs every stored slot on the prediction vector and logs a communication trace (per-model votes plus panel consensus and disagreement).
- **Stacked Ensemble** fits Wolpert-style stacking weights on the validation split, then adds a stacked ensemble slot.
- **Save/Load Registry** writes `italtensor.model_registry` JSON files containing NumPy models and ensembles (Keras slots must be saved individually).

## Research And Design Anchors

- Polynomial feature expansion follows the same idea exposed by scikit-learn `PolynomialFeatures`.
- Random Fourier features are based on Rahimi and Recht's random kitchen sinks work for approximating kernel methods.
- Conformal prediction design is informed by Vovk, Gammerman, and Shafer's [Algorithmic Learning in a Random World](https://link.springer.com/book/10.1007/978-3-031-06649-8), Angelopoulos and Bates' [gentle introduction](https://arxiv.org/abs/2107.07511), and Romano, Sesia, and Candes' classification-set work, [Classification with Valid and Adaptive Coverage](https://papers.nips.cc/paper/2020/hash/244edd7e85dc81602b7615cd705545f5-Abstract.html).
- The post-hoc conformal diagnostic follows split-conformal classification practice: calibrate nonconformity scores on one split, evaluate prediction-set coverage and efficiency on another, and report target coverage separately from observed coverage. This mirrors recent practical tutorials and robust-conformal framing while keeping Italtensor dependency-free.
- Batch review priority follows classic uncertainty-sampling intuition from Burr Settles' [Active Learning Literature Survey](https://burrsettles.com/pub/settles.activelearning.pdf): examples closest to the model's decision boundary are often the most informative to inspect or label next.
- Reviewed-label import closes the pool-based active-learning loop: score an unlabeled pool, label the most useful rows, merge them into the training set, and retrain. This mirrors human-in-the-loop active-learning workflows summarized in recent HITL surveys such as [Human-in-the-loop machine learning: a state of the art](https://link.springer.com/article/10.1007/s10462-022-10246-w).
- Sample review is inspired by confident-learning style label-audit workflows such as Northcutt, Jiang, and Chuang's [Confident Learning](https://arxiv.org/abs/1911.00068) and dataset cartography ideas that use model behavior to find hard or ambiguous examples ([Dataset Cartography](https://arxiv.org/abs/2009.10795)). Italtensor uses a lightweight probability/loss heuristic rather than a full Cleanlab implementation.
- Feature ablation/reliance diagnostics follow model-agnostic inspection ideas in scikit-learn's [permutation importance](https://scikit-learn.org/stable/modules/permutation_importance.html) guidance and Fisher, Rudin, and Dominici's model-reliance framing ([arXiv](https://arxiv.org/abs/1801.01489)). The simple ablation view is also related to partial-dependence and ICE-style local response inspection described in scikit-learn's [PDP/ICE documentation](https://scikit-learn.org/stable/modules/partial_dependence.html).
- Batch drift flags are a lightweight standardized-distance diagnostic inspired by distance-based OOD detection work such as Lee et al.'s [simple unified framework for detecting out-of-distribution samples](https://proceedings.neurips.cc/paper/2018/file/abdeb6f575ac5c6676b747bca8d09cc2-Paper.pdf). Italtensor uses per-row z-score summaries rather than a full covariance model to stay dependency-free.
- Slice diagnostics are inspired by Slice Finder's focus on interpretable subsets where model performance is poor ([Google Research summary](https://research.google/pubs/slice-finder-automated-data-slicing-for-model-validation/), [arXiv](https://arxiv.org/abs/1807.06068)) and newer slice-discovery work such as [Error Slice Discovery via Manifold Compactness](https://ojs.aaai.org/index.php/AAAI/article/view/40016). Italtensor keeps the first version lightweight by using raw-feature quantile bins rather than automated semantic slicing.
- Threshold tradeoff diagnostics follow standard decision-threshold tuning practice described in scikit-learn's [tuning the decision threshold](https://scikit-learn.org/stable/modules/classification_threshold.html) guidance and classic cost-sensitive classification framing, where probability estimates and deployment costs are separate decisions.
- Decision-curve diagnostics follow Vickers and Elkin's net-benefit framing for evaluating whether a prediction model adds value over default strategies, with practical reporting guidance from decision-curve analysis reviews ([technical note](https://pmc.ncbi.nlm.nih.gov/articles/PMC6123195/), [reporting guidance](https://pmc.ncbi.nlm.nih.gov/articles/PMC6261531/), [net-benefit intervals](https://link.springer.com/article/10.1186/s41512-023-00148-y)).
- Selective prediction follows the risk-coverage framing from El-Yaniv and Wiener's foundations of selective classification and Geifman and El-Yaniv's post-hoc selective classification work ([JMLR](https://jmlr.csail.mit.edu/papers/v11/el-yaniv10a.html), [NeurIPS 2017](https://papers.neurips.cc/paper/7073-selective-classification-for-deep-neural-networks), [SelectiveNet](https://proceedings.mlr.press/v97/geifman19a.html)). Italtensor implements a lightweight confidence-threshold sweep rather than training a separate rejector.
- Counterfactual recourse follows the black-box counterfactual explanation line from Wachter, Mittelstadt, and Russell's [counterfactual explanations without opening the black box](https://arxiv.org/abs/1711.00399) and the diverse-counterfactual framing in Mothilal, Sharma, and Tan's [DiCE work](https://arxiv.org/abs/1905.07697). Italtensor intentionally keeps v1 dependency-free and reports one nearby flip rather than a constrained causal recourse plan.
- The stress lab follows the common-corruption evaluation idea from Hendrycks and Dietterich's [robustness benchmark](https://arxiv.org/abs/1903.12261) and the shortcut-learning concern described by Geirhos et al. in [Shortcut Learning in Deep Neural Networks](https://arxiv.org/abs/2004.07780), adapted to dependency-free numeric tabular vectors.
- Multi-model stacking follows David Wolpert's stacked generalization idea: base models produce validation probabilities, and a linear meta-learner combines them ([stacked generalization](https://www.ml.cmu.edu/research/dap-papers/dap-wolpert-stacked-generalization.pdf)).
- Panel vote fusion and disagreement scores echo ensemble diversity diagnostics used in mixture-of-experts and committee-based classifiers; high disagreement flags inputs worth human review alongside conformal abstention.
- Brier/log-loss style probability evaluation and threshold tuning are standard classification-evaluation tools; Italtensor exposes both threshold-tuned metrics and fixed-threshold baselines so validation gains are visible instead of hidden.
