# Classifying Dry Bean Varieties with Logistic Regression and SVMs

Project 7, applying logistic regression and support vector machines to a real dataset,
finding a meaningful target variable, selecting explanatory variables, and testing how
well linear and kernel classifiers suit the problem.

Everything lives in one notebook: **[project7.ipynb](project7.ipynb)**.

---

## The dataset

[**Dry Bean Dataset**](https://archive.ics.uci.edu/dataset/602/dry+bean+dataset) (UCI Machine
Learning Repository). A grain company photographed individual beans of seven registered varieties
and ran each image through a segmentation pipeline that produced 16 continuous geometric
measurements per bean.

| | |
|---|---|
| Rows | 13,611 raw → **13,543** after removing 68 duplicate rows |
| Features | **16 quantitative** (area, perimeter, axis lengths, eccentricity, solidity, roundness, compactness, four shape factors) |
| Target | `Class`, the bean variety, **7 classes** |
| Missing values | none |
| Class balance | **imbalanced, 6.79 : 1** (DERMASON 3,546 → BOMBAY 522) |
| Majority-class baseline | **26.18%** accuracy |

### Why I chose it

- The target is naturally categorical with **7 classes**, enough for the multi-class strategy
  question (OVR fits 7 models, OVO fits 21) to have a real answer, few enough to stay fast.
- All 16 explanatory variables are **quantitative**, so no categorical encoding was needed.
- It has three properties that make the project questions bite rather than being formalities:
  - **class imbalance (6.79:1)**, so `class_weight` has something to act on;
  - **features spanning five orders of magnitude**, so scaling is genuinely required, not a ritual;
  - classes that are **nearly but not quite linearly separable**, so linear and kernel models give
    a genuine, measurable comparison instead of one trivially winning.

---

## What I did

The brief asks for at least four of its listed questions. I worked through **all eight**, one per
notebook section.

| Section | Question | What I did |
|---|---|---|
| 3 | *(exploration)* | Class balance, feature-scale audit, correlation structure, class-conditional distributions, selected pair plots, PCA projection |
| 4 | **Which variables are predictive?** | Six independent methods, ANOVA F, mutual information, single-feature CV accuracy, multinomial coefficients, permutation importance on held-out data, and greedy forward selection, then reconciled where they disagree |
| 5 | **Can logistic regression / a linear SVM predict well?** | Dummy baselines, `GridSearchCV` for all four models, held-out metrics with confidence intervals, confusion matrices, per-class F1, and paired significance tests over repeated CV folds with a Holm correction |
| 6 | **Pair plots and decision boundaries** | Decision regions for 3 feature pairs x 4 models, a binary margin close-up with support vectors ringed, and a `gamma` sweep showing boundary shape from underfit to memorising |
| 7 | **Generalisability / bias-variance** | Learning curves, validation curves over each model's capacity parameter, a generalisation-gap table with model sizes, and bootstrap refits measuring prediction disagreement directly |
| 8 | **Polynomial vs RBF SVM** | Degree and `gamma` sweeps, the `coef0` effect, two-dimensional `C` x parameter searches, and a head-to-head counting the beans on which the two kernels actually disagree |
| 9 | **Effect of `class_weight`** | `None` vs `balanced` for three models, per-class recall deltas, a confusion-matrix difference map, and positive-class weight sweeps on both an *easy* imbalanced pair and a *hard* overlapping one |
| 10 | **Runtime, and OVR vs OVO** | Fit/predict timing across training sizes with fitted log-log scaling exponents, support-vector counts tied to prediction latency, explicit `OneVsRestClassifier` / `OneVsOneClassifier` comparison, and verification of what scikit-learn actually does |

---

## How I did it

### Protocol

- **One stratified 75/25 split** (10,157 train / 3,386 test), created once and reused by every
  section so results are comparable across sections.
- **Hyper-parameters chosen only by cross-validation on the training set.** The test set is used
  once, for the final reporting table.
- **One seed (`RANDOM_STATE = 42`)** for the split, every CV fold, and every resample.
- Because the classes are imbalanced, **balanced accuracy and macro F1 are reported alongside
  accuracy** everywhere.

### Decisions that actually changed a result

- **Scaling lives inside the pipeline.** Every model is a
  `Pipeline([StandardScaler(), classifier])`. The features range from `Area` ≈ 53,000 to
  `ShapeFactor2` ≈ 0.0017, so without scaling `Area` dominates both the L2 penalty and the RBF
  kernel distance. Keeping the scaler inside the pipeline means each fold's statistics are learnt
  from its training folds only, standardising up front would leak the validation fold into the fit.
- **Duplicates removed before splitting.** 68 exactly duplicated rows would otherwise have landed
  on both sides of the split and inflated the reported accuracy.
- **Significance tested honestly.** Models are compared on *shared* repeated-CV folds with paired
  tests and a Holm correction for the six pairwise comparisons, and the notebook states plainly
  that overlapping CV folds violate the independence assumption, so the p-values indicate
  consistency rather than a rigorous level.
- **Decision boundaries are drawn from two-feature models.** Training on 16 features and slicing
  the surface at the median of the other 14 produces a picture that is not the model's boundary
  anywhere in particular. The honest cost is that these 2-D models are weaker, so every panel
  title reports both its 2-D accuracy and the full 16-feature accuracy.
- **`coef0=1.0` for the polynomial kernel** rather than scikit-learn's default of `0.0`, which
  drops the kernel's lower-order terms. Keeping them is worth about **+0.023** accuracy here.
- **The hard case was tested, not just the easy one.** The `class_weight` section sweeps both
  BOMBAY-vs-rest (rare but trivially separable) and a genuinely overlapping pair. Testing only the
  first would have supported the false conclusion that `class_weight` does nothing on this data.
- **Library behaviour was verified, not assumed**, see the correction below.

---

## Key results

### All four models cluster tightly

| Model | CV accuracy | Test accuracy | Balanced acc. | Macro F1 | Model size |
|---|---|---|---|---|---|
| Logistic Regression | 0.9259 | 0.9209 | 0.9321 | 0.9328 | 119 coefficients |
| Linear SVM | 0.9220 | 0.9158 | 0.9270 | 0.9282 | 119 coefficients |
| Polynomial SVM | 0.9334 | **0.9256** | 0.9356 | 0.9371 | 1,626 support vectors |
| RBF SVM | **0.9346** | 0.9253 | 0.9358 | 0.9366 | 1,828 support vectors |

Majority-class baseline: **0.2618**. The spread across all four models is **0.97 percentage
points**, and their confidence intervals all overlap.

### Findings

- **The kernel SVMs beat the linear models significantly, but barely.** Paired, Holm-corrected
  tests confirm the advantage is consistent (+0.6 to +1.0 points), and also show that **RBF vs
  polynomial is the one comparison that fails significance**. Those two agree on **99%** of test
  beans, and where they differ neither is systematically right.
- **One pair of varieties causes 45.4% of all errors.** DERMASON and SIRA genuinely overlap in
  these 16 measurements; about a fifth of the training points in that pair are support vectors.
  Every learning curve has flattened, so **more data would not help**, the ~93% ceiling is set by
  the features, and breaking it needs colour or texture, not a better classifier.
- **A handful of features is enough.** Greedy forward selection reaches within one point of the
  full 16-feature score using only **4** features, while adding features in ANOVA order needs
  **14**, because the F-test's top three picks are three near-copies of "how big is the bean".
- **The expected collinearity problem did not appear.** `Area` and `ConvexArea` correlate at
  r = 0.99994, but the coefficients did *not* seesaw: the L2 penalty splits weight evenly between
  them and they move *together* across bootstrap resamples (r = +0.997, sd ≈ 0.10). The real
  hazard is **double counting**, not instability. Weakening the penalty to `C = 1000` makes the
  coefficients balloon and their variability grow roughly 40-fold.
- **`gamma` has a dramatic failure mode and `degree` does not.** Across its sweep the RBF SVM's CV
  accuracy ranges **0.2508 to 0.9303**, versus **0.9168 to 0.9293** for polynomial degree. At
  `gamma = 100` the model reaches **1.0000 training accuracy with 0.2508 CV accuracy** and stores
  **100%** of the training set as support vectors, overfitting visible in the model's size, not
  just its score. The two kernels differ far more in *tuning sensitivity* than in best-case accuracy.
- **`class_weight="balanced"` changes little here, for an interesting reason.** The rarest class is
  also the *easiest*: BOMBAY beans are much larger than every other variety, and sweeping their
  weight from 1 to 50 moves precision and recall **not at all**, both sit at 1.0000 throughout,
  because there is simply no headroom to buy. Weighting instead shifts errors among the middle
  classes (SIRA, SEKER, HOROZ, CALI and BARBUNYA all gain recall; DERMASON pays for it). On a
  genuinely overlapping pair the same sweep buys **+0.087 recall for −0.183 precision**, a real
  trade-off. **Imbalance alone is not a problem; imbalance combined with class overlap is**, which
  is why I tested both cases.
- **Prediction cost, not fit cost, is what separates the models.** Fit time is sub-second for all
  four at this scale, but the fitted log-log exponents separate them cleanly: **0.85 and 1.04** for
  logistic regression and the linear SVM, against **1.43 and 1.54** for the RBF and polynomial
  SVMs. The decisive gap is at prediction: **0.157 µs per bean for logistic regression against
  65.8 µs for the RBF SVM, a 419x difference**, because a kernel SVM must compare each bean
  against all 1,828 stored support vectors while the linear model evaluates one fixed 7x16 matrix
  product.
- **Three times the models is not three times the cost, OVO was actually *faster*.** OVO fits 21
  binary models against OVR's 7, but each sees only ~2/7 of the data, so with super-linear training
  cost it came out at **0.31x** the fit time for the RBF SVM and **0.54x** for the linear SVM.
  That is exactly why `SVC` uses OVO internally. The cost reappears at prediction time, where all
  21 models must be evaluated and their votes combined (1.5x to 4.9x slower here), and it grows
  quadratically with the class count, fine at 7 classes, prohibitive at 50.

### A correction to the brief's premise

The brief states that logistic regression and `LinearSVC` use one-vs-rest while `SVC` uses
one-vs-one. I verified this from the fitted objects rather than repeating it, and **it is no longer
accurate for logistic regression**: since scikit-learn 1.5, `LogisticRegression` fits a single
**multinomial (softmax)** model for multi-class problems, not a set of one-vs-rest models, which
is why its `predict_proba` rows sum to exactly 1 without renormalisation. `LinearSVC` *is* genuinely
one-vs-rest, and `SVC` *is* always internally one-vs-one, with `decision_function_shape` only
reformatting the output (21 pairwise columns vs 7). Section 10.5 demonstrates all three in code.

### Model I would choose

**Logistic regression**, despite the RBF SVM being the most accurate. The kernel model's advantage
is real and statistically significant but under one percentage point, and it costs roughly **15x
more stored parameters**, orders-of-magnitude slower prediction per bean, and the loss of
interpretable per-class coefficients. For a seed-sorting line, latency and explainability outweigh
the last half point. If maximum accuracy were the only criterion, the RBF SVM would be the right
answer, and the significance testing makes that preference defensible rather than arbitrary.

---

## Running it

The notebook is **self-contained**, it defines everything it needs and runs top to bottom with no
other project files. The dataset downloads automatically on first run and is then cached in
`data/raw/`, so later runs need no network.

```bash
cd CS_Project7

# create the environment and register the Jupyter kernel
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m ipykernel install --user --name project7-venv \
    --display-name "Python (project7)"

# then open the notebook and select the "Python (project7)" kernel
jupyter lab project7.ipynb
```

Select the **`Python (project7)`** kernel, then *Run All*. A full run took **2 minutes 24 seconds**
end to end on the machine it was developed on (multi-core, with `n_jobs=-1` for the searches);
expect longer on fewer cores. The slowest parts are the grid searches, the learning curves and the
kernel hyper-parameter sweeps. The committed notebook already has every output embedded, so it can
equally just be read without running anything.

### Files

| Path | What it is |
|---|---|
| `project7.ipynb` | The entire analysis, all code, figures and written commentary |
| `Dry_Bean_Classification_Report.docx` | The written report section (Word) |
| `Dry_Bean_Classification_Slides.pptx` | The presentation deck, 14 slides with speaker notes (PowerPoint) |
| `requirements.txt` | Pinned versions the analysis was developed and tested against |
| `data/raw/` | Dataset cache, created on first run (git-ignored) |

Every figure quoted in the report and the deck comes from the executed notebook.

### Environment

Python 3.10.11, with numpy 2.0.0, pandas 2.3.3, scikit-learn 1.7.2, matplotlib 3.10.9 and
scipy 1.15.3. No seaborn, all figures are plain matplotlib, with a colour-blind-safe
(Okabe-Ito) palette applied consistently so each class and each model keeps the same colour in
every figure.

---

## Limitations

- The data comes from a **single study with one camera and one segmentation pipeline**. Geometric
  features are exactly the kind that shift with calibration, so accuracy on images from different
  equipment is unknown.
- Only **shape** is measured. Colour and texture separate several of these varieties to the human
  eye, and their absence is the likely cause of the DERMASON/SIRA ceiling.
- The seven classes are the seven varieties in this study. A deployed sorter would meet varieties
  it was never trained on, and none of these models can answer "unknown", each must assign one of
  seven labels.
- **One train/test split** was used throughout. Cross-validation and bootstrap refits characterise
  the variability, but a repeated nested cross-validation would estimate generalisation error more
  rigorously at proportionally greater cost.
