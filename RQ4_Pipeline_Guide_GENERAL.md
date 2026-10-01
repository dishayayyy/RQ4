# RQ4 -- Counterfactual Reasoning + Algorithmic Recourse Pipeline Guide (General, AAA-GGG)

This guide documents the planned general-purpose RQ4 pipeline, covering both the domain-wise
(3-DAG-split) and combined-DAG structures. It applies uniformly to all seven modules and is
built against the finalized RQ3 guide (same edge lists, same reachability exclusions, same
LassoCV overrides, same capping conventions). AAA is the template; BBB-GGG follow.

Professor's directive (carried over): **domain-wise (3-DAG-split) FIRST, combined-DAG SECOND**,
per module. RQ4 must not be shallow: counterfactuals come from a properly fitted SCM, and
recourse must be causally verified, not just generated.

**Core idea:** RQ3 asked why a student was anomalous. RQ4 asks (1) what would this student's
outcome have been had something been different (counterfactual reasoning), and (2) what
concrete, feasible changes would get the student to a passing outcome (algorithmic recourse).

**Method split (confirmed plan):**
- **Layer 1 -- Counterfactual reasoning:** `dowhy.gcm.counterfactual_samples` on the fitted SCM.
- **Layer 2 -- Algorithmic recourse:** dice-ml generates candidate recourse; every candidate is
  then pushed back through the SCM to test whether it actually works causally. dice-ml alone is
  associational (it only sees a classifier), so the SCM verification step is what makes the
  recourse causal.
- **Reported gap:** share of dice-ml-valid recourses that stay valid after causal verification.
  This "associational vs causal recourse" gap is a headline result for the paper.

---

## Group and target definitions

**Cohorts (never pooled, same as RQ3):**
- **Withdrawn** = `final_result` 0
- **Fail** = `final_result` 1

Success (2, 3) is not a recourse cohort. It is used only as the reference group (e.g. median
values for "move to the Success median" interventions).

**Target outcome:** Pass or higher (`final_result` >= 2).
- DiCE side: binary label `success = (final_result >= 2)`.
- SCM side: `final_result` is continuous in the SCM, so the counterfactual value is compared to
  `SUCCESS_THRESHOLD = 1.5` (rounds to Pass). Document this rule in the writeup.

Unlike RQ3, Fail is NOT expected to be trivial here. Recourse does not depend on anomaly
attribution, so Fail gets a real result even in modules where RQ3's Fail attribution was 0.0.

---

## Actionability rules (most important design decision)

| Class | Variables | Treatment in RQ4 |
|---|---|---|
| Immutable | gender, age_band, region, imd_band, highest_education, disability, num_of_prev_attempts | Never varied in recourse. Used only for fairness counterfactuals (DAG1). |
| Actionable (engagement) | total_clicks (combined), individual click types (DAG3) | Primary recourse levers. Most upstream, propagate downstream through the SCM. |
| Actionable (performance) | ass_N_score | Recourse candidates, but downstream of clicks. Interpret as "achievable performance level", not a free lever. |
| Outcome | final_result | Never varied. |

Consequences:
- **DAG1 (demographics) has no actionable nodes**, so no recourse is run on it. It gets
  counterfactual fairness questions only (Layer 1), tying back to the Phase 2 fairness work.
- **DAG2 and DAG3 and the combined DAG** get both layers.
- `studied_credits` is treated as immutable for recourse (a student cannot retroactively change
  the credit load that produced the current outcome). Flag if the professor disagrees.

Optional refinement (decide on AAA): a checkpoint constraint. A student who has already
completed ass_1-ass_3 can only still change ass_4 onward. Implement by fixing earlier
assessments via `features_to_vary`. This makes recourse realistic as an early-warning
intervention. Report both unconstrained and checkpoint-constrained results only if time allows.

---

## Notebook structure and file locations

**Use new notebooks for RQ4, not the RQ3 ones.** Reasons: RQ3 notebooks are long and slow to
rerun, a kernel restart in RQ3 has already cost progress before, and the project's standing
pattern is one notebook per structurally different pipeline.

- Domain-wise: `RQ4\{MODULE}_RQ4_DomainWise.ipynb`
- Combined DAG: `RQ4\{MODULE}_RQ4_Combined.ipynb`
- All outputs: `RQ4\{MODULE}\`
- Naming convention: `{MODULE}_RQ4_{DAG name}_{artifact type}.{ext}`
  e.g. `AAA_RQ4_Combined_Withdrawn_recourse.csv`, `AAA_RQ4_DAG2_Assessments_cf_summary.csv`

Run modules sequentially only (parallel runs cause resource contention). Keep the laptop
plugged in with sleep disabled. Save intermediate CSVs after each cohort finishes, not only at
the end, so a kernel loss does not wipe a long run.

---

## Shared setup (both notebooks start with this)

```python
import subprocess, sys

def pip_install(pkg):
    subprocess.run([sys.executable, "-m", "pip", "install", pkg], check=True)

# Uncomment if not yet installed in the analytics env
# pip_install("dowhy")
# pip_install("dice-ml")
```

```python
import pandas as pd
import numpy as np
import networkx as nx
import dowhy.gcm as gcm
import dice_ml
import matplotlib.pyplot as plt
import os
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

MODULE = "AAA"   # change per module
OUTPUT_DIR = rf"C:\Users\Disha Sharma\Desktop\Paper_1\RQ4\{MODULE}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SUCCESS_THRESHOLD = 1.5   # SCM-side rule for "Pass or higher"
N_RECOURSE_QUERIES = 100  # max students per cohort sent to DiCE (tune on AAA)
TOTAL_CFS = 3             # DiCE candidates per student

np.random.seed(42)
```

Data loading and preprocessing are IDENTICAL to the RQ3 guide (same encodings, same imd_band
'10-20' dual-key fix, same -1 / 0 fill rules, `total_clicks` quartile bin for the combined
notebook only). Do not re-derive; paste that block. Check `ass_cols` per module.

---

## Step 0: SCM readiness check (do this first on AAA)

RQ3 saved fitted SCMs as pickles, but `gcm.counterfactual_samples` needs an
**invertible** SCM (noise must be recoverable from observed data), and RQ3's SCMs were built
as plain `gcm.StructuralCausalModel`. Expect that the RQ3 pickles cannot be reused directly
for counterfactuals. Verify with one quick test before building anything:

```python
with open(os.path.join(RQ3_DIR, f"{MODULE}_scm_dag2_assessments.pkl"), "rb") as f:
    scm_test = pickle.load(f)
try:
    _ = gcm.counterfactual_samples(scm_test, {'ass_1_score': lambda x: 1},
                                   observed_data=df_dag2.head(5))
    print("RQ3 pickle is usable for counterfactuals")
except Exception as e:
    print("Not usable, refit as invertible SCM:", e)
```

If it fails (expected), refit each SCM as `gcm.InvertibleStructuralCausalModel` using the SAME
edge list, SAME reachability exclusions, and SAME overrides as that module's RQ3 notebook.
Copy the override list per module from the RQ3 notebook markdown cells, do not rediscover it.
Overrides use additive-noise models, which are invertible:

```python
scm = gcm.InvertibleStructuralCausalModel(dag)
gcm.auto.assign_causal_mechanisms(scm, df_dag)
# re-apply this module's RQ3 overrides, e.g.
scm.set_causal_mechanism('age_band', gcm.AdditiveNoiseModel(gcm.ml.create_lasso_regressor()))
gcm.fit(scm, df_dag)
eval_result = gcm.evaluate_causal_model(scm, df_dag, compare_mechanism_baselines=True)
```

Confirm from the evaluation report that mechanisms are additive-noise and check the
invertibility assumption result, particularly for nodes that needed overrides in RQ3
(e.g. total_clicks in BBB). If an invertibility check is rejected for a node that lies on the
recourse path, flag it as a limitation for that module.

Save the refit SCMs as new pickles in the RQ4 output folder:
`{MODULE}_rq4_scm_{dagname}.pkl`.

---

## Reusable helper functions

```python
def get_cohort(df_dag, code, cap=None, seed=42):
    """code: 0=Withdrawn, 1=Fail, [2,3]=Success reference"""
    if isinstance(code, list):
        group = df_dag[df_dag['final_result'].isin(code)]
    else:
        group = df_dag[df_dag['final_result'] == code]
    if cap is not None and len(group) > cap:
        group = group.sample(n=cap, random_state=seed)
    return group
```

```python
def run_counterfactual(scm, cohort_df, interventions):
    """Unit-level counterfactuals: each student keeps their own noise term.
    Returns a dataframe with observed and counterfactual final_result."""
    cf = gcm.counterfactual_samples(scm, interventions, observed_data=cohort_df.reset_index(drop=True))
    out = pd.DataFrame({
        'observed_final_result': cohort_df['final_result'].values,
        'cf_final_result': cf['final_result'].values,
    })
    out['shift'] = out['cf_final_result'] - out['observed_final_result']
    out['reaches_success'] = out['cf_final_result'] >= SUCCESS_THRESHOLD
    return out
```

```python
def summarize_counterfactual(cf_df, label):
    return pd.Series({
        'intervention': label,
        'n': len(cf_df),
        'mean_shift': cf_df['shift'].mean(),
        'median_shift': cf_df['shift'].median(),
        'pct_reaching_success': cf_df['reaches_success'].mean() * 100,
    })
```

---

## PART A -- Domain-Wise (3-DAG-Split) Pipeline

Reuses the confirmed RQ1/RQ3 3-DAG-split edge lists. **Always request the edge list as pasted
TAIL/ARROW text from Disha before building.** Apply the same reachability exclusions as RQ3
for that module (never assume another module's set carries over).

### Layer 1: Counterfactual reasoning

Per DAG, three question types.

**1a. Unit/group-level "shift to Success" interventions (DAG2, DAG3).** Choose the intervened
nodes from RQ3's top contributors for Withdrawn (read the RQ3 summary CSVs) and from RQ2's
largest ATEs. Two intervention styles:

```python
success_ref = get_cohort(df_dag2, [2, 3])
target_val = success_ref['ass_1_score'].median()

# style 1: set to the Success-group median
cf_median = run_counterfactual(scm_dag2, withdrawn_dag2,
                               {'ass_1_score': lambda x: target_val})

# style 2: move up by one unit (bounded)
cf_shift = run_counterfactual(scm_dag2, withdrawn_dag2,
                              {'ass_1_score': lambda x: x + 1})
```

Repeat for each selected node and for both cohorts (Withdrawn, Fail). No cap is needed for
counterfactuals themselves (cheap relative to `attribute_anomalies`), but if a cohort is very
large, apply the standard 500-row cap with seed=42 and document it.

**1b. Cross-check against RQ2.** Compare the group-average counterfactual shift for a
node against that node's RQ2 ATE. Agreement supports the SCM; disagreement is informative
(the counterfactual is conditional on the at-risk cohort, the ATE is population-wide) and
should be discussed, not hidden.

**1c. Fairness counterfactuals (DAG1 only).** Flip an immutable attribute and measure the
outcome change, holding each student's noise fixed:

```python
ref_imd = df_dag1['imd_band'].max()
cf_fair = run_counterfactual(scm_dag1, withdrawn_dag1, {'imd_band': lambda x: ref_imd})
```

Report `highest_education` and `imd_band` first (the two cross-module direct drivers from
RQ1). This is a counterfactual fairness check in the Phase 2 sense, not recourse.

Save per DAG: `{MODULE}_RQ4_{DAG}_cf_summary.csv` (one row per intervention x cohort).

### Layer 2: Algorithmic recourse (DAG2 and DAG3 only)

**2a. Train the recourse classifier.** Features = all nodes in the DAG except `final_result`;
label = `success`. Report AUC on a held-out split. If AUC is poor, recourse from it is
meaningless: stop and flag before continuing.

```python
feat_cols = [c for c in df_dag2.columns if c != 'final_result']
data = df_dag2.copy()
data['success'] = (data['final_result'] >= 2).astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    data[feat_cols], data['success'], test_size=0.2, random_state=42, stratify=data['success'])
clf = RandomForestClassifier(n_estimators=200, random_state=42).fit(X_train, y_train)
print("AUC:", roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1]))
```

**2b. Generate candidates with DiCE (actionable features only).**

```python
train_df = X_train.copy(); train_df['success'] = y_train.values
d = dice_ml.Data(dataframe=train_df, continuous_features=feat_cols, outcome_name='success')
m = dice_ml.Model(model=clf, backend='sklearn')
exp = dice_ml.Dice(d, m, method='random')

actionable = feat_cols                      # DAG2: all assessments; DAG3: all click types
permitted_range = {c: [train_df[c].min(), train_df[c].max()] for c in actionable}

queries = get_cohort(df_dag2, 0, cap=N_RECOURSE_QUERIES)[feat_cols]
cfs = exp.generate_counterfactuals(queries, total_CFs=TOTAL_CFS, desired_class=1,
                                   features_to_vary=actionable,
                                   permitted_range=permitted_range)
```

Notes: DiCE is slow per query, which is why recourse queries are capped (default 100 per
cohort, fixed seed). Record the cap and the cohort size in the writeup. `-1` non-submission
values in assessments are legitimate values (a student can move from "not submitted" to a
score), so keep the `-1` in the permitted range.

**2c. Causal verification (the key step).** For each DiCE candidate, identify the changed
features and push the change through the SCM as an intervention, then read the counterfactual
`final_result` for that same student:

```python
def causal_check(scm, student_row, cf_row, changed, mode="hard"):
    """mode='hard': intervene on every changed node.
    mode='propagated': intervene only on changed nodes with no changed ancestor,
    letting descendants respond through the SCM (more causal, stricter)."""
    nodes = changed
    if mode == "propagated":
        g = scm.graph
        nodes = [n for n in changed if not any(a in changed for a in nx.ancestors(g, n))]
    interventions = {n: (lambda x, v=cf_row[n]: v) for n in nodes}
    out = gcm.counterfactual_samples(scm, interventions, observed_data=student_row.to_frame().T)
    return float(out['final_result'].iloc[0])
```

A candidate is **causally valid** if the returned value is >= `SUCCESS_THRESHOLD`. Run both
modes. If DiCE changes a downstream assessment directly, "hard" credits the change as given,
while "propagated" tests whether the upstream change alone would produce it.

**2d. Metrics per module x DAG x cohort:**

| Metric | Definition |
|---|---|
| Validity | % of DiCE candidates that flip the classifier to Success |
| Causal validity | % of candidates with SCM counterfactual final_result >= 1.5 |
| Validity gap | Validity minus causal validity (associational overpromise) |
| Proximity | Mean normalized distance between original and candidate |
| Sparsity | Mean number of features changed per candidate |
| Diversity | Mean pairwise distance among the TOTAL_CFS candidates per student |
| Most-changed features | Frequency table of which features recourse changes |

Save per DAG per cohort: `{MODULE}_RQ4_{DAG}_{cohort}_recourse.csv` (row per candidate with
both validity flags) and one `{MODULE}_RQ4_{DAG}_recourse_metrics.csv` (metrics table).

### Visualization (bar charts only, as in RQ3)

- One bar chart per DAG: validity vs causal validity (hard and propagated) by cohort.
- One bar chart per DAG: frequency of features changed in recourse, by cohort.
- No heatmaps, no per-student plots.

---

## PART B -- Combined-DAG Pipeline

Separate notebook `{MODULE}_RQ4_Combined.ipynb`. Same reachability exclusions as the module's
RQ3 combined notebook (per-module, not assumed). `total_clicks` is added and quartile-binned
exactly as in RQ3.

- Layer 1: same as Part A (1a and 1b), with `total_clicks` and the top RQ3 Withdrawn
  contributors as the intervened nodes. This is the main counterfactual result, since clicks
  and assessments sit in one SCM and effects propagate from engagement to assessment to outcome.
  Skip 1c here (fairness is covered in DAG1) unless the combined SCM retains a demographic
  that has a path to `final_result`.
- Layer 2: same as Part A. This is the main recourse result. Actionable set = `total_clicks`
  plus assessments retained in the combined SCM; immutable = everything else retained. Because
  `total_clicks` is upstream of assessments, the "propagated" mode is the key comparison here.
- Expect mixed-type features (ordinal bins plus scores). Treat all as continuous for DiCE, and
  round recommended values for interpretation only (never before the causal check).

---

## Optional Layer 3 (only if time allows)

SCM-native recourse: search over interventions on actionable nodes directly through
`counterfactual_samples` to find the minimal-change intervention that reaches the threshold,
and compare with DiCE's output. This gives a causal baseline independent of any classifier.
Do not start this before Layers 1 and 2 are finished for AAA.

---

## Stage 2 (SS vs STEM) -- observation

Adaptive drill-down, same rule as RQ3:
1. Group level (Social Sciences: AAA, BBB, GGG vs STEM: CCC, DDD, EEE, FFF): compare validity,
   causal validity, gap, sparsity, and most-changed features.
2. If nothing concrete, module level.
3. If still nothing, presentation level. First check whether the merged ModuleWise CSVs still
   carry a presentation identifier; if not, this level is not possible without re-deriving it.

Findings to look for: whether the validity gap is larger in one discipline; whether recourse
concentrates on the same click types found in RQ1/RQ2 (`clicks_oucontent`, `clicks_quiz`);
whether the CCC suppression effect (negative `clicks_oucontent` ATE) makes any recourse
recommendation counterproductive there.

---

## Expected issues to watch for

- **Non-invertible mechanisms:** counterfactuals fail or silently misbehave if a node's
  mechanism is not additive-noise. Step 0 is the safeguard.
- **Weak classifier:** DiCE recourse is only as good as the classifier. Always report AUC.
- **Discrete and binned nodes:** quartile-binned and score nodes are continuous in the SCM;
  counterfactual values may be non-integer. Interpret, do not force-round before checks.
- **Extreme non-submission:** later assessments are dominated by `-1` values in several
  modules (BBB, CCC). Recourse on those nodes mostly means "submit the assessment"; describe it
  that way in the writeup.
- **Sign instability:** CCC's `clicks_oucontent` (negative ATE), FFF's niche-click anomalies
  (`clicks_questionnaire`, `clicks_dualpane`). Cross-check any recourse that recommends these.
- **AAA small sample (748 rows):** cohort sizes for Withdrawn/Fail are small (RQ3 had 126 and
  91 earlier). Confirm actual sizes; flag results if a cohort is under 30.
- **Runtime:** DiCE per-query cost is the bottleneck. If runtime forces it, lower
  `N_RECOURSE_QUERIES` and document the value.

---

## Decisions to lock on AAA before scaling

1. `SUCCESS_THRESHOLD` (1.5 proposed) and the classifier label rule.
2. DiCE method (`random` proposed for speed; `genetic` as alternative) and `TOTAL_CFS`.
3. `N_RECOURSE_QUERIES` cap.
4. Whether `studied_credits` stays immutable.
5. Whether the checkpoint (time-aware) constraint is included.
6. Which counterfactual intervention styles are reported (median-to-Success, +1 shift, both).

---

## Status Checklist

| Module | Domain-split (DAG1 fairness CF; DAG2, DAG3 CF + recourse) | Combined DAG (CF + recourse) |
|---|---|---|
| AAA | Not started | Not started |
| BBB | Not started | Not started |
| CCC | Not started | Not started |
| DDD | Not started | Not started |
| EEE | Not started | Not started |
| FFF | Not started | Not started |
| GGG | Not started | Not started |

## Immediate Next Step

Share for AAA: the RQ3 domain-wise and combined notebooks (SCM fitting cells and override
documentation), the confirmed edge lists as pasted TAIL/ARROW text, and the RQ3 Withdrawn and
Fail summary CSVs. Then build AAA Step 0 (SCM readiness), Part A Layer 1, Part A Layer 2, then
Part B, and lock the decisions listed above before replicating to BBB-GGG.
