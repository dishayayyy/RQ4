# implementationPlan.md - BBB checkpoint pipeline, to replicate for CCC, DDD, EEE, FFF, GGG (and AAA if wanted)

This file lets a new chat rebuild exactly what was done for BBB. Read sections 0 to 4 first, then follow section 5 stage by stage.
BBB's results are in `BBB_Final_Summary.md` (use them to sanity-check the shape of every new module, not the exact numbers).

---

## 0. What this project is

- User: Disha. Research project for SURGE'27 (IIT Kanpur). Paper on OULAD with causal AI.
- Current task: RQ4 (counterfactual reasoning + recourse) for each module. BBB is finished and is the template.
- Professor's rules: marks are never changed; only student activity on the course site (VLE clicks) can change; background stays fixed (`studied_credits` immutable); Fail and Withdrawn are never pooled.
- Modules: Social Sciences = AAA, BBB, GGG. STEM = CCC, DDD, EEE, FFF. After all modules, Stage 2 compares the two groups (section 9).
- Presentations used per module must match the merged `ModuleWise` files from RQ1 (BBB excludes 2014J; DDD excludes 2013B; others use all presentations).

---

## 1. How Disha wants cells given (follow exactly)

1. Before every major cell, give ONE markdown line, as a heading, for example: `### Stage 3.2 - Run PC with time-ordering rules, then read every edge`. Say "Markdown cell:" and put the line in its own block, then the code cell.
2. Inside every code cell, add short comments in simple words that say what each step does. No long paragraphs, no jargon where a plain word works.
3. Give cells in run order. Keep each cell to one job.
4. After each group of cells, say exactly what to paste back ("paste the printout of ...").
5. If a cell errors, Disha pastes the error. Fix only that cell and give the whole corrected cell (not a diff).
6. Reset any list that a loop appends to (`rows = []`) at the top of the cell, with a comment, so reruns do not duplicate rows.
7. Never assume a file path. Use the paths in section 4. If something may differ per module, mark it "INPUT NEEDED".

## 2. How Disha wants replies

- Start every reply with "Disha".
- No emojis, ever.
- Ask a follow-up question before jumping to a solution, but keep it to one question at a time.
- Do not re-explain basics already covered. Never ask Disha to repeat background that is in this file or in the summary.
- Ask before making any visual artifact (charts and figures are for her report; ask first).
- Prefer applied, intuition-first explanations over theory. Findings should be explainable in plain language.
- Say plainly when a result is provisional or a bug is found. If something Claude said earlier was wrong, say so and correct it.
- Do not put numbers in a message for the professor until the checks behind them are done.
- When asked for a professor update: short points, layman language, findings only, no jargon about the internal checks. Use the message tool.

---

## 3. Inputs needed from Disha (checklist, in the order they come up)

| When | What to ask for | Why |
|---|---|---|
| Before Stage 0 | Module, list of presentations used in RQ1's merged file (and any excluded ones) | Must match the RQ1 student count |
| Before Stage 0 | Folder of raw OULAD files (BBB: `Paper_1\DATA`) and output folder (`Paper_1\RQ4\{MODULE}`) | Paths |
| After Stage 0.2 | Student count and outcome counts, compare with the RQ1 merged file | Catches wrong presentation list |
| After Stage 0.3 | List of activity types of the module | Decides which activities can be levers |
| After Stage 1.3 | First assessment per presentation (type and due day) | Confirms day-28 checkpoint makes sense |
| Before Stage 3.1 | `win_acts`: the window click types to use. Default = the module's click types that appeared in its whole-course RQ1 DAG3 (BBB: 7 types). Ask Disha for that list, or use types used by at least about 10% of students in days 29-56 | Defines the graph variables |
| After Stage 3.1 | The variable check table | Variables with 1 level must be dropped |
| After Stage 3.2 | Edge list, direct parents of the outcome, isolated variables, the picture | Review of the graph |
| After Stage 4.5 | Confirmation of the recourse levers, the participation-type and volume-type activities, and the three kinds of direct edges to add | These are decisions, not computed |
| After Stage 5.2c | The model table | Confirms the main model choice |
| Before any professor message | OK to send numbers | Numbers only after checks |

---

## 4. Settings and folder layout

Raw files: `studentInfo.csv`, `studentRegistration.csv`, `studentVle.csv` (very large, read in chunks), `vle.csv`, `assessments.csv`, `studentAssessment.csv` in `C:\Users\Disha Sharma\Desktop\Paper_1\DATA`.

Outputs per module: `C:\Users\Disha Sharma\Desktop\Paper_1\RQ4\{MODULE}\` and one notebook `{MODULE}_RQ4.ipynb` there.
Environment: Windows, Jupyter, conda env `analytics` (Python 3.12). Graphviz dot path: `C:\Users\Disha Sharma\.conda\envs\analytics\Library\bin\dot.exe`. Keep the laptop plugged in and awake for long cells (click-file read, model evaluation).

| Setting | BBB value | Other modules |
|---|---|---|
| `MODULE` | "BBB" | INPUT NEEDED |
| `PRES` | ["2013B","2013J","2014B"] | INPUT NEEDED (match RQ1 merged file) |
| `CUTOFFS` | [28, 56] | Same (end of week 4 and week 8) |
| `FOCUS_ACT` | "quiz" | The module's quiz activity; if none, ask Disha which activity to use |
| `win_acts` | 7 types: forumng, homepage, oucontent, quiz, resource, subpage, url | INPUT NEEDED |
| Recourse levers | quiz, forum, content pages (start-type); homepage (volume-type) | Decide from Stage 4.5 |

Expected student counts from the RQ1 merges: AAA 748; BBB 5,617; CCC 4,434 (2014B 1,936 + 2014J 2,498); DDD 4,969; EEE 2,934 (1,052 + 694 + 1,188); FFF 7,762; GGG 2,534. Compare with Stage 0.2 and resolve any difference before going on.

Known data quirk: in the raw `studentInfo.csv` the 10-20% deprivation band is stored as `Oct-20`. The code below maps it to band 1.

---

## 5. Pipeline, stage by stage

Run the stages in this order in one notebook per module. All code is the final working version used for BBB.

### PART 0 - Build the data

Markdown cell: `### Stage 0.1 - Setup: imports, paths, module and checkpoint settings`
```python
# Libraries used in the data build and the feasibility check
import pandas as pd
import numpy as np
import os

# ---- Settings: change these per module ----
MODULE = "BBB"                                    # INPUT NEEDED per module
PRES = ["2013B", "2013J", "2014B"]                # INPUT NEEDED: same presentations as the RQ1 merged file
FOCUS_ACT = "quiz"                                # the activity we look at most closely (quiz in BBB)
RAW_DIR = r"C:\Users\Disha Sharma\Desktop\Paper_1\DATA"            # folder with the original OULAD CSVs
OUT_DIR = rf"C:\Users\Disha Sharma\Desktop\Paper_1\RQ4\{MODULE}"   # all outputs go here
os.makedirs(OUT_DIR, exist_ok=True)

# ---- Checkpoints in days from the course start: 28 = end of week 4, 56 = end of week 8 ----
CUTOFFS = [28, 56]
EARLY_END, WIN_END = 28, 56      # early period = up to day 28; action window = days 29 to 56
```

Markdown cell: `### Stage 0.2 - Load the students: final result plus registration and unregistration dates`
```python
# studentInfo: final outcome for every student; studentRegistration: when they registered and left
info = pd.read_csv(os.path.join(RAW_DIR, "studentInfo.csv"))
reg = pd.read_csv(os.path.join(RAW_DIR, "studentRegistration.csv"))

# Keep only this module and its presentations
info = info[(info.code_module == MODULE) & (info.code_presentation.isin(PRES))]
reg = reg[(reg.code_module == MODULE) & (reg.code_presentation.isin(PRES))]

# One row per student (date_unregistration is empty for students who never left)
stu = info.merge(
    reg[['code_presentation', 'id_student', 'date_registration', 'date_unregistration']],
    on=['code_presentation', 'id_student'], how='left')

# Check: should match the RQ1 merged file
print("students:", len(stu))
print(stu.final_result.value_counts().to_dict())
```
PASTE BACK: the two printed lines. Compare with the RQ1 count.

Markdown cell: `### Stage 0.3 - Load the activity-type lookup (which page ids are quizzes, forums, etc.)`
```python
# vle.csv tells us the activity type of every course page
vle = pd.read_csv(os.path.join(RAW_DIR, "vle.csv"))
vle = vle[(vle.code_module == MODULE) & (vle.code_presentation.isin(PRES))][
    ['id_site', 'code_presentation', 'activity_type']]
print("pages:", len(vle), "| activity types:", sorted(vle.activity_type.unique()))
```
PASTE BACK: the list of activity types. INPUT NEEDED later: which become `win_acts`.

Markdown cell: `### Stage 0.4 - Read the big click file in chunks and keep only this module (slowest cell, a few minutes)`
```python
# The click file is too big to load at once: read 500,000 rows at a time and keep only this module
parts = []
for i, chunk in enumerate(pd.read_csv(
        os.path.join(RAW_DIR, "studentVle.csv"),
        usecols=['code_module', 'code_presentation', 'id_student', 'id_site', 'date', 'sum_click'],
        chunksize=500_000)):
    chunk = chunk[(chunk.code_module == MODULE) & (chunk.code_presentation.isin(PRES))]
    if not chunk.empty:
        chunk = chunk.merge(vle, on=['id_site', 'code_presentation'], how='left')   # add the activity type
        parts.append(chunk[['code_presentation', 'id_student', 'activity_type', 'date', 'sum_click']])
    if i % 5 == 0:
        print("chunks read:", i + 1)       # progress

clicks = pd.concat(parts, ignore_index=True)
# 'date' = day of the click counted from the start (negative = before the start)
print("click rows:", len(clicks), "| date range:", clicks.date.min(), "to", clicks.date.max())
print("clicks with no activity type (should be about 0):", int(clicks.activity_type.isna().sum()))
```
PASTE BACK: the three printed lines.

Markdown cell: `### Stage 0.5 - Each student's clicks up to every checkpoint, and who is still enrolled then`
```python
keys = ['code_presentation', 'id_student']        # identifies one student
feat = stu[keys + ['final_result', 'date_registration', 'date_unregistration']].copy()

for c in CUTOFFS:
    sub = clicks[clicks.date <= c]                 # clicks up to the checkpoint day
    tot = (sub.groupby(keys).sum_click.sum().rename(f'total_clicks_d{c}').reset_index())
    foc = (sub[sub.activity_type == FOCUS_ACT].groupby(keys).sum_click.sum()
           .rename(f'focus_clicks_d{c}').reset_index())
    feat = feat.merge(tot, on=keys, how='left').merge(foc, on=keys, how='left')
    feat[[f'total_clicks_d{c}', f'focus_clicks_d{c}']] = feat[[f'total_clicks_d{c}', f'focus_clicks_d{c}']].fillna(0)   # no clicks -> 0
    # Still enrolled = registered by then AND not yet unregistered (missing unregistration = never left)
    feat[f'enrolled_d{c}'] = ((feat.date_registration.fillna(-999) <= c) &
                              (feat.date_unregistration.isna() | (feat.date_unregistration > c)))

feat.to_csv(os.path.join(OUT_DIR, f"{MODULE}_early_features.csv"), index=False)
print(feat.shape)
```

### PART 1 - Feasibility: is a checkpoint approach worth it for this module?

Markdown cell: `### Stage 1.1 - Can we still reach at-risk students, and does early activity separate outcomes?`
```python
def banded(s):
    # Bands: exactly 0, then quartile bands of students who had some activity
    nz = s[s > 0]
    qs = sorted({int(round(x)) for x in nz.quantile([.25, .5, .75])} - {0}) if len(nz) else []
    return pd.cut(s, bins=[-1, 0] + qs + [np.inf])

def summarize(df, band_col):
    # Per band: students, % ending Pass/Distinction, Fail, Withdrawn
    return df.groupby(band_col, observed=True).final_result.agg(
        n='size',
        pct_pass=lambda s: s.isin(['Pass', 'Distinction']).mean() * 100,
        pct_fail=lambda s: (s == 'Fail').mean() * 100,
        pct_withdrawn=lambda s: (s == 'Withdrawn').mean() * 100).round(1)

total_w = (feat.final_result == 'Withdrawn').sum()
total_f = (feat.final_result == 'Fail').sum()
for c in CUTOFFS:
    e = feat[feat[f'enrolled_d{c}']].copy()            # students still enrolled at this checkpoint
    print(f"\n=== Checkpoint: day {c} (week {c // 7}) ===")
    print("still enrolled:", len(e), "of", len(feat))
    print(f"eventual Withdrawn still enrolled: {(e.final_result == 'Withdrawn').sum()} of {total_w}")
    print(f"eventual Fail still enrolled: {(e.final_result == 'Fail').sum()} of {total_f}")
    e['focus_band'] = banded(e[f'focus_clicks_d{c}'])
    print(f"\nOutcome by {FOCUS_ACT} clicks up to this day:")
    print(summarize(e, 'focus_band').to_string())
    e['total_band'] = pd.qcut(e[f'total_clicks_d{c}'], 4, duplicates='drop')
    print("\nOutcome by total clicks up to this day (quartiles):")
    print(summarize(e, 'total_band').to_string())
```
PASTE BACK: both checkpoint printouts. Read: how many Withdrawn are still reachable; does total-click quartile separate outcomes (BBB: pass rate 33% to 74% at day 28).

Markdown cell: `### Stage 1.2 - When do the Withdrawn students leave? (decides how many a checkpoint can reach)`
```python
w = feat[feat.final_result == 'Withdrawn'].copy()
print("Withdrawn:", len(w), "| with an unregistration date:", int(w.date_unregistration.notna().sum()))
bins = [-np.inf, -0.5, 14, 28, 56, 112, np.inf]       # day of leaving: before start, then windows
labels = ['before day 0', 'day 0-14', 'day 15-28', 'day 29-56', 'day 57-112', 'after day 112']
w['left_when'] = pd.cut(w.date_unregistration, bins=bins, labels=labels)
t = w.left_when.value_counts().reindex(labels).to_frame('n')
t['pct'] = (t.n / t.n.sum() * 100).round(1)
t['cumulative_pct'] = t.pct.cumsum().round(1)
print(t.to_string())
print("median day of leaving:", w.date_unregistration.median())
```
PASTE BACK: the table. (BBB: 37.7% left before day 0, 59.1% by day 28.)

Markdown cell: `### Stage 1.3 - First assessment: due dates, and is early FOCUS activity just the first assessment?`
```python
# The first assessment of each presentation = the one with the earliest due date
assess = pd.read_csv(os.path.join(RAW_DIR, "assessments.csv"))
assess = assess[(assess.code_module == MODULE) & (assess.code_presentation.isin(PRES))]
first = (assess.sort_values(['code_presentation', 'date', 'id_assessment'])
         .groupby('code_presentation').head(1)
         [['id_assessment', 'code_presentation', 'date', 'assessment_type']])
print("First assessment per presentation (date = due day):")
print(first.to_string(index=False))

# Who submitted it, and on which day
sa = pd.read_csv(os.path.join(RAW_DIR, "studentAssessment.csv"))
sa1 = sa.merge(first[['id_assessment', 'code_presentation']], on='id_assessment')
sub1 = (sa1[['code_presentation', 'id_student', 'date_submitted']]
        .rename(columns={'date_submitted': 'first_ass_day'}))
chk = feat.merge(sub1, on=keys, how='left')

# Among students still enrolled at day 56: zero FOCUS clicks vs first assessment submitted
e = chk[chk.enrolled_d56].copy()
e['zero_focus_d56'] = e.focus_clicks_d56 == 0
e['submitted_first_by_d56'] = e.first_ass_day.notna() & (e.first_ass_day <= 56)
print("\nCounts:")
print(pd.crosstab(e.zero_focus_d56, e.submitted_first_by_d56, margins=True))
print("\nRow percentages:")
print((pd.crosstab(e.zero_focus_d56, e.submitted_first_by_d56, normalize='index') * 100).round(1))
```
PASTE BACK: the first-assessment table and both crosstabs. (BBB: first TMA due day 19/19/12.)

Optional Markdown cell: `### Stage 1.4 - Did students who left before day 0 re-enrol somewhere else?`
```python
info_all = pd.read_csv(os.path.join(RAW_DIR, "studentInfo.csv"))     # all modules and presentations
pre = feat[(feat.final_result == 'Withdrawn') & (feat.date_unregistration < 0)][keys].copy()
print("Left before day 0:", len(pre))
others = pre.merge(info_all[['id_student', 'code_module', 'code_presentation', 'final_result']],
                   on='id_student', how='left', suffixes=('_x', '_y'))     # _x = the row we started from
others = others[~((others.code_module == MODULE) & (others.code_presentation_x == others.code_presentation_y))]
n_other = others[['code_presentation_x', 'id_student']].drop_duplicates().shape[0]
print(f"Appear in another module or presentation: {n_other} of {len(pre)} ({n_other / max(len(pre), 1) * 100:.1f}%)")
same_mod = others[others.code_module == MODULE]
later = same_mod[same_mod.code_presentation_y > same_mod.code_presentation_x]
n_later = later[['code_presentation_x', 'id_student']].drop_duplicates().shape[0]
print(f"Appear in a later presentation of this module: {n_later} of {len(pre)}")
```

DECISION GATE after Part 1: if most Withdrawn students are still enrolled at day 28 and the total-click quartile separates outcomes, continue. If a module differs sharply from BBB (for example very few early leavers), tell Disha and agree whether the same checkpoint days still make sense.

### PART 2 - Cohorts and action-window features

Markdown cell: `### Stage 2.1 - Clicks by activity type: early period (up to day 28) and action window (days 29 to 56)`
```python
def clicks_by_type(df, label):
    # One column per activity type = each student's total clicks of that type, named clicks_<type>_<label>
    p = (df.groupby(keys + ['activity_type']).sum_click.sum().unstack('activity_type', fill_value=0))
    p.columns = [f'clicks_{c}_{label}' for c in p.columns]
    return p.reset_index()

early = clicks[clicks.date <= EARLY_END]                              # up to the checkpoint
window = clicks[(clicks.date > EARLY_END) & (clicks.date <= WIN_END)] # the action window

X = stu[keys].copy()                                                  # one row per student
X = X.merge(clicks_by_type(early, 'early'), on=keys, how='left')
X = X.merge(clicks_by_type(window, 'win'), on=keys, how='left')
click_cols = [c for c in X.columns if c.startswith('clicks_')]
X[click_cols] = X[click_cols].fillna(0)                               # no clicks in the period -> 0
print(len(click_cols), "click columns")
```

Markdown cell: `### Stage 2.2 - First assessment as fixed context: submitted by day 28, and the score`
```python
# 'first' and 'sa' come from Stage 1.3
sa1 = sa.merge(first[['id_assessment', 'code_presentation']], on='id_assessment')
first_ass = (sa1[['code_presentation', 'id_student', 'date_submitted', 'score']]
             .rename(columns={'date_submitted': 'first_ass_day', 'score': 'first_ass_score'}))
X = X.merge(first_ass, on=keys, how='left')

# Only what was known at the checkpoint counts
X['first_ass_submitted'] = (X.first_ass_day.notna() & (X.first_ass_day <= EARLY_END)).astype(int)
X['first_ass_score'] = np.where(X.first_ass_submitted == 1, X.first_ass_score, -1)   # -1 = not submitted
X['first_ass_score'] = X.first_ass_score.fillna(-1)
X = X.drop(columns='first_ass_day')
print(X.first_ass_submitted.value_counts().to_dict())
```

Markdown cell: `### Stage 2.3 - Add background, outcomes and cohort labels, then save the table`
```python
demo = ['gender', 'region', 'highest_education', 'imd_band', 'age_band',
        'num_of_prev_attempts', 'studied_credits', 'disability']
X = X.merge(stu[keys + demo + ['final_result', 'date_registration', 'date_unregistration']], on=keys, how='left')

# Still enrolled at each checkpoint (same rule as Stage 0.5)
for c in [28, 56]:
    X[f'enrolled_d{c}'] = ((X.date_registration.fillna(-999) <= c) &
                           (X.date_unregistration.isna() | (X.date_unregistration > c)))

# Cohort labels: the first matching rule wins
is_w = X.final_result == 'Withdrawn'
X['cohort'] = np.select(
    [is_w & (X.date_unregistration <= 28),                         # Withdrawn, left by day 28
     is_w & X.enrolled_d28 & ~X.enrolled_d56,                      # Withdrawn, left between day 29 and 56
     is_w & X.enrolled_d56,                                        # Withdrawn, still enrolled at day 56
     (X.final_result == 'Fail') & X.enrolled_d56,                  # Fail, enrolled at day 56
     X.final_result.isin(['Pass', 'Distinction']) & X.enrolled_d56],   # Success, enrolled at day 56
    ['early_leaver', 'mid_leaver', 'late_leaver', 'fail', 'success'], default='other')
print(X.cohort.value_counts().to_string())

early_c = [c for c in X.columns if c.endswith('_early')]
win_c = [c for c in X.columns if c.endswith('_win')]
X['total_early'] = X[early_c].sum(axis=1)
X['total_win'] = X[win_c].sum(axis=1)
print(X.groupby('cohort')[['total_early', 'total_win', 'first_ass_submitted']].median().round(1).to_string())
X.to_csv(os.path.join(OUT_DIR, f"{MODULE}_checkpoint_features.csv"), index=False)
print(X.shape)
```
PASTE BACK: cohort counts and the median table. (BBB: 968 / 142 / 529 / 1,372 / 2,601 / 5.)

Markdown cell: `### Stage 2.4 - What do Fail and late-leaver students do differently in the window? Median clicks per activity, by cohort`
```python
sub = X[X.cohort.isin(['fail', 'late_leaver', 'success'])].copy()
win_cols = [c for c in X.columns if c.endswith('_win') and c != 'total_win']
short = lambda t: t.replace('clicks_', '').replace('_win', '')     # readable names

med = sub.groupby('cohort')[win_cols].median().T.rename(index=short)[['fail', 'late_leaver', 'success']]
med['fail_%_of_success'] = (med.fail / med.success.replace(0, np.nan) * 100).round(0)
med['late_leaver_%_of_success'] = (med.late_leaver / med.success.replace(0, np.nan) * 100).round(0)
print("Median clicks per student in days 29-56:")
print(med.to_string())

share = (sub.groupby('cohort')[win_cols].apply(lambda d: (d > 0).mean() * 100)
         .T.rename(index=short).round(1)[['fail', 'late_leaver', 'success']])
print("\n% of students with at least one click of that type in days 29-56:")
print(share.to_string())
```
PASTE BACK: both tables. INPUT NEEDED: choose `win_acts` (see section 4).

### PART 3 - Causal graph

Markdown cell: `### Stage 3.1 - Build the graph table: the students enrolled at day 56, encoded and binned (zero-aware)`
```python
# Students in the graph = still enrolled at day 56 (Fail + late leavers + Success)
G = X[X.cohort.isin(['fail', 'late_leaver', 'success'])].copy()
print("graph sample:", len(G), "|", G.cohort.value_counts().to_dict())

# INPUT NEEDED: the window activities used as graph variables (BBB values shown)
win_acts = ['clicks_forumng_win', 'clicks_homepage_win', 'clicks_oucontent_win', 'clicks_quiz_win',
            'clicks_resource_win', 'clicks_subpage_win', 'clicks_url_win']
short = lambda t: t.replace('clicks_', '').replace('_win', '')

# Outcome: Withdrawn 0, Fail 1, Pass 2, Distinction 3
G['final_result'] = G.final_result.map({'Withdrawn': 0, 'Fail': 1, 'Pass': 2, 'Distinction': 3})

# Deprivation band: 'Oct-20' is the date-mangled 10-20% band, so map it to band 1
imd_map = {'0-10%': 0, 'Oct-20': 1, '10-20': 1, '10-20%': 1, '20-30%': 2, '30-40%': 3, '40-50%': 4,
           '50-60%': 5, '60-70%': 6, '70-80%': 7, '80-90%': 8, '90-100%': 9}
raw_imd = X.loc[G.index, 'imd_band']
imd = raw_imd.map(imd_map)
print("unmapped imd labels (should be only truly missing):", raw_imd[imd.isna()].value_counts(dropna=False).to_dict())
G['imd_band'] = imd.fillna(imd.mode()[0]).astype(int)           # missing -> most common band

G['highest_education'] = G.highest_education.map({
    'No Formal quals': 0, 'Lower Than A Level': 1, 'A Level or Equivalent': 2,
    'HE Qualification': 3, 'Post Graduate Qualification': 4})
G['age_band'] = G.age_band.map({'0-35': 0, '35-55': 1, '55<=': 2})
G['gender'] = G.gender.map({'M': 0, 'F': 1})
G['disability'] = G.disability.map({'N': 0, 'Y': 1})
G['region'] = G.region.astype('category').cat.codes                  # nominal label code
G['studied_credits'] = pd.cut(G.studied_credits, bins=[0, 60, 90, np.inf], labels=False)   # 3 groups: <=60, 61-90, >90
G['num_of_prev_attempts'] = G.num_of_prev_attempts.clip(upper=2)      # 0, 1, 2 or more

def bin_zero_aware(s, n_levels=3):
    # Level 0 = no activity; levels 1..3 = equal-size bands among students who were active
    out = pd.Series(0, index=s.index, dtype=int)
    active = s > 0
    if active.sum() > 0:
        try:
            bands = pd.qcut(s[active], q=n_levels, labels=False, duplicates='drop')
            out[active] = bands.astype(int) + 1
        except ValueError:
            out[active] = 1                      # too few distinct values: just 'some activity'
    return out

G['total_early'] = pd.qcut(G.total_early, q=4, labels=False, duplicates='drop')   # almost everyone has early activity: plain quartiles
raw_fa = X.loc[G.index, 'first_ass_score']                     # first assessment: level 0 = not submitted
fa = pd.Series(0, index=G.index, dtype=int)
submitted = raw_fa >= 0
fa[submitted] = pd.qcut(raw_fa[submitted], q=3, labels=False, duplicates='drop').astype(int) + 1
G['first_ass_score'] = fa
for c in win_acts:
    G[c] = bin_zero_aware(X.loc[G.index, c])                   # window activities from the raw counts in X

graph_vars = demo + ['total_early', 'first_ass_score'] + win_acts + ['final_result']
Gdf = G[graph_vars].copy()

# Check every variable: levels, missing values, share at level 0, share in the largest level
chk = pd.DataFrame({'n_levels': Gdf.nunique(), 'missing': Gdf.isna().sum(),
                    'pct_level_0': (Gdf == 0).mean().mul(100).round(1),
                    'pct_in_largest_level': (Gdf.apply(lambda s: s.value_counts(normalize=True).iloc[0]) * 100).round(1)})
print(chk.to_string())
Gdf.to_csv(os.path.join(OUT_DIR, f"{MODULE}_graph_table.csv"), index=False)
```
PASTE BACK: the printout. Rules: any variable with 1 level is dropped; imd_band should have 10 levels; missing should be 0.

Markdown cell: `### Stage 3.2 - Run PC (G-square test, alpha 0.05) with time-ordering rules, then read every edge`
```python
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.cit import chisq, gsq
from causallearn.graph.GraphNode import GraphNode
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge

node_names = list(Gdf.columns)
data = Gdf.values.astype(int)                  # integer codes for the G-square test
nodes = [GraphNode(name) for name in node_names]
node_map = {name: node for name, node in zip(node_names, nodes)}

# Time order: earlier tiers can cause later tiers, never the reverse
tiers = {0: demo,                               # background (known at registration)
         1: ['total_early', 'first_ass_score'], # early engagement and first assessment (up to day 28)
         2: win_acts,                           # window activities (days 29-56)
         3: ['final_result']}                   # outcome
tier_of = {n: t for t, ns in tiers.items() for n in ns}

bk = BackgroundKnowledge()
# Rule A: a later tier cannot cause an earlier tier
for t_late, ns_late in tiers.items():
    for t_early, ns_early in tiers.items():
        if t_late > t_early:
            for late in ns_late:
                for early_ in ns_early:
                    bk.add_forbidden_by_node(node_map[late], node_map[early_])
# Rule B: nothing causes gender (a root)
for n in node_names:
    if n != 'gender':
        bk.add_forbidden_by_node(node_map[n], node_map['gender'])
# Rule C: the first assessment score cannot cause early engagement
bk.add_forbidden_by_node(node_map['first_ass_score'], node_map['total_early'])

# For many variables (20+), add depth=3 to pc(...) so it finishes in reasonable time
cg = pc(data, alpha=0.05, indep_test=gsq, background_knowledge=bk, node_names=node_names, show_progress=False)
print("Number of edges:", cg.G.get_num_edges())

# Read every edge from its endpoints (more reliable than reading the picture)
directed, undirected = [], []
for e in cg.G.get_graph_edges():
    a, b = e.get_node1().get_name(), e.get_node2().get_name()
    ea, eb = e.get_endpoint1().name, e.get_endpoint2().name
    if ea == 'TAIL' and eb == 'ARROW':
        directed.append((a, b))
    elif ea == 'ARROW' and eb == 'TAIL':
        directed.append((b, a))
    else:
        undirected.append((a, b, ea, eb))
print("directed:", len(directed), "| undirected/other:", len(undirected))
bad = [(a, b) for a, b in directed if tier_of[a] > tier_of[b] or b == 'gender']
print("rule violations:", bad)
for a, b in directed:
    print(f"{a} [TAIL] -- [ARROW] {b}")
for a, b, ea, eb in undirected:
    print(f"{a} [{ea}] -- [{eb}] {b}")
print("\ndirect parents of final_result:", [a for a, b in directed if b == 'final_result'])
connected = {x for a, b in directed for x in (a, b)} | {x for a, b, _, _ in undirected for x in (a, b)}
print("isolated variables:", [n for n in node_names if n not in connected])
```
PASTE BACK: the text output. If the window activities come out isolated from the outcome, tell Disha: the fix is a composite "total window clicks" node (same idea as `total_clicks` in RQ1).

Markdown cell: `### Stage 3.3 - Draw the graph: one colour per time tier, tiers stacked, edges touching final_result in green`
```python
import io, pydot
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from causallearn.utils.GraphUtils import GraphUtils

dot_path = r"C:\Users\Disha Sharma\.conda\envs\analytics\Library\bin\dot.exe"   # full path: plain 'dot' fails here
pyd = GraphUtils.to_pydot(cg.G, labels=node_names)

def to_idx(x):
    # pydot may give a node reference as an int or as quoted text: turn both into a plain int
    return int(str(x).strip('"'))

tier_color = {0: 'lightgray', 1: 'lightblue', 2: 'lightsalmon', 3: 'lightgreen'}
for node in pyd.get_nodes():
    s = str(node.get_name()).strip('"')
    if not s.isdigit():
        continue                                     # skip pydot's default entries
    label = node_names[int(s)]
    node.set_style('filled'); node.set_fillcolor(tier_color[tier_of[label]])
    node.set_fontsize('10'); node.set_fixedsize('false'); node.set_margin('0.15,0.08')

for edge in pyd.get_edges():                         # edges touching the outcome in green
    src = node_names[to_idx(edge.get_source())]
    dst = node_names[to_idx(edge.get_destination())]
    if 'final_result' in (src, dst):
        edge.set_color('forestgreen'); edge.set_penwidth('2')

for t, names in tiers.items():                       # every variable of one tier on the same row
    sg = pydot.Subgraph(rank='same')
    for n in names:
        sg.add_node(pydot.Node(str(node_names.index(n))))
    pyd.add_subgraph(sg)

png_data = pyd.create_png(prog=dot_path)             # not pyd.write_png(): it calls plain 'dot'
with open(os.path.join(OUT_DIR, f"{MODULE}_checkpoint_PC_graph.png"), "wb") as f:
    f.write(png_data)
plt.figure(figsize=(16, 12))
plt.imshow(mpimg.imread(io.BytesIO(png_data)))
plt.axis('off')
plt.show()
```
Ask Disha before making any other figure. Open the saved PNG outside Jupyter if labels look crowded.

Markdown cell: `### Stage 3.4 - Freeze the DAG: PC edges + early-context edges (+ outcome edges after Stage 4.5); roles and adjustment sets`
```python
import networkx as nx

# Decisions from Stage 4.5 (empty on the first run, filled in later, then rerun this cell)
outcome_edge_acts = globals().get('outcome_edge_acts', [])
recourse_levers = globals().get('recourse_levers', [])

dag = nx.DiGraph()
dag.add_nodes_from(node_names)
dag.add_edges_from(directed)                     # edges found by PC (undirected pairs left out: no usable direction)

# Addition 1: what a student did before the window can influence what they do in it
ctx_edges = [(src, c) for c in win_acts for src in ['total_early', 'first_ass_score'] if not dag.has_edge(src, c)]
dag.add_edges_from(ctx_edges)
# Addition 2: activities with a clear effect on the outcome when all others are held fixed (Stage 4.5)
outcome_edges = [(c, 'final_result') for c in outcome_edge_acts if not dag.has_edge(c, 'final_result')]
dag.add_edges_from(outcome_edges)
assert nx.is_directed_acyclic_graph(dag), "graph has a cycle"
print("edges from PC:", len(directed), "| context edges added:", len(ctx_edges), "| outcome edges added:", len(outcome_edges))

edges_out = pd.DataFrame([(a, b, 'PC') for a, b in directed] +
                         [(a, b, 'added: early context') for a, b in ctx_edges] +
                         [(a, b, 'added: direct effect (Stage 4.5)') for a, b in outcome_edges],
                         columns=['from', 'to', 'source'])
edges_out.to_csv(os.path.join(OUT_DIR, f"{MODULE}_checkpoint_dag_edges.csv"), index=False)

# Role of every variable, whether it reaches the outcome, and its adjustment set (= its direct parents)
role = {0: 'immutable context', 1: 'fixed context (known at checkpoint)', 2: 'action-window activity', 3: 'outcome'}
rows = []
for n in node_names:
    if n == 'final_result':
        continue
    rows.append({'variable': n, 'role': role[tier_of[n]],
                 'path_to_outcome': nx.has_path(dag, n, 'final_result'),
                 'direct_to_outcome': dag.has_edge(n, 'final_result'),
                 'adjustment_set': sorted(dag.predecessors(n))})
roles = pd.DataFrame(rows)
roles.to_csv(os.path.join(OUT_DIR, f"{MODULE}_checkpoint_roles_adjustment_sets.csv"), index=False)
levers = roles[(roles.role == 'action-window activity') & roles.path_to_outcome].variable.tolist()
print("treatments for effect estimation:", levers)
print("recourse levers:", recourse_levers)
print(roles[['variable', 'path_to_outcome', 'direct_to_outcome']].to_string(index=False))
```

Markdown cell (optional): `### Stage 3.5 - Sensitivity: the same PC run with alpha 0.10 (sparse variables may gain edges)`
```python
cg10 = pc(data, alpha=0.10, indep_test=gsq, background_knowledge=bk, node_names=node_names, show_progress=False)

def edge_sets(cg_):
    # Split PC edges into directed pairs and undirected pairs
    d, u = set(), set()
    for e in cg_.G.get_graph_edges():
        a, b = e.get_node1().get_name(), e.get_node2().get_name()
        ea, eb = e.get_endpoint1().name, e.get_endpoint2().name
        if ea == 'TAIL' and eb == 'ARROW':
            d.add((a, b))
        elif ea == 'ARROW' and eb == 'TAIL':
            d.add((b, a))
        else:
            u.add(tuple(sorted((a, b))))
    return d, u

d05, u05 = set(directed), {tuple(sorted((a, b))) for a, b, _, _ in undirected}
d10, u10 = edge_sets(cg10)
print("edges at 0.05:", len(d05) + len(u05), "| at 0.10:", len(d10) + len(u10))
print("directed edges gained at 0.10:", sorted(d10 - d05))
print("directed edges lost at 0.10:", sorted(d05 - d10))
print("direct parents of final_result at 0.10:", sorted(a for a, b in d10 if b == 'final_result'))
```

### PART 4 - Effects

Markdown cell: `### Stage 4.1 - Effect of each variable on final_result (DoWhy linear backdoor estimate, 95% interval, placebo and random-common-cause refutation)`
```python
import logging, warnings
from dowhy import CausalModel
warnings.filterwarnings('ignore')
logging.getLogger('dowhy').setLevel(logging.ERROR)

treatments = roles[roles.path_to_outcome].variable.tolist()      # every variable with a path to the outcome
print(len(treatments), "treatments:", treatments)

def estimate_one(t, adj):
    # One treatment at a time, adjusting only for its parents (the adjustment set)
    model = CausalModel(data=Gdf, treatment=t, outcome='final_result', common_causes=adj if adj else None)
    ident = model.identify_effect(proceed_when_unidentifiable=True)
    est = model.estimate_effect(ident, method_name="backdoor.linear_regression", confidence_intervals=True)
    ci = np.array(est.get_confidence_intervals(), dtype=float).flatten()
    out = {'ATE': float(est.value), 'CI_low': float(ci[0]), 'CI_high': float(ci[1])}
    try:   # refutation 1: a random fake treatment should have about zero effect
        r = model.refute_estimate(ident, est, method_name="placebo_treatment_refuter", placebo_type="permute", num_simulations=50)
        out['placebo_effect'] = float(r.new_effect)
        out['placebo_p'] = r.refutation_result.get('p_value') if isinstance(r.refutation_result, dict) else None
    except Exception as e:
        out['placebo_effect'], out['placebo_p'] = None, str(e)[:40]
    try:   # refutation 2: adding a random extra confounder should barely change the effect
        r = model.refute_estimate(ident, est, method_name="random_common_cause", num_simulations=50)
        out['random_cc_effect'] = float(r.new_effect)
    except Exception:
        out['random_cc_effect'] = None
    return out

results = []                         # reset before every rerun, otherwise reruns add duplicate rows
for t in treatments:
    adj = roles.loc[roles.variable == t, 'adjustment_set'].iloc[0]
    row = {'variable': t, 'role': roles.loc[roles.variable == t, 'role'].iloc[0],
           'adjustment_set': ', '.join(adj) if adj else '(none)'}
    row.update(estimate_one(t, adj))
    row['significant'] = (row['CI_low'] > 0) or (row['CI_high'] < 0)
    results.append(row)
    print("done:", t)
ate = pd.DataFrame(results)
cols = ['variable', 'role', 'ATE', 'CI_low', 'CI_high', 'significant', 'placebo_effect', 'placebo_p', 'random_cc_effect', 'adjustment_set']
print(ate[cols].round(3).to_string(index=False))
ate.to_csv(os.path.join(OUT_DIR, f"{MODULE}_checkpoint_ATE.csv"), index=False)
```
Effects are per one-level step of the 0-3 coding. Placebo effect should be about 0 and random-common-cause effect equal to the ATE.

Markdown cell: `### Stage 4.2 - Robustness: effects under heavier adjustment, and why the focus activity effect shrinks step by step`
```python
import statsmodels.api as sm

context = ['total_early', 'first_ass_score']                 # known before the window: safe to adjust for
background = ['gender', 'highest_education', 'imd_band', 'age_band', 'num_of_prev_attempts', 'studied_credits', 'disability']
reg_dummies = pd.get_dummies(Gdf['region'], prefix='reg', drop_first=True).astype(float)   # region is nominal: dummies

def ols_effect(t, adj, with_region=False):
    # Same linear model DoWhy uses: outcome ~ treatment + adjustment variables
    parts = [Gdf[[t] + adj].astype(float)]
    if with_region:
        parts.append(reg_dummies)
    X_ = sm.add_constant(pd.concat(parts, axis=1))
    m = sm.OLS(Gdf['final_result'].astype(float), X_).fit()
    lo, hi = m.conf_int().loc[t]
    return round(m.params[t], 3), round(lo, 3), round(hi, 3)

rows = []                                                    # reset before every rerun
for t in levers:
    graph_adj = roles.loc[roles.variable == t, 'adjustment_set'].iloc[0]
    plus_ctx = sorted(set(graph_adj) | set(context))
    plus_all = sorted(set(plus_ctx) | set(background))
    rows.append({'activity': t, 'graph_set': ols_effect(t, graph_adj),
                 '+context': ols_effect(t, plus_ctx), '+context+background': ols_effect(t, plus_all, with_region=True)})
print(pd.DataFrame(rows).to_string(index=False))

focus_win = f'clicks_{FOCUS_ACT}_win'
g_adj = roles.loc[roles.variable == focus_win, 'adjustment_set'].iloc[0]
steps = [('nothing', []), ('+ early clicks', ['total_early']), ('+ first assessment', context),
         ('+ graph set', sorted(set(g_adj) | set(context))), ('+ background', sorted(set(g_adj) | set(context) | set(background)))]
q = [{'adjusted for': l, 'effect (ATE, CI low, CI high)': ols_effect(focus_win, a, with_region=('background' in l))} for l, a in steps]
print("\nFocus activity effect, step by step:")
print(pd.DataFrame(q).to_string(index=False))
```

Markdown cell (optional, needs the module's whole-course RQ2 value): `### Stage 4.3 - Where did the whole-course focus effect go? Same simple model on three samples, then "any activity vs none"`
```python
base = feat.copy()                                           # all students, focus clicks up to day 56
base['y'] = base.final_result.map({'Withdrawn': 0, 'Fail': 1, 'Pass': 2, 'Distinction': 3})

def simple_effect(d, qcol, label):
    # Quartile-bin the focus clicks inside this sample, then regress the outcome on the bin number
    b = pd.qcut(d[qcol], q=4, labels=False, duplicates='drop').astype(float).rename('bin')
    m = sm.OLS(d['y'].astype(float), sm.add_constant(b)).fit()
    lo, hi = m.conf_int().loc['bin']
    return {'sample': label, 'n': len(d), 'effect_per_bin': round(m.params['bin'], 3), 'CI': (round(lo, 3), round(hi, 3))}

rows = [simple_effect(base, 'focus_clicks_d56', 'all students, clicks up to day 56'),
        simple_effect(base[base.enrolled_d56], 'focus_clicks_d56', 'enrolled at day 56, clicks up to day 56')]
win_df = pd.DataFrame({'y': G.final_result, 'q': X.loc[G.index, focus_win]})
rows.append(simple_effect(win_df, 'q', 'enrolled at day 56, window clicks only'))
print(pd.DataFrame(rows).to_string(index=False))

any_focus = (X.loc[G.index, focus_win] > 0).astype(float).rename('any_focus')
adj_focus = sorted(set(g_adj) | set(context))
Xd = sm.add_constant(pd.concat([any_focus, Gdf[adj_focus].astype(float)], axis=1))
mm = sm.OLS(Gdf['final_result'].astype(float), Xd).fit()
lo, hi = mm.conf_int().loc['any_focus']
print("share with any focus clicks in the window:", round(any_focus.mean() * 100, 1), "%")
print("effect of any vs none:", round(mm.params['any_focus'], 3), (round(lo, 3), round(hi, 3)))
```
INPUT NEEDED: the module's whole-course RQ2 value for this activity, to compare.

Markdown cell: `### Stage 4.4 - Each activity level versus no activity (adjusted for early clicks and first assessment only)`
```python
context_vars = ['total_early', 'first_ass_score']

def level_effects(t):
    # Effect of level 1, 2, 3 versus level 0 (no activity)
    d = pd.get_dummies(Gdf[t].astype(int), prefix='lvl').astype(float).drop(columns='lvl_0')
    X_ = sm.add_constant(pd.concat([d, Gdf[context_vars].astype(float)], axis=1))
    m = sm.OLS(Gdf['final_result'].astype(float), X_).fit()
    out = {'activity': short(t), '% with none': round((Gdf[t] == 0).mean() * 100, 1)}
    for lv in ['lvl_1', 'lvl_2', 'lvl_3']:
        if lv in m.params.index:
            lo, hi = m.conf_int().loc[lv]
            out[lv.replace('lvl_', 'level ') + ' vs none'] = f"{m.params[lv]:.3f} ({lo:.3f}, {hi:.3f})"
        else:
            out[lv.replace('lvl_', 'level ') + ' vs none'] = 'n/a'
    return out

print("Effect on final_result (0-3) versus no activity; level 1 = lowest third of active students")
print(pd.DataFrame([level_effects(t) for t in win_acts]).to_string(index=False))
```

Markdown cell: `### Stage 4.5 - All activities together, split into participation (any vs none) and volume (each extra level)`
```python
parts = {}
for t in win_acts:
    lvl = Gdf[t].astype(int)
    parts[short(t) + '_any'] = (lvl > 0).astype(float)               # participation: any activity vs none
    parts[short(t) + '_more'] = (lvl - 1).clip(lower=0).astype(float)  # volume: levels above the first

Xj2 = sm.add_constant(pd.concat([pd.DataFrame(parts), Gdf[context_vars].astype(float)], axis=1))
mj2 = sm.OLS(Gdf['final_result'].astype(float), Xj2).fit()            # one regression, all activities together
ci = mj2.conf_int()
res = pd.DataFrame({'effect': mj2.params, 'lo': ci[0], 'hi': ci[1]})
rows = []
for t in win_acts:
    r = {'activity': short(t)}
    for kind, label in [('_any', 'participation (any vs none)'), ('_more', 'volume (per extra level)')]:
        e = res.loc[short(t) + kind]
        r[label] = f"{e.effect:.3f} ({e.lo:.3f}, {e.hi:.3f})"
    rows.append(r)
print(pd.DataFrame(rows).to_string(index=False))
print("R-squared:", round(mj2.rsquared, 3))
res.round(3).to_csv(os.path.join(OUT_DIR, f"{MODULE}_checkpoint_participation_volume_effects.csv"))
```
PASTE BACK: the 4.1, 4.2, 4.4 and 4.5 outputs.

DECISION GATE (Stage 5.0, INPUT NEEDED): from Stage 4.5, decide with Disha
- `part_nodes`: activities where starting matters (participation interval clearly above 0, volume not clear). BBB: quiz, forum, content pages.
- `vol_nodes`: activities where volume matters. BBB: homepage (log-in regularity).
- `outcome_edge_acts`: activities with a clear participation or volume effect (positive or negative) that PC did not link to the outcome. BBB: forum, content pages, resource.
- `recourse_levers` = `part_nodes + vol_nodes`. Activities with no independent effect (BBB: URL, subpage) or a negative one (BBB: resource) are not recommended.

Markdown cell: `### Stage 5.0 - Record the decisions, then rerun Stage 3.4 and Stage 4.1`
```python
# INPUT NEEDED: BBB values shown
part_nodes = ['clicks_quiz_win', 'clicks_forumng_win', 'clicks_oucontent_win']      # start-type levers (coded 0/1 in the model)
vol_nodes = ['clicks_homepage_win']                                                  # volume-type lever (raise by one level)
outcome_edge_acts = ['clicks_forumng_win', 'clicks_oucontent_win', 'clicks_resource_win']   # direct edges to add
recourse_levers = part_nodes + vol_nodes
# Now rerun Stage 3.4 (it reads outcome_edge_acts and recourse_levers), then Stage 4.1.
```

### PART 5 - Causal model, what-if questions, recourse plans

Markdown cell: `### Stage 5.1 - Fit the invertible causal model on the frozen DAG and check it (fit, invertibility, graph structure)`
```python
import dowhy.gcm as gcm
import pickle
try:
    from dowhy.graph import get_ordered_predecessors          # path in this DoWhy version
except ImportError:
    get_ordered_predecessors = lambda g, n: sorted(g.predecessors(n))

# Model variables = variables with a path to the outcome, plus the outcome
scm_cols = roles[roles.path_to_outcome].variable.tolist() + ['final_result']
dag_scm = dag.subgraph(scm_cols).copy()
df_scm = Gdf[scm_cols].astype(float)                           # gcm wants numeric data
print("nodes:", dag_scm.number_of_nodes(), "| edges:", dag_scm.number_of_edges())

scm = gcm.InvertibleStructuralCausalModel(dag_scm.copy())      # invertible: each student keeps their own noise
gcm.auto.assign_causal_mechanisms(scm, df_scm)
for n in dag_scm.nodes:
    mech = scm.causal_mechanism(n)
    print(n, "|", type(mech).__name__, "|", type(getattr(mech, "prediction_model", None)).__name__)

gcm.fit(scm, df_scm)
ev = gcm.evaluate_causal_model(scm, df_scm, compare_mechanism_baselines=True)   # slow: keep the laptop awake
with open(os.path.join(OUT_DIR, f"{MODULE}_checkpoint_scm.pkl"), "wb") as f:
    pickle.dump(scm, f)
with open(os.path.join(OUT_DIR, f"{MODULE}_checkpoint_scm_evaluation.txt"), "w") as f:
    f.write(str(ev))
for line in str(ev).splitlines():
    if any(k in line for k in ["--- Node", "R2 coefficient", "model assumption", "violates", "overall average KL"]):
        print(line)
```
PASTE BACK: the filtered printout. Expect: structure test passes; invertibility rejected at most nodes (BBB: all); weak R-squared at the outcome (about 0.1-0.2).

Markdown cell: `### Stage 5.2a - Helpers: read the evaluation report, and build a model on its OWN copy of the graph`
```python
import re

def to_num(s):
    # Report sentences end with a full stop right after the number
    return float(s.rstrip('.'))

def parse_eval(text):
    # Pull R-squared and the invertibility result per node out of the report text
    r2, inv, cur = {}, {}, None
    for line in text.splitlines():
        m = re.match(r'--- Node (\S+)', line)
        if m:
            cur = m.group(1)
            continue
        m = re.search(r'The R2 coefficient is ([\-0-9.eE+]+)', line)
        if m and cur:
            r2[cur] = to_num(m.group(1))
        m = re.search(r'assumption for node (\S+) is (not rejected|rejected) with a p-value of ([0-9.eE+\-]+)', line)
        if m:
            inv[m.group(1)] = (m.group(2), to_num(m.group(3)))
    return r2, inv

def build_scm(variant, data, graph):
    # IMPORTANT: every model gets its OWN copy of the graph. Models built on one shared graph overwrite each other.
    s = gcm.InvertibleStructuralCausalModel(graph.copy())
    gcm.auto.assign_causal_mechanisms(s, data)                # roots keep their empirical distributions
    if variant == 'auto':
        return s
    for n in graph.nodes:
        if graph.in_degree(n) == 0:
            continue
        reg = gcm.ml.create_linear_regressor() if variant == 'linear' else gcm.ml.create_hist_gradient_boost_regressor()
        s.set_causal_mechanism(n, gcm.AdditiveNoiseModel(reg))
    return s

def comparison_table(reports, graph):
    # One row per non-root node, one R-squared and invertibility column pair per model
    rows = []
    for n in [x for x in graph.nodes if graph.in_degree(x) > 0]:
        row = {'node': n}
        for v, (r2, inv) in reports.items():
            row[v + ' R2'] = round(r2.get(n, np.nan), 3)
            row[v + ' invertible'] = f"{'no' if inv.get(n, ('', 0))[0] == 'rejected' else 'yes'} (p={inv.get(n, ('', np.nan))[1]:.3g})"
        rows.append(row)
    return pd.DataFrame(rows)
```

Markdown cell: `### Stage 5.2b - Compare mechanism types on the level-coded data: automatic, linear, boosted (fit and invertibility per node)`
```python
reports = {}
for v in ['auto', 'linear', 'boosted']:
    s = build_scm(v, df_scm, dag_scm)
    gcm.fit(s, df_scm)
    # skip the slow structure test and KL here: only per-node fit and invertibility are needed
    ev_v = gcm.evaluate_causal_model(s, df_scm, compare_mechanism_baselines=False,
                                     evaluate_causal_structure=False, evaluate_overall_kl_divergence=False)
    reports[v] = parse_eval(str(ev_v))
    with open(os.path.join(OUT_DIR, f"{MODULE}_checkpoint_scm_{v}_levels.pkl"), "wb") as f:
        pickle.dump(s, f)
comp = comparison_table(reports, dag_scm)
print(comp.to_string(index=False))
comp.to_csv(os.path.join(OUT_DIR, f"{MODULE}_checkpoint_scm_mechanism_comparison.csv"), index=False)
```
If the call complains about `evaluate_causal_structure` or `evaluate_overall_kl_divergence`, delete those two arguments.

Markdown cell: `### Stage 5.2c - Rebuild the models with participation coding (0/1) for the start-type activities, each on its own graph copy`
```python
# Start-type activities (Stage 5.0) become 0 = none, 1 = any: levels would hide the start effect
df_scm2 = df_scm.copy()
for c in part_nodes:
    df_scm2[c] = (df_scm2[c] > 0).astype(float)

scms2, reports2 = {}, {}
for v in ['linear', 'boosted']:
    s = build_scm(v, df_scm2, dag_scm)
    gcm.fit(s, df_scm2)
    ev_v = gcm.evaluate_causal_model(s, df_scm2, compare_mechanism_baselines=False,
                                     evaluate_causal_structure=False, evaluate_overall_kl_divergence=False)
    scms2[v] = s
    reports2[v] = parse_eval(str(ev_v))
    with open(os.path.join(OUT_DIR, f"{MODULE}_checkpoint_scm_{v}_participation.pkl"), "wb") as f:
        pickle.dump(s, f)

# The outcome mechanism of the linear model: effect per parent (start-type parents are now 'start vs none')
lin2 = scms2['linear'].causal_mechanism('final_result').prediction_model.sklearn_model
par2 = get_ordered_predecessors(scms2['linear'].graph, 'final_result')
print("linear outcome mechanism (participation coding):")
print(pd.Series(lin2.coef_.ravel(), index=par2).round(3).to_string())
print("\n", comparison_table(reports2, dag_scm).to_string(index=False))
```
PASTE BACK: the coefficients and the table. Check: coefficients for start-type activities should sit close to the Stage 4.5 participation effects (BBB: quiz 0.207 vs 0.220; content 0.169 vs 0.146; forum 0.089 vs 0.096). Main model = linear (best outcome fit in BBB: R-squared 0.213); boosted = robustness check.

Markdown cell: `### Stage 5.3 - What-if shifts: start each start-type activity, raise volume-type activities one level, and bundles (Fail and late leavers)`
```python
cohort_lab = G.loc[df_scm2.index, 'cohort']              # 'fail', 'late_leaver' or 'success' per row

def raise_to(nodes, level=1):
    # Raise-only: students already at or above the level are left unchanged
    return {n: (lambda x, l=level: np.maximum(x, l)) for n in nodes}

def step_up(nodes):
    # One level more, capped at level 3
    return {n: (lambda x: np.minimum(x + 1, 3)) for n in nodes}

# The actions we may recommend (from Stage 5.0)
actions = {short(n): raise_to([n]) for n in part_nodes}
actions.update({short(n) + '_up': step_up([n]) for n in vol_nodes})

def merge_actions(names):
    iv = {}
    for a in names:
        iv.update(actions[a])
    return iv

scenarios = {a: actions[a] for a in actions}
scenarios['all starts'] = merge_actions([short(n) for n in part_nodes])
scenarios['all actions'] = merge_actions(list(actions))

rows = []                                                  # reset before every rerun
for model_name, model in scms2.items():
    for cname in ['fail', 'late_leaver']:
        d = df_scm2[cohort_lab == cname].reset_index(drop=True)
        for sname, iv in scenarios.items():
            cf = gcm.counterfactual_samples(model, iv, observed_data=d)      # student-level what-if (approximate)
            shift = cf['final_result'].values - d['final_result'].values
            elig = np.zeros(len(d), dtype=bool)                              # students the action actually changes
            for n, fn in iv.items():
                elig |= (np.asarray(fn(d[n].values)) != d[n].values)
            rows.append({'model': model_name, 'cohort': cname, 'scenario': sname, 'n': len(d),
                         '% affected': round(elig.mean() * 100, 1),
                         'mean shift (all)': round(shift.mean(), 3),
                         'mean shift (affected)': round(shift[elig].mean(), 3) if elig.any() else np.nan})
res2 = pd.DataFrame(rows)
res2.to_csv(os.path.join(OUT_DIR, f"{MODULE}_checkpoint_whatif_shifts_participation.csv"), index=False)
print(res2[res2.model == 'linear'].drop(columns='model').to_string(index=False))
print("\nMean shift among affected students, linear vs boosted:")
print(res2.pivot_table(index=['cohort', 'scenario'], columns='model', values='mean shift (affected)').round(3).to_string())
```
Do not report "share reaching Pass" from these shifts: with additive noise every Fail student ends at 1 plus the same shift, so the share jumps from 0% to 100% (an artifact). Use Stage 5.4 for probabilities. Check that linear and boosted DIFFER (identical results mean a shared-graph bug).

Markdown cell: `### Stage 5.4 - Chance of reaching the target before and after each action (noise from the whole sample, no student-level noise recovery)`
```python
def outcome_mean(model, data):
    # Predicted outcome without noise. .ravel() because the linear model returns a column, the boosted one a flat list
    mech = model.causal_mechanism('final_result')
    par = get_ordered_predecessors(model.graph, 'final_result')
    return np.asarray(mech.prediction_model.predict(data[par].to_numpy())).ravel()

def prob_reach(f_vals, resid, T):
    # For each student: share of noise draws for which predicted outcome + noise reaches the target T
    assert f_vals.ndim == 1 and resid.ndim == 1, "predictions and residuals must both be flat"
    return (f_vals[:, None] + resid[None, :] >= T).mean(axis=1)

targets = {'fail': 1.5, 'late_leaver': 0.5}      # Fail: reach Pass (>= 1.5); late leaver: stay, not withdraw (>= 0.5)
rows, calib = [], []                             # reset before every rerun
for model_name, model in scms2.items():
    f_all = outcome_mean(model, df_scm2)
    resid = df_scm2['final_result'].values - f_all                  # pool of unexplained variation, all enrolled students
    for T, real in [(1.5, (df_scm2.final_result >= 2).mean()), (0.5, (df_scm2.final_result >= 1).mean())]:
        calib.append({'model': model_name, 'target': T,
                      'model average': round(prob_reach(f_all, resid, T).mean(), 3), 'real share': round(real, 3)})
    for cname, T in targets.items():
        d = df_scm2[cohort_lab == cname].reset_index(drop=True)
        f0 = outcome_mean(model, d)
        p0 = prob_reach(f0, resid, T)                               # chance now
        for sname, iv in scenarios.items():
            cf = gcm.counterfactual_samples(model, iv, observed_data=d)
            delta = cf['final_result'].values - d['final_result'].values
            p1 = prob_reach(f0 + delta, resid, T)                   # chance after the action
            elig = np.zeros(len(d), dtype=bool)
            for n, fn in iv.items():
                elig |= (np.asarray(fn(d[n].values)) != d[n].values)
            rows.append({'model': model_name, 'cohort': cname, 'scenario': sname, '% affected': round(elig.mean() * 100, 1),
                         'chance before (%)': round(p0.mean() * 100, 1), 'chance after (%)': round(p1.mean() * 100, 1),
                         'gain, all (pts)': round((p1 - p0).mean() * 100, 1),
                         'gain, affected (pts)': round((p1 - p0)[elig].mean() * 100, 1) if elig.any() else np.nan})
print("Sanity check (model average vs real share; should be within about 3 points):")
print(pd.DataFrame(calib).to_string(index=False))
res_p = pd.DataFrame(rows)
res_p.to_csv(os.path.join(OUT_DIR, f"{MODULE}_checkpoint_probability_gains.csv"), index=False)
print("\nMAIN MODEL (linear):")
print(res_p[res_p.model == 'linear'].drop(columns='model').to_string(index=False))
print("\nGain among affected students (points), linear vs boosted:")
print(res_p.pivot_table(index=['cohort', 'scenario'], columns='model', values='gain, affected (pts)').round(1).to_string())
```
Never build a 4,502 x 4,502 x 4,502 array: if a MemoryError appears, check that predictions are flat (`.ravel()`).

Markdown cell: `### Stage 5.5 - Recourse plans: for each student, the smallest set of actions that raises the chance by at least g points`
```python
from itertools import combinations
from collections import Counter

def effective(d, iv):
    # Which students does this action actually change?
    e = np.zeros(len(d), dtype=bool)
    for n, fn in iv.items():
        e |= (np.asarray(fn(d[n].values)) != d[n].values)
    return e

gains_grid = [5, 10, 15, 20]                       # minimum improvement in the chance, in points
cover_rows, plan_counts = [], {}                   # reset before every rerun
for model_name, model in scms2.items():
    f_all = outcome_mean(model, df_scm2)
    resid = df_scm2['final_result'].values - f_all
    for cname, T in targets.items():
        d = df_scm2[cohort_lab == cname].reset_index(drop=True)
        f0 = outcome_mean(model, d)
        p0 = prob_reach(f0, resid, T)
        elig = {a: effective(d, iv) for a, iv in actions.items()}
        plan_gain, plan_valid = {}, {}
        for r in range(1, len(actions) + 1):
            for S in combinations(actions, r):         # every combination of actions
                iv = merge_actions(S)
                cf = gcm.counterfactual_samples(model, iv, observed_data=d)
                delta = cf['final_result'].values - d['final_result'].values
                plan_gain[S] = (prob_reach(f0 + delta, resid, T) - p0) * 100
                plan_valid[S] = np.all([elig[a] for a in S], axis=0)    # every action in the plan applies to the student
        none_available = ~np.any(list(elig.values()), axis=0)
        for g in gains_grid:                            # share of students with a plan of at most k actions reaching g points
            row = {'model': model_name, 'cohort': cname, 'min gain (pts)': g}
            for k in range(1, len(actions) + 1):
                ok = np.zeros(len(d), dtype=bool)
                for S in plan_gain:
                    if len(S) <= k:
                        ok |= plan_valid[S] & (plan_gain[S] >= g)
                row[f'<= {k} actions (%)'] = round(ok.mean() * 100, 1)
            cover_rows.append(row)
        plans = []                                      # most common minimal plan at 10 points: fewest actions, then larger gain
        for i in range(len(d)):
            best = None
            for S in plan_gain:
                if plan_valid[S][i] and plan_gain[S][i] >= 10:
                    key = (len(S), -plan_gain[S][i])
                    if best is None or key < best[0]:
                        best = (key, S)
            plans.append(' + '.join(best[1]) if best else ('no action applies' if none_available[i] else 'no plan reaches 10 pts'))
        plan_counts[(model_name, cname)] = Counter(plans)

cover = pd.DataFrame(cover_rows)
cover.to_csv(os.path.join(OUT_DIR, f"{MODULE}_checkpoint_recourse_coverage.csv"), index=False)
print(cover.to_string(index=False))
for (m, c), cnt in plan_counts.items():
    if m == 'linear':
        print(f"\nMost common minimal plans, {c}, gain >= 10 points ({m} model):")
        for plan, n in cnt.most_common(6):
            print(f"  {plan}: {n} students ({n / sum(cnt.values()) * 100:.1f}%)")
```
PASTE BACK: the coverage table and the common plans.

### PART 6 - Early leavers (risk analysis, fair window)

Markdown cell: `### Stage 6.1 - Early leavers: build the table (registration-time factors only) and the fair window (enrolled on day 0)`
```python
import re
import statsmodels.formula.api as smf

# One row per student; factors are all known at or before the start of the course
E = X[keys + demo + ['date_registration', 'date_unregistration']].copy()
E['left_early'] = (E.date_unregistration <= 28).astype(int)          # left by day 28 (missing date = never left)
print("students:", len(E), "| left by day 28:", int(E.left_early.sum()), f"({E.left_early.mean() * 100:.1f}%)")

E['imd_band'] = E.imd_band.replace({'Oct-20': '10-20%', '10-20': '10-20%'}).fillna('missing')   # 'Oct-20' = 10-20%
E['studied_credits'] = pd.cut(E.studied_credits, bins=[0, 60, 90, np.inf], labels=['<=60', '61-90', '>90'])
E['num_of_prev_attempts'] = E.num_of_prev_attempts.clip(upper=2).map({0: '0', 1: '1', 2: '2+'})
days_before = (-E.date_registration).fillna((-E.date_registration).median())   # days registered before the start
E['reg_timing'] = pd.cut(days_before, bins=[-np.inf, 0, 30, 60, 90, np.inf],
                         labels=['day 0 or later', '1-30 days before', '31-60 days before',
                                 '61-90 days before', 'over 90 days before'])
factors = ['gender', 'age_band', 'highest_education', 'imd_band', 'disability', 'region',
           'num_of_prev_attempts', 'studied_credits', 'reg_timing']

# Fair window: only students enrolled on day 0, so everyone has the same 28 days at risk.
# (Using all students makes registration timing look important only because early registrants had longer to leave.)
E2 = E[(E.date_registration.fillna(-999) <= 0) & (E.date_unregistration.isna() | (E.date_unregistration >= 0))].copy()
E2['left_28'] = (E2.date_unregistration <= 28).astype(int)           # left during days 0-28
for v in factors:
    E2[v] = E2[v].astype(str)                                        # plain text categories for the models
# INPUT NEEDED: merge tiny groups (BBB: 3 students aged 55+, 10 with a postgraduate qualification)
E2['age_band'] = E2.age_band.replace({'55<=': '35-55'})
E2['highest_education'] = E2.highest_education.replace({'Post Graduate Qualification': 'HE Qualification'})
print("enrolled on day 0:", len(E2), "| left in days 0-28:", int(E2.left_28.sum()), f"({E2.left_28.mean() * 100:.1f}%)")

rate = lambda s: s.mean() * 100
print("\nrate by presentation:")
print(E2.groupby('code_presentation').left_28.agg(n='size', rate_pct=rate).round(1).to_string())
for v in factors:
    print(f"\n{v}")
    print(E2.groupby(v).left_28.agg(n='size', rate_pct=rate).round(1).to_string())
E2.to_csv(os.path.join(OUT_DIR, f"{MODULE}_early_leaver_table.csv"), index=False)
```

Markdown cell: `### Stage 6.2 - Early leavers: adjusted effects in points (presentation controlled), and a cross-validated risk ranking`
```python
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# Linear probability model: each effect = change in the chance of leaving, in points, versus the most common level
factors2 = factors + ['code_presentation']
refs2 = {v: E2[v].value_counts().idxmax() for v in factors2}
formula2 = 'left_28 ~ ' + ' + '.join(f"C({v}, Treatment(reference='{refs2[v]}'))" for v in factors2)
m2 = smf.ols(formula2, data=E2).fit(cov_type='HC1')               # robust standard errors
ci2 = m2.conf_int()
rows = []
for name in m2.params.index:
    if name == 'Intercept':
        continue
    var, ref, lvl = re.match(r"C\((\w+), Treatment\(reference='(.*)'\)\)\[T\.(.*)\]", name).groups()
    rows.append({'factor': var, 'level': lvl, 'compared with': ref, 'effect (pts)': round(m2.params[name] * 100, 1),
                 'CI low': round(ci2.loc[name, 0] * 100, 1), 'CI high': round(ci2.loc[name, 1] * 100, 1),
                 'clear': (ci2.loc[name, 0] > 0) or (ci2.loc[name, 1] < 0)})
eff2 = pd.DataFrame(rows)
eff2.to_csv(os.path.join(OUT_DIR, f"{MODULE}_early_leaver_effects_fair_window.csv"), index=False)
print("Factors with a clear effect, largest first:")
print(eff2[eff2.clear].assign(a=lambda d: d['effect (pts)'].abs()).sort_values('a', ascending=False)
      .drop(columns=['a', 'clear']).to_string(index=False))

# Risk ranking: every student is scored by a model that never saw them (5-fold cross-validation)
Xr2 = pd.get_dummies(E2[factors2], drop_first=True).astype(float)
y2 = E2.left_28.values
risk2 = cross_val_predict(LogisticRegression(max_iter=2000), Xr2, y2,
                          cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42), method='predict_proba')[:, 1]
print(f"\ncross-validated AUC = {roc_auc_score(y2, risk2):.3f}  (0.5 = no better than chance)")
order2 = np.argsort(-risk2)
out = []
for pct in [10, 20, 30, 40]:
    top = order2[: int(len(y2) * pct / 100)]
    out.append({'contact top % by risk': pct, 'students': len(top),
                'early-leave rate in group (%)': round(y2[top].mean() * 100, 1),
                'share of early leavers reached (%)': round(y2[top].sum() / y2.sum() * 100, 1)})
print(pd.DataFrame(out).to_string(index=False))
print(f"baseline rate: {y2.mean() * 100:.1f}%")
```

---

## 6. What to collect from every module (for the BBB-style summary and the Stage 2 comparison)

1. Student count and outcome counts; cohort sizes (Stage 2.3).
2. Share of Withdrawn who leave before day 0, by day 28, by day 56 (Stage 1.2); first assessment due days (Stage 1.3).
3. Early-signal table: pass rate by total-click quartile at day 28 and 56 (Stage 1.1).
4. Median clicks per cohort (Stage 2.3) and window activity tables (Stage 2.4).
5. Graph: edge counts, direct parents of the outcome, isolated variables, picture (Stage 3).
6. Effect table (4.1), robustness (4.2), participation and volume (4.5).
7. Model: structure test, outcome R-squared per mechanism, invertibility pattern (5.1, 5.2).
8. Probability table (5.4) and recourse coverage and common plans (5.5).
9. Early leavers: fair-window rate, strongest factors, AUC and lift (6.2).

After each module write a short `{MODULE}_Final_Summary.md` in the same style as `BBB_Final_Summary.md`.

---

## 7. Pitfalls found in BBB (do not repeat)

1. **Shared graph bug:** models built on the same graph object overwrite each other's mechanisms, so three "different" models gave identical results. Always build each model with `graph.copy()` (`build_scm`).
2. **MemoryError in Stage 5.4:** the linear model returns predictions as a column; flatten with `.ravel()`.
3. **Pydot node references** may be ints: convert with `int(str(x).strip('"'))`.
4. **Deprivation band** `Oct-20` means 10-20%.
5. **Quartile binning of sparse activities** collapses (cut points identical). Use zero-aware binning.
6. **First assessment:** keep only the score (with 0 = not submitted); do not also keep a "submitted" flag, because it is a perfect copy and confuses the independence tests.
7. **Region** is nominal; its linear effect is not interpretable. Use dummies in regressions.
8. **Level coding hides start effects**: use participation coding (0/1) inside the causal model for start-type activities.
9. **Additive-noise counterfactuals** give every Fail student the same shift, so "share reaching Pass" is an artifact. Use the probability view.
10. **Invertibility is rejected at every non-root node.** This is expected for bounded 0-3 coded variables and hidden shared motivation. Report it as a limitation; do not keep tuning.
11. **Registration timing** looks important in all-student analyses only because of exposure time. Use the fair window.
12. **Directions inside the click group** change with coding and alpha. Only the direct parents of the outcome are stable; never interpret within-group arrows.
13. `show_progress=True` in PC can show a "Loading widget..." placeholder; use `show_progress=False`. For 20+ variables add `depth=3` to `pc(...)`.
14. Reset any list that a loop appends to before rerunning (`rows = []`).
15. Whole-course DiCE: the classifier accuracy (AUC) was not recorded for BBB; record it if DiCE is ever used again.

---

## 8. Reporting rules

- Report group-level results, not individual predictions: "for students with the same profile".
- Always say the results are estimates from past data, not tested on real students, and that the model explains only about a fifth of the outcome because marks are not in it.
- State deviations from the earlier pipeline (list in `BBB_Final_Summary.md`, section 8) in the methods.
- A professor update is short bullet points in plain language, findings only, one message per milestone. Hold numbers until their checks are done.

## 9. Stage 2 (after all modules): Social Sciences vs STEM

Compare, with the BBB items from section 6 for every module: share of early leavers and when they leave; early-signal strength; direct parents of the outcome and effect sizes; which activities are start-type or volume-type; probability gains and recourse coverage for Fail and late leavers; early-leaver factors (region, education, age). Use the adaptive drill-down given by the professor: group level first (Social Sciences vs STEM), then module level, then presentation level only if nothing concrete is found.
Observation items from RQ1 to carry over: the demographic-bypass pattern appeared in STEM combined DAGs (EEE, FFF); AAA is the small-sample outlier.

## 10. Status checklist

| Module | Presentations | Checkpoint pipeline | Early leavers | Summary written |
|---|---|---|---|---|
| BBB | 2013B, 2013J, 2014B | Done | Done | Done |
| CCC | INPUT NEEDED (likely 2014B, 2014J) | Not started | Not started | - |
| DDD | INPUT NEEDED (RQ1 excluded 2013B) | Not started | Not started | - |
| EEE | INPUT NEEDED | Not started | Not started | - |
| FFF | INPUT NEEDED | Not started | Not started | - |
| GGG | INPUT NEEDED | Not started | Not started | - |
| AAA | Optional (small sample) | Not started | Not started | - |
