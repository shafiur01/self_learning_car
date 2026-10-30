# Diagnostic 015: confidence blocks UP; profiles add little beyond simple rankings

> Git-tracked records: [configuration, results and provenance](../results/015_profile_diagnostic/README.md). Original run/data links below require the local private archives.

Executed 2026-10-16, America/Chicago. [Frozen diagnostic plan](../EVIDENCE_PROFILE_DIAGNOSTIC_PROTOCOL.md),
[full report](../experiments/runs/evidence_profile_diagnostic/20261016T142255449351Z/REPORT.md),
[figure](../experiments/runs/evidence_profile_diagnostic/20261016T142255449351Z/analysis/figure.png).
No new fits, recordings, dependencies, checkpoint changes or app activation.
No 48-development or 96-endpoint features/predictions were loaded or rescored.

## Scope and controls

Use pilot014's saved native-subset teachers/anchors and supervised160 models.
Five hidden-training pools provide 1225 seed–clip pairs involving 385 distinct
clips. These overlap across seeds and were recorded by one person; they are not
independent trials. Prior outcomes/labels were already partly inspected. This is
retrospective development analysis, not independent hypothesis confirmation.

All new distances, ranking selections and supervised160 predictions froze before
this diagnostic's oracle access. The score-phase runtime audit permits only
training targets/learner arrays and traps training calls. Dense profile/embedding
distances reproduce the original gated weights before any ranking analysis.
Every ranker uses the same agreed, jointly supported candidate domain and the
same per-predicted-command quotas. Primary fraction25%; 10/50/100% points were
fixed in advance and all retained. An additional25% control matches fixed
confidence-bin counts as well. Unsupported cases are reported, not assigned
perfect precision or silently removed from only one method.

## Different causes of missing UP and NO_COMMAND targets

| Actual command | Hidden pairs | Mean-teacher prediction correct | Both teachers correctly agree | Original confidence gate admits | Profile admits |
| --- | ---: | ---: | ---: | ---: | ---: |
| UP | 150 | 100 | 61 | 0 | 0 |
| NO_COMMAND | 105 | 25 | 5 | 4 | 0 |

UP: all 61 correct agreements fail the0.95 gate. Their minimum-teacher class
confidence ranges .2378–.9120. Original first failures are73 disagreements and
77 low-confidence agreements, including wrong agreements. This demonstrates
where the frozen selector loses UP coverage, not that every low-confidence UP
proposal is correct. A common high cutoff blocks some correct proposals.

NO_COMMAND: teachers correctly agree on only5/105 pairs; the mean prediction is
correct25/105. Of39 agreements,34 are wrong. The four pairs passing the original
confidence gate are actually NO_COMMAND but predicted DOWN; the profile radius
rejects allfour. Correct agreements have confidence .3379–.6235. This is also a
teacher recognition problem, so lowering confidence alone is not a demonstrated
fix. It could admit more incorrect commands.

Agreement alone includes846/1225 pairs. Requiring both methods' class support
leaves627;219 agreed pairs are excluded from every ranking. For predicted UP,
42 agreed pairs have shared support, all in seeds33/44. Predicted NO_COMMAND has
only2 supported agreed pairs, both in seed33. Missing support is masked by the
earlier confidence gate in the original first-failure trace: zero cases reach
the unsupported-anchor stage after passing0.95. This does not mean anchor
support is adequate across the full pool. Full true/predicted-class tables kept.

## Ranking correctness at controlled coverage

Primary25%-per-class quota retains164 pairs from the same627 candidates:

| Ranker | Wrong labels / retained pairs | Error (%) |
| --- | ---: | ---: |
| Confidence | 7 /164 | 4.27 |
| Response profile | 5 /164 | 3.05 |
| Embedding | 1 /164 | .61 |
| Fixed random | 17 /164 | 10.37 |

Profile minus confidence error counts by seed:0,-1,+1,-2,0. The two fewer pooled
errors are a small descriptive difference, not a reliable advantage. Embeddings
rank better at this declared point. Predicted UP contributes11 retained pairs:
profile/embedding0errors, confidence1, random2. NO_COMMAND contributes only1
retained pair per ranker, with no error; far too little to evaluate that class.
Correctness of these shadow selections is not a trained-model accuracy result.

All frozen coverage points (wrong labels / retained pairs):

| Nominal per-class fraction | Confidence | Profile | Embedding | Random |
| --- | ---: | ---: | ---: | ---: |
| .10 | 3/73 | 1/73 | 0/73 | 8/73 |
| .25 (primary) | 7/164 | 5/164 | 1/164 | 17/164 |
| .50 | 15/320 | 23/320 | 17/320 | 39/320 |
| 1.00 | 80/627 | 80/627 | 80/627 | 80/627 |

At50% coverage, confidence beats both distance rankings. Do not select a
favorable fraction afterward or describe embeddings as uniformly best. Ceil
quotas mean actual fractions exceed nominal values in small classes.

Matching predicted-class AND coarse confidence-bin quotas retains192 pairs:
confidence21errors, profile20, embedding14, random22. Profile's pooled advantage
is only one error; per-seed differences+1,-2,-1,-1,+2. Bins do not exactly match
continuous confidence, so this is not a proof of conditional predictive value.
Restricting to the original gate retains24 at25%: confidence1error, profile0,
embedding0, random2. That tiny/error-sparse domain cannot establish superiority.

## Are the retained labels new information for the student?

All37 originally retained profile pairs are already classified correctly by the
saved supervised160 student;35 have student confidence>=.95. All39 embedding
pairs are already correct;36 are high-confidence. Confidence-only:83/89 already
correct. Of706 correct-agreed pairs excluded by profiles,678 are already correct.

This is compatible with selection favoring examples the student already knows,
but does not prove they are useless. Correct examples can still help through
regularization/confidence changes. This diagnostic does not intervene on training
or estimate causal learning benefit. High pseudo-label purity is an incomplete
objective; the earlier controlled student results remain the evidence for gains.

## Integrity and decision

Seven meaningful software checks pass. Reproduced five score/anchor bundles,
180 exact ranking selections,1225 supervised160 vectors and original selector
weights. Scores freeze before oracle access; all parent files remain unchanged.
Score/inference phase .208s, excluding hash/setup/report/verification work; no
efficiency saving inferred. Every class/seed/fraction and exclusion is archived.

Current response profiles have not earned priority over confidence and embedding
baselines. The next research hypothesis should address command coverage and weak
teacher recognition, and test actual learning value at fixed budgets. A future
class-aware admission change must preserve wrong-command evaluation and compare
against the stronger simple baselines. It is a candidate direction, not a new
declared algorithm or a proven fix. No additional student/gate/budget experiment
has started. No novelty, independent generalization, robot reliability, saved
annotation effort, SOTA or publication-readiness claim follows from this study.
