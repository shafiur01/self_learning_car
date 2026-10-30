# Pilot013 — response-based pseudo-label selection did not improve the baseline

> Git-tracked records: [configuration, results and provenance](../results/013_evidence_profile/README.md). Original run/data links below require the local private archives.

Executed2026-10-15, America/Chicago. Five fixed seeds,65 fresh fits, no new
recordings or validation-to-training transfer. [Frozen protocol](../EVIDENCE_PROFILE_PROTOCOL.md),
[complete run report](../experiments/runs/evidence_profile/20261016T030844123147Z/REPORT.md),
[scientific figure](../experiments/runs/evidence_profile/20261016T030844123147Z/analysis/figure.png).

## Data and comparison

405 original training clips contain F86/B63/L46/R41/STOP44/NONE41/UP50/DOWN34.
133 model-selected corrections preserved but excluded from the primary pool.
Each seed reveals exactly10 labels/command80 total, five/class in each of two
whole-session teacher groups.325 additional inputs have labels hidden from the
learner; they are training inputs, not test examples. Fresh weights throughout.
All8 command labels are represented, but random low-label sampling does not
guarantee every style: seed22's10 UP labels lack index-UP; several five-example
teachers miss a variant. These are recorded coverage limitations, not corrected
after seeing the outcomes. The full405 pool contains allthree UP variants.

Existing48 selects checkpoints; existing96 is a fixed retrospective endpoint,
scored only after every target/model froze. Both were historically inspected;
no untouched-test claim.80 training+48 development labels are128 supervised
labels before96 endpoint labels. No existing538-label teacher/encoder warmstart.
All direct models have49,840 parameters; identical source feature extraction.

Mean endpoint results across allfive seeds:

| Method | Accuracy (%) | MacroF1 | NONE→movement (%) |
| --- | ---: | ---: | ---: |
| Supervised80 | 74.17 | .7247 | 68.33 |
| Supervised80,1680 updates | 74.79 | .7320 | 66.67 |
| Full-teacher confidence self-training | 74.17 | .7247 | 68.33 |
| Subset-teacher confidence self-training | 74.79 | .7295 | 68.33 |
| Proposed response profile | 74.38 | .7266 | 68.33 |
| Embedding support | 74.17 | .7247 | 68.33 |
| Count/weight-matched confidence | 74.38 | .7266 | 68.33 |
| Count/weight-matched random | 74.38 | .7266 | 68.33 |
| Full405 supervised reference | 93.33 | .9326 | 8.33 |

Primary paired profile−subset-confidence macroF1−.002873;0/5 positive, four ties.
By sitting−.012097/+.006987. No advantage over matched selectors. The small
profile versus sparse-supervised numerical difference is not a supported gain.
STOP recall96.67% for most low-label variants,98.33% for full405; no robot action
or general-reliability evidence. NONE errors remain especially large with80 labels.

## The useful negative finding: almost no training signal was admitted

Native teachers agree on182–204 of325 inputs/seed, but requiring both confidence
scores>=.95 admits0/2/1/0/0. The profile rule admits0/1/0/0/0, average0.2/325.
Full-only confidence and embedding support admit nothing. These settings are
not usable at this supervision/teacher-session split. This does not establish
that every rejected label is wrong, or that response profiles can never help.

Post-freeze oracle diagnostics of the **three** confidence-admitted cases:

- Seed22 DOWN clip82249ff... predictedDOWN, score.96434: profile accepts.
- Seed22 NONE clipad05d45... predictedDOWN, score.96717: profile rejects;
  the matched highest-confidence and fixed-random controls select this error.
- Seed33 BACKWARD clipcf86870... predictedBACKWARD, score.95658: profile rejects.

The one retained profile target is correct, but one case cannot establish a
precision advantage or publication claim. The report's0% profile pseudo-error
means0/1, not measured reliability. Confidence-only has1/3 pooled admitted errors;
its25% table value is the mean of the two nonempty seed error rates, not pooled
1/3. Empty pools have undefined precision and are recorded asnull.

Student loss averages weights over the fixed325, so tiny coverage yields tiny
unsupervised signal. Zero-admission fits reproduce supervised80 exactly.
Active/no-active SSL branches can consume different dropout streams, as declared;
tiny downstream changes cannot be assigned purely to pseudo-label correctness.
Where profile is active, matched selectors share count, weight multiset, target
class, initialization and sampling. No useful student advantage appears there.

## Integrity and costs

Seven meaningful synthetic software checks passed before fitting. Reproduced
all65 checkpoints,7,440 saved development/endpoint vectors and allfive selector
bundles. Re-extracted549 arrays from405training+48development+96endpoint raw
recordings identically. All8 classes guaranteed in each teacher/student seed.
All sources/settings/sampled IDs/weights/targets/checkpoint histories and every
unfavorable result archived. Original recordings/checkpoints/012 ledger unchanged.

65 physical fits use152.90s numerical fit loops total. Standalone charged mean
fit+selector envelope: supervised80~1.93s, profile~6.24s, full405~1.93s. This
includes the profile's two teachers but excludes setup/I/O and broader software/
historical costs; shared selector envelope contains unused diagnostic work.
The long supervised control matches optimizer-update count, not FLOPs/time.
No measured training-memory/inference-speed saving; no actual annotation saving.

## Decision and next controlled step

Do not promote this prototype as an improvement or activate its weights. Keep
the negative run intact.30 new checking clips/class are not needed to diagnose
this admission bottleneck, and moving validation into training would obscure
the comparison. All existing405 inputs already include every command.

Recommended next factor: raise revealed training labels to160 (20/class) from
the same405 pool, leaving245 labels hidden. Keep80-label subsets nested and
hold the architecture/gates/masks/sessions/evaluation policy fixed. This tests
whether teacher/anchor supervision is the bottleneck; it is not a promised fix.
Changing to adaptive admission is a separate later control, not a simultaneous
unreported rescue. Freeze the next run before fitting; current endpoint remains
retrospective. No next-budget run has been executed yet.
Metadata feasibility confirms that the same fixed sampling permutations at160
retain each original80 subset, include20/class, and include allthree UP variants
in both teachers for allfive seeds, without style-based reselection.

The paper-relevant record is this result, source/config/input fingerprints and
complete tables—not a successful new algorithm claim. Strong external baselines,
broader benchmarks and novelty evidence remain outstanding. No new lesson file.
