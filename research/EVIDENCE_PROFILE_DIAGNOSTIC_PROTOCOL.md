# Diagnostic 015: coverage and ranking information

Declared 2026-10-16, America/Chicago, before computing the new diagnostic scores.
Designed after the complete outcomes of pilots 013/014 were inspected. This is
retrospective development analysis, not preregistration of an independent test.
Primary source: pilot014 `evidence_profile_budget160/20261016T133936682213Z`.
No training, new recordings, new dependencies, checkpoint changes or activation.
Do not load or rescore the 48 development or 96 endpoint feature/score arrays.

Questions:

1. For each command, where do the original teachers/gates lose hidden training
   examples: disagreement, confidence, missing correct anchors, or profile radius?
2. Within the same candidate pool and predicted-command quotas, do response
   distances rank correct pseudo-labels better than confidence or embeddings?
3. Are retained examples already correctly recognized by supervised160? This
   is a redundancy proxy, not proof of uselessness or causal training benefit.

Use all five saved seeds and their 245 hidden TRAINING inputs (1225 seed–clip
pairs with repeated original clips). Known anchor labels remain the original
opposite-session 80 labels. Reuse saved native-subset teacher full/omit logits,
embeddings and anchor outputs. No new teacher or calibration fit. Infer only
saved supervised160 students on these hidden inputs, never evaluation endpoints.
Keep the model, source inputs, class definitions and prior selector unchanged.

Before opening oracle labels in this new run, compute and freeze all scores,
selected row IDs and supervised160 probabilities. The prior oracle outcomes
were already examined; this access separation prevents accidental outcome-based
ranking changes, not prior knowledge or an untouched-label claim. Hash complete
parent artifacts and new sources/config. Preserve all old ledgers and scores.

For candidate class c=argmax(mean teacher full probabilities), define confidence
as min(pA(c),pB(c)). Dense profile distance is max over teachers of nearest
same-class anchor squared standardized response distance divided by that class's
original leave-one-out radius. Use exactly the original MAD floor, anchor
confidence/minimum-count rule, feature responses and radius floor. Compute dense
embedding distance identically from the saved embeddings. Unsupported class
scores are infinite, not zero or reliable negatives. Reproduce original gated
weights explicitly before new analysis. These dense scores outside the gates
are diagnostics, not a changed production pseudo-label selector.

Primary candidate domain: teacher agreement AND class support in BOTH teachers
for BOTH profile and embedding. Apply this identical domain to every ranker.
Report all exclusions by true/predicted class and never hide unsupported cases.
Secondary domain: the same support domain intersected with the original .95
dual-confidence gate. Include its zero/error-free limitations.

Primary ranking comparison: select ceil(25% of available examples) separately
within each predicted class, using lower profile distance, lower embedding
distance, higher confidence, or one fixed seeded random order. All rankers have
identical predicted-class counts. Stable original row order breaks ties. Random
seed is original seed+15,000,000; no random-seed search. Fractions .10/.50/1.0
are descriptive coverage curves, never tuned to choose a favorable point.

Additional fixed control: within primary domain, match quotas jointly by
predicted class and confidence interval [0,.5), [.5,.7), [.7,.85), [.85,.95),
[.95,1]. Select 25% per cell. This controls confidence only coarsely; it does not
prove conditional independence or incremental predictive value at exact scores.
No fitted correctness classifier, confidence sweep or chosen new threshold.

Report per-seed/pooled error counts and retained counts, matched differences,
each class, selection IDs, unique clips and exclusions. Undefined empty error
rates remain null, not perfect precision. Repeated pairs/overlapping seeds are
not independent trials; no population p-values or significance claim. Record
all frozen coverage points and all unfavorable outcomes.

Trace true-class hidden examples through agreement, confidence, anchor and
radius stages using a disjoint first-failure partition. Separately distinguish
correct and wrong agreement. Report teacher/mean prediction recalls, confidence
quantiles for correct agreement, and actual/predicted command counts. Check
original target files unchanged. Plot true-class coverage and paired ranking
errors with explicit denominators, plus supervised160 correctness of accepted
and rejected correct pseudo-labels. Correctness on these reused training inputs
is not independent accuracy or a measure of robot behavior.

The fixed 25% error comparison assesses recorded-label ranking only. It cannot
establish that selected clips cause larger student gains. Learning utility needs
a later separately controlled training study. No algorithm novelty, annotation
or compute savings, SOTA, broad generalization or publication claim. Do not start
a new threshold/budget/student experiment as an unreported continuation.
