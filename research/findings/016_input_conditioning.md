# Experiment 016: a small accuracy gain conceals worse background recognition

> Git-tracked records: [configuration, results and provenance](../results/016_input_conditioning/README.md). Original run/data links below require the local private archives.

Completed 2026-10-17, America/Chicago. [Frozen plan](../INPUT_CONDITIONING_PROTOCOL.md),
[complete report](../experiments/runs/input_conditioning/20261017T063852365459Z/REPORT.md),
[figure](../experiments/runs/input_conditioning/20261017T063852365459Z/analysis/figure.png).

## Question and correction

Earlier camera/hand comparisons changed landmark reference frames and explicit
motion features. We did not isolate a physical camera-viewpoint intervention;
later96 results had mixed motion gains. Existing landmarks already normalize
hand position/size, but numerical units differ between continuous feature types.
This test asks whether additional conditioning or explicit-clock removal helps
the low-label student and weak teachers. Normalization is a standard baseline,
not an invented learning algorithm.

Four conditions, five fixed seeds, native-subset teacherA/B and supervised160
student per seed:60 fresh fits. Same160 labels (20/class), teachers80 each,
same405 inputs, all commands/UP styles,149-input/49,840-weight LSTM,560 updates,
48 checkpoint-selection clips. Normalization estimates seven positive RMS block
divisors from each model's own revealed training inputs only. Geometry/motion
masks exclude invalid samples; clips receive equal weight; no centering, label,
direction or raw-feature change. Clock removal zeros elapsed/sample interval
columns144/145, leaving physical-time derivatives intact.

The15 raw controls reproduce parent014 weight-state hashes exactly. All60
weights/scales/settings froze before separate96 retrospective scoring and hidden
teacher diagnostics. Training plus development uses208 labels; endpoint96 labels
and earlier pretraining/development costs count separately. No new recordings,
validation-to-training transfer, SSL student, gate tuning, dependency, lesson or
camera activation. This remains a same-user retrospective comparison.

## Full comparison

Means across five seeds; the96 clips contain12 unique examples per class.

| Input condition | Student accuracy (%) | Macro F1 | NONE→movement (%) | UP recall (%) | STOP recall (%) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original, clock retained | 86.25 | .8580 | 28.33 | 66.67 | 100.00 |
| RMS, clock retained | 87.08 | .8636 | 43.33 | 75.00 | 100.00 |
| Original, clock removed | 86.25 | .8594 | 23.33 | 73.33 | 100.00 |
| RMS, clock removed | 85.42 | .8488 | 48.33 | 73.33 | 98.33 |

Primary RMS-clock minus raw-clock macroF1 **+.005575**, positive3/5 seeds.
Differences:+.056674,-.079364,-.004923,+.042011,+.013479. Sitting1 mean-.003072;
sitting2+.015408. Accuracy SD rises3.16→6.01 percentage points. There is no
consistent broad improvement: false movement on NONE rises by15 percentage
points. NONE recall declines71.67→56.67%, while LEFT78.33→91.67% and UP66.67→75%.
Student NLL improves .5257→.4306; this does not establish calibrated confidence
or compensate for the background-command errors in robot control.

Clock removal alone leaves accuracy unchanged and changes macroF1+.001359,
positive in only1/5 seeds. NONE movement declines28.33→23.33% and UP recall rises,
but FORWARD recall declines85→75%. Removing the clock with RMS reduces macroF1
by.014797. The factorial student interaction is-.016155. All contrasts/sittings
and cases are retained; no favorable seed/condition is selected for activation.

## Teachers and the original admission bottleneck

| Condition | Opposite-session anchor F1 | Hidden mean-teacher F1 | Confidence-admitted pairs | Wrong labels | Predicted UP / NONE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original + clock | .7265 | .7429 | 89 | 5 | 0 /0 |
| RMS + clock | .6968 | .7118 | 33 | 1 | 0 /0 |
| Original, no clock | .7046 | .7249 | 44 | 3 | 0 /0 |
| RMS, no clock | .7188 | .7157 | 72 | 8 | 0 /0 |

RMS-clock anchor F1 is lower in4/5 paired seeds; hidden mean-teacher F1 lower
in3/5. Hidden NONE recall falls23.81→19.05%, UP66.67→63.33%. These conditions do
not repair teacher recognition or the missing UP/NONE pseudo-label coverage.
Admission counts are repeated seed–clip pairs from the original hidden pools,
not independent examples or a new student training outcome. No response-profile
or class-specific admission rule was changed here.

## What input scales and gate measurements establish

Across student training sets, RMS values are roughly .73 for normalized pose,
.27–.30 wrist displacement, .08–.10 log-scale,1.40–1.47 velocities, .58–.62 scale
velocity and .5774 orientation. Differing units exist, but that alone does not
prove optimization failure. The original model can learn different weights.

Final teachers on opposite-session anchors: RMS-clock reduces saturated input
gates3.95→2.36% and candidate gates8.80→8.21%; forget gates2.07→2.13% and output
gates3.51→4.08% rise. Teacher recognition still declines. This descriptive
measure does not establish a causal explanation or a universal saturation fix.
The gate equations were checked against the LSTM's complete hidden sequence;
see the [official equations](https://docs.pytorch.org/docs/stable/generated/torch.nn.LSTM.html).
The [MediaPipe definitions](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker/python#handle_and_display_results)
also distinguish normalized image XYZ from world coordinates; this experiment
does not synthesize or validate real camera viewpoint changes.

## Integrity and decision

Seven software checks passed. Reproduced60 checkpoints,17800 saved prediction
vectors,15 own-training-only scale estimates and every within-role paired
initialization/batch/mask stream. A runtime fitting allowlist admitted only five
learner bundles and48-selection data; zero oracle/endpoint loads during fitting.
All parent artifacts remain unchanged. Fit loops116.97s total, excluding
preparation/scaling/I/O/evaluation/report/prior work; no total efficiency saving.
PNG/PDF figure checked. Post-hoc error/change CSVs support later case review
without altering model choices or existing labels.

Do not promote this RMS recipe as a robotics improvement. A slightly higher
aggregate accuracy can hide worse NO_COMMAND behavior; report both. Keep the
raw baseline and all four outcomes. Feature units/explicit clocks are not a
supported complete explanation of the weak teachers. The next controlled
hypothesis should target background/command discrimination and class-specific
learning value, using the stronger confidence/embedding baselines from015.
No next fit is declared or executed. This result establishes no novelty,
independent reliability, unseen-user generalization, annotation saving, SOTA or
publication readiness.
