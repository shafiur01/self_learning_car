# Current research state

Updated 2026-10-30, America/Chicago. Read PROTOCOL.md and latest LOG.md too.
History/full tables stay in LOG.md and findings/; keep this file concise.

## Goal and constraints

Develop/test label-efficient temporal gesture learning with robotics relevance.
No novelty, SOTA, actual annotation savings, broad reliability or publication-
readiness claim without evidence. User is learning ML: explain tests/limits.
Preserve recordings/checkpoints/negative results. Existing .venv, code under
experiments/, no dependencies without need. Do not edit any frozen older Python
source, now including online_softmatch.py, its audit and test modules.

User authorizes training as needed, but no more personal validation/test recording
requests. Optional30/class offer remains unused. No validation-to-training transfer.
Existing48/96 already inspected: development/retrospective, never untouched final
benchmarks. Keep sessions together; no camera activation or new lesson files.
Representation is normalized MediaPipe imageXYZ, not calibrated world coordinates.

## Repository preparation2026-10-30 — ready for user review/push

User explicitly will commit/push after preparation; no actual Gitindex/branch/
commit/remote change byagent. Read docs/GITHUB_PUSH_GUIDE.md andREPOSITORY_LAYOUT.
Added separate tools/export_research_records,check_repository,run_software_checks;
requirements-research.txt and Gitignore/attributes protect privatefiles/hashbytes.
Historical science modules were NOT moved/refactored: pinnedpaths/hashes remain.
20record exports/634files≈8.27MiB inresearch/results;594payloads compared against
originals with numericvalues preserved andlocalpaths normalized. Personalnotes,
rawdata/weights/fullruns stayignored. Privateartifactinventory is localonly,
storage_uri remainsnull; nooff-devicebackup performed. No fullfreshclonehistorical
training claimed: some oldloaders pinoriginalabsolutepaths/parentarchives.
198syntheticsoftware checks pass indatafreecopy; three fullymockedcamera tests
need a temporary assetfixture supplied bynewrunner, notproductionguard changes.
ScratchGitstaging/content/whitespace/exclusionchecks pass. Existingoriginalfiles
preserved. Currentoriginpublic; guidealso hasprivate-remote option. No newresearchfit,
personalrecording/evaluation or modelactivation. NextMLstep remains below.

## Experiment018 — complete; feasibility passes, ranking reverses at final

Run experiments/runs/online_softmatch/20261020T205543410735Z.
Plan ONLINE_SOFTMATCH_PROTOCOL.md; concise finding018_online_softmatch.md.
15fresh fits: supervised160/onlineconfidence/SoftMatchadaptation ×5seeds.
Same014original405/160labels20perclass/245hidden; excludes133corrections.
Raw149LSTM49,840weights/32frames,560Adam.001updates,batch32,twoCPUthreads.
140sharedsupervisedwarmup then420jointwith140ramp. Identical sequence inputs,
deterministic weak/currentmodel and classifierdropout .2 strong; pairedprivate
label/strong RNG. PaperAlgorithm1 rawconfidence stats then UAweight/rawtarget;
explicit difference from currentUSB aligned-statistics hook. Firstbatchbootstrap,
m=2^(-32/245) onehiddenpool-pass half-life, no parameterEMA in anymethod.
This is an adaptation/control, not full paper reproduction or invention.

Means on48inspected development, methodorder supervised/confidence/soft:
- CE-selected accuracy85.00/86.25/87.50%; F1 .8456/.8575/.8721.
- Selected NONE→movement30.00/20.00/13.33%; STOP86.67/86.67/90.00%.
- Final560 accuracy84.17/89.58/88.33%; F1 .8370/.8950/.8818.
- Final NONE→movement20.00/16.67/13.33%; STOP86.67/90.00/83.33%.
- Originalselectedcontinuationgate passes: softF1+.014560 vsconfidence, paired
  differences[-.045688,+.073443,0,+.021212,+.023830],3positive/1negative/1tie;
  NONE improves6.67pp/STOP3.33pp. Feasibility only. Versussupervised +.026436,
  only2positive/1negative/2ties. Finalsoft-minus-confidence F1-.013291 and
  STOP-6.67pp: no robust advantage/promotion established.
- Soft selectedsteps320/560/100/160/120; seeds33/55 select BEFORE SSL, identical
  supervised checkpoints. Seed55's selectedgain overconfidence is not a better
  classifier learned frompseudo-labels. Never retroactively change primary rule.

All15models/30selected+final states/traces froze before hiddenoracle analysis.
No96payload load/rescoring or newlabels. Confidence vssoft trainingmass57475/
62255.97, wrong7074/9283.35 (12.31/14.91%), correctUP5524/5906.43, actualNONE→
movementtargetmass1606/2168.46. Repeatedexposures, notindependentclips; models'
candidates evolve differently, so no causal selector-quality/utility ranking.
SelectedF1gain despite higherpseudo-error reinforces purity-is-insufficient.

Nine preflight checks pass. All5supervisedselected states reproduce014 exactly;
pairedinit/warmup/labeldraw/dropout streams match. Reproduced30checkpoint outputs/
1440vectors/134400targets+weights, independent30confusions/67200scalarsoftweights;
trainingNumPyallowlist only ownlearner+48selection. Parent014 intact; no source
or checkpoint replaced. Fits48.74s incldevchecks, subprocesswall65.39s; setup/
audit/review additional. Peak340.25MiB. Forwardtraining samples89600supervised vs
224000eachSSLgroup(2.5x); equalsteps not equalcompute/no general efficiencyclaim.
Allseed/class/failureCSV, tracedlosses, hashes, accesslogs andcheckedfigure saved.

Next justified work: publicmulti-person temporalgesture benchmark compatibility/
split/license/pretraining/compute audit, then frozen checkpoint-evaluation rule
before more algorithm selection. Keepconfidence+soft as baselines, no rescuefit/
96selection. Do not map arbitrary public gesturelabels to our eight commands.
No public data downloaded or new mechanism declared. Main contribution unproven.

## Review/diagnostic017 — why this comparison was chosen

Focused primary ICML/ICLR/CoRL review in finding017, access limits explicit.
SoftMatch/FreeMatch adaptiveweighting is published; Ren2018/MPL/PabLO2025/
IA-SFT2026 constrain risk/utility novelty. KnowNo/CRC motivate later decision
metrics, no guarantee on reuseddata. Published m=.999 retains57.10% initialstat
at560updates versus2^20paper recipe;018 explicitly adapts timescale/views.
Run published_selector_replay/20261017T072612284949Z: zero-fit snapshotcomponents
on1225hidden-training pairs/385uniqueclips. CorrectUPmass0(originaldual95) to99
classsnapshot, but58wrongNONE→movement. Not fullalgorithmtraining or rankings.
Allweights froze before neworacle analysis, but prioroutcomes known/retrospective.

## Earlier evidence and limits

016 input_conditioning/20261017T063852365459Z:60fits raw/RMS×clock/noclock.
Retrospective96 accuracy86.25/87.08/86.25/85.42%; F1.8580/.8636/.8594/.8488.
RMSclock worsensNONE→movement28.33→43.33%, teacherF1declines; do notpromote.
All15rawparentstates match014,60models/17800vectors/15scales audited, no leakage.
Earlier coordinate/motion work did not isolate physicalcamera viewpoint changes.

015 evidence_profile_diagnostic/20261016T142255449351Z:1225trainingseed–clip
pairs,0fits. UP61correctagreements all fail.95; NONEcorrect25/105. Profile37
accepted targets allalreadycorrectinsupervised160; only a redundancyproxy.
Fixed25% quotas confidence7/164errors,profile5,embedding1,random17; at50%
confidence15/320 beatsprofile23/embedding17. No uniformresponse-profile benefit.

013 evidence_profile/20261016T030844123147Z:80labels/325hidden,65fits.
Retrospective96 supervised74.17%,confidence74.79%,profile74.38%,full40593.33%.
014 evidence_profile_budget160/20261016T133936682213Z:160/245,65fits.
96accuracy supervised86.25%,nativeconfidence87.71%,profile87.29%,embedding88.54%,
full40593.33%. Profile-minus-confidenceF1-.001118 (1positive/2negative/2ties).
Profile0wrong/37pairs,confidence5/89,embedding0/39; noUP/NONEtargets atoldgate.
All8labels/UPstylespresent;48devselects/96retrospective; no405warmstart.

012 motion gains did not replicate uniformly onlater96 clips. Oneuser/twosittings
oneevening is not unseen-user/robot evidence. Fullpaper needs strong external
baselines/benchmarks/broaderreplication. Do not sweep to rescue a lucky seed or
claim that supervised→semi/self-supervised staging is universally best.
