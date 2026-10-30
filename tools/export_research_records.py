"""Export explicitly allowed scientific records; never copy tensors or viewers.

Uses the standard library. Originals are read-only. Exported numerical values
are preserved; local path strings are made portable. No Git mutation or upload.
"""

import argparse
import csv
import hashlib
import io
import json
import platform
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = 'research/experiments/runs/'
STUDIES = (
    ('001_augmentation', RUNS+'20261007T050900637997Z', '001_augmentation.md'),
    ('002_label_budget', RUNS+'20261008T005037258990Z', '002_label_budget.md'),
    ('002_new_session', RUNS+'20261008T005037258990Z/session_evaluations/20261008T013739665513Z', '002_new_session_01.md'),
    ('003_commands_v2', RUNS+'commands_v2/20261008T025353704748Z', '003_commands_v2.md'),
    ('003_second_session', RUNS+'commands_v2/20261008T025353704748Z/session_evaluations/20261008T052213777935Z', '003_commands_v2_session_02.md'),
    ('004_boundary_data', RUNS+'boundary_data/20261008T072040903427Z', '004_boundary_data.md'),
    ('005_boundary_test', 'research/experiments/data/boundary_test/evaluation', '005_boundary_test.md'),
    ('006_commands_v3', RUNS+'commands_v3/20261012T205727638067Z', '006_commands_v3.md'),
    ('007_motion_probe', RUNS+'motion_probes/20261012T221029684780Z', '007_motion_sensitivity.md'),
    ('008_temporal_architectures', RUNS+'temporal_comparison/20261014T011524632211Z', '008_temporal_architectures.md'),
    ('009_failure_review', RUNS+'failure_review/20261014T053532241062Z', '009_failure_review.md'),
    ('010_motion_representation', RUNS+'motion_representation/20261015T051753816210Z', '010_motion_representation.md'),
    ('011_cross_model_motion', RUNS+'cross_model_motion/20261015T164901242592Z', '011_cross_model_motion.md'),
    ('012_later_sessions', 'research/experiments/data/prospective_motion_011/analysis', '012_prospective_motion_sessions.md'),
    ('013_evidence_profile', RUNS+'evidence_profile/20261016T030844123147Z', '013_evidence_profile.md'),
    ('014_budget160', RUNS+'evidence_profile_budget160/20261016T133936682213Z', '014_evidence_profile_budget160.md'),
    ('015_profile_diagnostic', RUNS+'evidence_profile_diagnostic/20261016T142255449351Z', '015_evidence_profile_diagnostic.md'),
    ('016_input_conditioning', RUNS+'input_conditioning/20261017T063852365459Z', '016_input_conditioning.md'),
    ('017_selector_replay', RUNS+'published_selector_replay/20261017T072612284949Z', '017_literature_grounded_decision.md'),
    ('018_online_softmatch', RUNS+'online_softmatch/20261020T205543410735Z', '018_online_softmatch.md'),
)
JSON_NAMES = (
    'config.json', 'summary.json', 'trials.json', 'dataset_summary.json',
    'coverage.json', 'sampling.json', 'selections.json', 'environment.json',
    'costs.json', 'status.json', 'verification.json', 'completion_verification.json',
    'preflight.json', 'contract.json', 'evaluation_plan.json',
)
CSV_NAMES = (
    'model_results.csv', 'per_class_results.csv', 'unique_failure_counts.csv',
    'predicted_class_stages.csv', 'ranking_classes.csv', 'ranking_pooled.csv',
    'ranking_results.csv', 'student_proxies.csv', 'true_class_stages.csv',
    'costs.csv', 'gate_saturation.csv', 'development_class_metrics.csv',
    'development_per_seed.csv', 'pseudo_label_diagnostics.csv', 'per_class.csv',
    'per_seed.csv', 'up_style_posthoc.csv',
)
FIGURE_NAMES = ('figure.png', 'figure.pdf', 'label_budget_curve.png',
                'label_budget_curve.pdf', 'development_comparison.png',
                'development_comparison.pdf')
FORBIDDEN_KEYS = {'xyz', 'landmarks', 'coordinates', 'raw_frames', 'anchor_embedding',
                  'subset_query_embedding', 'subset_query_logits', 'review_data'}
ARCHIVE_ROOTS = ('research/experiments/runs', 'research/experiments/data',
                 'gesture_commands/data', 'gesture_commands/runs', 'models', 'research/notes')
CACHE_PARTS = {'__pycache__', 'plot_cache', '.DS_Store'}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def portable_string(text, root):
    prefixes={str(Path(root).absolute()),str(Path(root).resolve())}
    for prefix in sorted(prefixes,key=len,reverse=True):
        text = text.replace(prefix+'/', '')
        text = text.replace(prefix, '<PROJECT>')
    # Foreign historical home/temp paths are identifiers, never executable input.
    text = re.sub(r'/Users/[^/\s"\x27]+', '<USER_HOME>', text)
    text = re.sub(r'/home/[^/\s"\x27]+', '<USER_HOME>', text)
    text = re.sub(r'/private/var/folders/[^\s"\x27]+', '<LOCAL_TEMP>', text)
    return text


def sanitize(value, root):
    if isinstance(value, dict):
        if FORBIDDEN_KEYS.intersection(str(k).lower() for k in value):
            raise ValueError('Refusing a possible landmark/embedding payload.')
        result = {}
        for key, item in value.items():
            cleaned = portable_string(str(key), root)
            if cleaned in result:
                raise ValueError('Portable path conversion would collide dictionary keys.')
            result[cleaned] = sanitize(item, root)
        return result
    if isinstance(value, list):
        return [sanitize(item, root) for item in value]
    if isinstance(value, str):
        return portable_string(value, root)
    return value


def json_bytes(value):
    return (json.dumps(value, indent=2, allow_nan=False)+'\n').encode()


def safe_write(path, data):
    path = Path(path)
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f'Existing export differs; preserve it and review explicitly: {path}')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def export_one(root, output, descriptor):
    identifier, relative, finding = descriptor
    source = root/relative
    if not source.is_dir():
        raise FileNotFoundError(f'Private source archive is unavailable: {relative}')
    target = output/identifier
    sources, exports = {}, {}

    def add(path, destination, content=None):
        source_name = path.relative_to(source).as_posix()
        sources[source_name] = sha(path)
        data = content if content is not None else path.read_bytes()
        exports[destination] = data

    for base in (source, source/'analysis'):
        if not base.is_dir():
            continue
        prefix = '' if base == source else 'analysis/'
        for name in JSON_NAMES:
            path = base/name
            if path.exists():
                value = json.loads(path.read_text())
                add(path, prefix+name, json_bytes(sanitize(value, root)))
        for name in CSV_NAMES:
            path = base/name
            if path.exists():
                with path.open(newline='') as stream:
                    rows = list(csv.reader(stream))
                result = io.StringIO(newline='')
                csv.writer(result, lineterminator='\n').writerows(
                    [[portable_string(cell, root) for cell in row] for row in rows])
                add(path, prefix+name, result.getvalue().encode())
        for name in FIGURE_NAMES:
            path = base/name
            if path.exists():
                add(path, prefix+name)
    # Validation-loss curves only; no per-clip features, predictions or viewers.
    for path in sorted(source.rglob('history.json')):
        rel = path.relative_to(source)
        if len(rel.parts)>4 or {'source','session_evaluations','analysis'}.intersection(rel.parts):
            continue
        add(path, 'histories/'+rel.as_posix(), json_bytes(sanitize(json.loads(path.read_text()), root)))
    if identifier=='009_failure_review':
        path=source/'analysis.json'
        add(path,'review_summary.json',json_bytes(sanitize(json.loads(path.read_text())['summary'],root)))
    # Pin full original manifests/freezes by hash without exporting their payloads.
    references = {name:sha(source/name) for name in
                  ('freeze.json','freeze.sha256','manifest.json','registry.json','fit_freeze.json')
                  if (source/name).is_file()}
    info = {'schema_version':1, 'experiment':identifier,'original_archive':relative,
            'finding':'research/findings/'+finding,
            'scope':'Historical development/retrospective records; see finding for endpoint roles and limits.',
            'transformations':['Local paths made portable; numerical values retained.',
                               'Allowed metadata/tables/charts/loss histories only; no recordings, weights or per-clip viewer data.'],
            'source_sha256':sources, 'private_reference_sha256':references,
            'export_sha256':{name:hashlib.sha256(data).hexdigest() for name,data in exports.items()}}
    exports['provenance.json']=json_bytes(info)
    files='\n'.join(f'- [{name}]({name})' for name in sorted(exports))
    note=(f'# {identifier}\n\n[Interpretation and limitations](../../findings/{finding}).\n\n'
          f'Original local archive: `{relative}`.\n\n'
          'Paths in exported metadata are portable identifiers. These copies do not replace the original freezes '
          'and must not be fed to historical audit loaders as original manifests.\n\n'
          'All available seed outcomes in the selected records are retained, including unfavorable results. '
          'Private input archives are required for a full historical rerun. No external artifact upload is configured.\n\n'
          +files+'\n')
    exports['README.md']=note.encode()
    # Validate every file before writing any file for this experiment.
    for name,data in exports.items():
        path=target/name
        if path.exists() and path.read_bytes()!=data:
            raise ValueError(f'Existing export differs; preserve it and review explicitly: {path}')
    for name,data in exports.items():
        safe_write(target/name,data)
    return {'experiment':identifier,'archive':relative,'finding':'research/findings/'+finding,
            'exported_files':len(exports),'exported_bytes':sum(map(len,exports.values())),
            'provenance_sha256':sha(target/'provenance.json')}


def inventory(root):
    all_rows,groups=[],[]
    for name in ARCHIVE_ROOTS:
        base=root/name
        rows=[]
        for path in sorted(base.rglob('*')) if base.exists() else []:
            if not path.is_file() or path.is_symlink() or CACHE_PARTS.intersection(path.parts):
                continue
            row={'path':path.relative_to(root).as_posix(),'bytes':path.stat().st_size,'sha256':sha(path)}
            rows.append(row)
        all_rows.extend(rows)
        groups.append({'path':name,'files':len(rows),'bytes':sum(r['bytes'] for r in rows),
                       'content_index_sha256':hashlib.sha256(json_bytes(rows)).hexdigest()})
    return {'schema_version':1,'files':all_rows},groups


def verify_private_sources(root=ROOT):
    """Compare exports against original metadata, without loading any tensors."""
    root=Path(root).resolve();verified=0
    index=json.loads((root/'research/results/index.json').read_text())
    for row in index['studies']:
        source=root/row['archive'];target=root/'research/results'/row['experiment']
        provenance=json.loads((target/'provenance.json').read_text())
        for name,expected in {**provenance['source_sha256'],**provenance['private_reference_sha256']}.items():
            if sha(source/name)!=expected:raise ValueError('Original archive changed: '+row['experiment']+'/'+name)
        for name in provenance['export_sha256']:
            original=name.removeprefix('histories/')
            path=source/original
            if name=='review_summary.json':
                expected=json.loads((source/'analysis.json').read_text())['summary']
                if json.loads((target/name).read_text())!=sanitize(expected,root):
                    raise ValueError('Review summary changed.')
            elif name.endswith('.json'):
                if json.loads((target/name).read_text())!=sanitize(json.loads(path.read_text()),root):
                    raise ValueError('Exported JSON values changed: '+name)
            elif name.endswith('.csv'):
                with path.open(newline='') as f:before=list(csv.reader(f))
                with (target/name).open(newline='') as f:after=list(csv.reader(f))
                if after!=[[portable_string(v,root) for v in values] for values in before]:
                    raise ValueError('Exported CSV cells changed: '+name)
            elif sha(path)!=sha(target/name):
                raise ValueError('Exported figure changed: '+name)
            verified+=1
    return {'studies':len(index['studies']),'exported_payloads_verified_against_originals':verified,
            'all_recorded_source_hashes_match':True,'numerical_records_preserved':True}


def export_all(root=ROOT):
    root=Path(root).resolve()
    output=root/'research/results'
    records=[export_one(root,output,descriptor) for descriptor in STUDIES]
    safe_write(output/'index.json',json_bytes({'schema_version':1,'studies':records}))
    private,groups=inventory(root)
    # Local backup inventory is deliberately excluded by .gitignore.
    private_path=root/'research/private/artifact_inventory.json'
    private_path.parent.mkdir(parents=True,exist_ok=True)
    private_path.write_bytes(json_bytes(private))
    safe_write(output/'archives.json',json_bytes({'schema_version':1,'artifact_roots':groups,
               'backup_status':'No off-device upload/backup was performed by this preparation.',
               'storage_uri':None,'private_index':'research/private/artifact_inventory.json'}))
    # Freeze the direct environment observed during this packaging action, not
    # an invented universal dependency lock for all historical studies.
    import importlib.metadata
    environment={'scope':'Environment observed during repository preparation; historical environments are in each study export.',
                 'python':platform.python_version(),'platform':platform.platform(),
                 'packages':{p:importlib.metadata.version(p) for p in
                             ('torch','numpy','matplotlib','mediapipe','opencv-contrib-python')}}
    safe_write(output/'preparation_environment.json',json_bytes(environment))
    return {'studies':len(records),'exported_files':sum(r['exported_files'] for r in records),
            'exported_bytes':sum(r['exported_bytes'] for r in records),
            'private_inventory_files':len(private['files'])}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    print(json.dumps(export_all(),indent=2))
