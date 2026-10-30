"""Read-only review of files intended for Git; --staged reads index contents.

No staging, commits, branch changes, uploads, model execution or file deletion.
This is a project packaging check and a basic secret-pattern check, not a proof
that arbitrary content contains no personal or sensitive information.
"""

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

from tools.export_research_records import FORBIDDEN_KEYS, verify_private_sources

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_PARTS = {'.venv','__pycache__','.pytest_cache','.mypy_cache','.ruff_cache','.git','backups'}
FORBIDDEN_PREFIXES = ('gesture_commands/data/','gesture_commands/runs/',
                      'research/experiments/data/','research/experiments/runs/',
                      'research/notes/','research/private/')
FORBIDDEN_SUFFIXES = {'.npz','.npy','.pt','.pth','.ckpt','.task','.pem','.key','.pyc'}
LOCAL_REDIRECTS = {'research/experiments/gesture_failure_review.html',
                   'research/experiments/prospective_motion_review.html'}
SECRET_PATTERNS = {
    'GitHub token': r'\bgh[pousr]_[A-Za-z0-9]{30,}\b',
    'private key': r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',
    'AWS access key': r'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b',
    'OpenAI-style key': r'\bsk-(?:proj-)?[A-Za-z0-9_-]{30,}\b',
}


def git(root,*args):
    return subprocess.check_output(['git','-C',str(root),*args])


def candidate_paths(root=ROOT,staged=False):
    args=('diff','--cached','--name-only','--diff-filter=ACMR','-z') if staged else (
          'ls-files','--cached','--others','--exclude-standard','-z')
    return sorted({name.decode() for name in git(root,*args).split(b'\0') if name})


def prohibited_path(name):
    path=Path(name)
    return (bool(FORBIDDEN_PARTS.intersection(path.parts)) or name.startswith(FORBIDDEN_PREFIXES)
            or path.suffix.lower() in FORBIDDEN_SUFFIXES or name in LOCAL_REDIRECTS
            or (path.name.startswith('.env') and path.name!='.env.example')
            or path.name=='.DS_Store')


def forbidden_payload(value):
    if isinstance(value,dict):
        return bool(FORBIDDEN_KEYS.intersection(str(k).lower() for k in value)) or any(
            forbidden_payload(item) for item in value.values())
    if isinstance(value,list):
        return any(forbidden_payload(item) for item in value)
    return False


def check_content(name,data):
    problems=[]
    if prohibited_path(name):problems.append('excluded personal/runtime/artifact path')
    if len(data)>10*1024*1024:problems.append('over the project 10 MiB per-file review threshold')
    if Path(name).suffix.lower() in {'.png','.pdf','.jpg','.jpeg','.gif'}:
        return problems
    try:text=data.decode('utf-8')
    except UnicodeDecodeError:
        return problems+['unexpected binary file']
    for label,pattern in SECRET_PATTERNS.items():
        if re.search(pattern,text):problems.append('possible '+label+' (value withheld)')
    if re.search(r'/(?:Users|home)/[\w.-]+/',text):
        problems.append('absolute personal home path')
    if name.startswith('research/results/') and name.endswith('.json'):
        try:
            if forbidden_payload(json.loads(text)):problems.append('possible landmark/embedding payload')
        except ValueError:problems.append('invalid JSON')
    return problems


def check_exports(root):
    errors=[]
    index=root/'research/results/index.json'
    if not index.exists():return ['Missing research/results/index.json; export records first.']
    for row in json.loads(index.read_text())['studies']:
        base=root/'research/results'/row['experiment']
        provenance=base/'provenance.json'
        if not provenance.exists():
            errors.append(f'{row["experiment"]}: missing provenance');continue
        if hashlib.sha256(provenance.read_bytes()).hexdigest()!=row['provenance_sha256']:
            errors.append(f'{row["experiment"]}: provenance differs from index')
        for name,expected in json.loads(provenance.read_text())['export_sha256'].items():
            path=base/name
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
                errors.append(f'{row["experiment"]}/{name}: missing or changed export')
    return errors


def check(root=ROOT,staged=False):
    root=Path(root).resolve()
    names=candidate_paths(root,staged)
    issues=[];total=0
    for name in names:
        path=root/name
        if staged:
            mode=git(root,'ls-files','--stage','--',name).decode().split()[0]
            if mode=='120000':
                issues.append({'file':name,'issues':['symlink requires explicit review']});continue
            data=git(root,'show',':'+name)
        else:
            if path.is_symlink():
                issues.append({'file':name,'issues':['symlink requires explicit review']});continue
            if not path.is_file():
                issues.append({'file':name,'issues':['tracked file missing from working tree']});continue
            data=path.read_bytes()
        total+=len(data)
        found=check_content(name,data)
        if found:issues.append({'file':name,'issues':found})
    # Working-tree export validation is independent of index inspection; staged
    # bytes are checked above and must also agree with the working-tree export.
    export_errors=check_exports(root)
    if staged:
        for name in names:
            if name.startswith('research/results/') and (root/name).is_file():
                if git(root,'show',':'+name)!=(root/name).read_bytes():
                    export_errors.append(name+': staged and validated export differ')
    return {'mode':'staged index' if staged else 'tracked + nonignored working files',
            'files_checked':len(names),'total_MiB':round(total/2**20,3),
            'issues':issues,'export_errors':export_errors,
            'passed':bool(names) and not issues and not export_errors,
            'scope':'Basic packaging/content checks; review the staged diff before committing. No files or Git state changed.'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--staged',action='store_true',help='Inspect the actual staged bytes, not only working files.')
    parser.add_argument('--with-private-sources',action='store_true',help='Also compare exports against the original local metadata archives.')
    args=parser.parse_args()
    result=check(staged=args.staged)
    if args.with_private_sources:
        try:result['private_source_verification']=verify_private_sources()
        except (OSError,ValueError) as error:
            result['passed']=False
            result['private_source_verification']={'error':str(error)}
    print(json.dumps(result,indent=2))
    raise SystemExit(0 if result['passed'] else 1)
