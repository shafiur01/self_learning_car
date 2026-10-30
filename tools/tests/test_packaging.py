import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from tools import check_repository as check
from tools import export_research_records as export


class PackagingTests(unittest.TestCase):
    def test_path_conversion_preserves_numeric_values(self):
        root=Path(tempfile.gettempdir())/'synthetic-project'
        data={'path':str(root/'research/experiments/runs/example/model.pt'),
              'metrics':[.1,None,{'accuracy':.875,'count':7}]}
        result=export.sanitize(data,root)
        self.assertEqual(result['path'],'research/experiments/runs/example/model.pt')
        self.assertEqual(result['metrics'],data['metrics'])

    def test_payload_and_key_collision_are_rejected(self):
        with self.assertRaisesRegex(ValueError,'payload'):
            export.sanitize({'xyz':[[1,2,3]]},Path('/project'))
        with self.assertRaisesRegex(ValueError,'collide'):
            export.sanitize({'/project/file':1,'file':2},Path('/project'))

    def test_allowlist_exports_both_outcomes_without_biometrics(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);source=root/'private-run';source.mkdir()
            (source/'summary.json').write_text(json.dumps({'gain':-.01,'seeds':[.8,.9]}))
            (source/'config.json').write_text(json.dumps({'path':str(source),'steps':560}))
            (source/'review_data.json').write_text(json.dumps({'xyz':[[1,2,3]]}))
            (source/'model.pt').write_bytes(b'private weight placeholder')
            before={p.name:export.sha(p) for p in source.iterdir()}
            descriptor=('fixture','private-run','fixture.md')
            report=export.export_one(root,root/'results',descriptor)
            output=root/'results/fixture'
            self.assertEqual(json.loads((output/'summary.json').read_text())['gain'],-.01)
            self.assertFalse((output/'model.pt').exists())
            self.assertFalse((output/'review_data.json').exists())
            self.assertEqual(before,{p.name:export.sha(p) for p in source.iterdir()})
            self.assertEqual(report,export.export_one(root,root/'results',descriptor))
            (source/'summary.json').write_text('{"gain": 0.5}')
            with self.assertRaisesRegex(ValueError,'differs'):
                export.export_one(root,root/'results',descriptor)

    def test_unsafe_staged_paths_remain_forbidden_even_if_gitignored(self):
        for name in ('research/experiments/data/clip.npz','elsewhere/model.pt',
                     'research/private/report.json','.env','research/notes/personal.md'):
            self.assertTrue(check.prohibited_path(name))
        self.assertFalse(check.prohibited_path('research/results/018/summary.json'))
        self.assertFalse(check.prohibited_path('.env.example'))

    def test_secret_patterns_do_not_expose_values(self):
        value='ghp_'+'a'*36
        problems=check.check_content('example.py',value.encode())
        self.assertTrue(any('token' in p for p in problems))
        self.assertTrue(all(value not in p for p in problems))

    def test_home_path_scanner_distinguishes_redaction_pattern(self):
        pattern=b"re.sub(r'/Users/[^/]+', '<USER_HOME>', text)"
        self.assertEqual(check.check_content('redaction.py',pattern),[])
        actual=b'/'+b'Users/'+b'test-user/file'
        self.assertIn('absolute personal home path',check.check_content('record.json',actual))

    def test_index_scan_checks_staged_bytes_after_working_file_changes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            subprocess.run(['git','init','-q',str(root)],check=True)
            records=root/'research/results';records.mkdir(parents=True)
            (records/'index.json').write_text('{"studies":[]}')
            secret='ghp_'+'b'*36
            (root/'example.txt').write_text(secret)
            subprocess.run(['git','-C',str(root),'add','example.txt','research/results/index.json'],check=True)
            (root/'example.txt').write_text('clean working copy')
            report=check.check(root,staged=True)
            self.assertFalse(report['passed'])
            self.assertTrue(any(x['file']=='example.txt' for x in report['issues']))
            self.assertEqual(check.git(root,'show',':example.txt').decode(),secret)

    def test_export_integrity_rejects_changed_summary(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);base=root/'research/results/fixture';base.mkdir(parents=True)
            (base/'summary.json').write_text('{"gain":-0.01}\n')
            provenance={'export_sha256':{'summary.json':export.sha(base/'summary.json')}}
            (base/'provenance.json').write_bytes(export.json_bytes(provenance))
            (root/'research/results/index.json').write_bytes(export.json_bytes({'studies':[
                {'experiment':'fixture','provenance_sha256':export.sha(base/'provenance.json')}]}))
            self.assertEqual(check.check_exports(root),[])
            (base/'summary.json').write_text('{"gain":1}\n')
            self.assertTrue(check.check_exports(root))


if __name__=='__main__':unittest.main()
