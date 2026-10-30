"""Run synthetic suites with missing legacy mocked-camera fixtures supplied.

No source changes, recording access or downloaded detector is required. The
placeholder exists only in a temporary directory for three fully mocked tests.
It is never installed in models/ or usable for actual detection.
"""

import argparse
import os
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
MOCKED_ASSET_TESTS={
    'research.experiments.test_commands_v2.EightCommandTests.test_camera_adapter_records_up_without_changing_legacy_labels',
    'gesture_commands.test_pipeline.AppWorkflowTests.test_record_freeze_label_save_exit',
    'gesture_commands.test_pipeline.AppWorkflowTests.test_wrong_prediction_freeze_correct_exact_clip',
}


@contextmanager
def mocked_asset_fixture():
    import cv2
    import main
    import mediapipe as mp
    with tempfile.TemporaryDirectory(prefix='mocked-detector-') as temporary:
        asset=Path(temporary)/'mock_detector.task'
        asset.write_bytes(b'Test-only placeholder; not a detector model.\n')
        # The legacy tests replace these guards with their controlled mocks.
        # If they stop mocking a device, fail before any real device access.
        with patch.object(main,'MODEL_PATH',asset), \
             patch.object(cv2,'VideoCapture',side_effect=AssertionError('Real camera access is forbidden in software checks.')), \
             patch.object(mp.tasks.vision.HandLandmarker,'create_from_options',
                          side_effect=AssertionError('Real detector loading is forbidden in mocked-camera checks.')):
            yield asset


def flatten(suite):
    for case in suite:
        if isinstance(case,unittest.TestSuite):yield from flatten(case)
        else:yield case


class FixtureSuite(unittest.TestSuite):
    def run(self,result,debug=False):
        with mocked_asset_fixture():
            return super().run(result,debug)


def build_suite(name='all'):
    loader=unittest.TestLoader();suites=[]
    if name in ('all','packaging'):
        suites.append(loader.discover(str(ROOT/'tools/tests'),top_level_dir=str(ROOT)))
    if name in ('all','research'):
        suites.append(loader.discover(str(ROOT/'research/experiments'),top_level_dir=str(ROOT)))
    if name in ('all','application'):
        suites.append(loader.loadTestsFromName('gesture_commands.test_pipeline'))
    return unittest.TestSuite(FixtureSuite([case]) if case.id() in MOCKED_ASSET_TESTS else case
                              for suite in suites for case in flatten(suite))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite',choices=('all','packaging','research','application'),default='all')
    args=parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='gesture-test-plot-cache-') as cache:
        previous=os.environ.get('MPLCONFIGDIR')
        os.environ.setdefault('MPLCONFIGDIR',cache)
        try:
            if args.suite!='packaging':
                import torch
                torch.set_num_threads(2)
            outcome=unittest.TextTestRunner(verbosity=2).run(build_suite(args.suite))
        finally:
            if previous is None:os.environ.pop('MPLCONFIGDIR',None)
    raise SystemExit(0 if outcome.wasSuccessful() else 1)
