import unittest

from tools.run_software_checks import mocked_asset_fixture


class FixtureTests(unittest.TestCase):
    def test_placeholder_is_temporary_and_real_devices_are_blocked(self):
        import cv2
        import main
        import mediapipe as mp
        original=main.MODEL_PATH
        with mocked_asset_fixture() as asset:
            self.assertTrue(asset.is_file())
            self.assertEqual(main.MODEL_PATH,asset)
            self.assertNotEqual(asset,original)
            with self.assertRaisesRegex(AssertionError,'camera'):
                cv2.VideoCapture(0)
            with self.assertRaisesRegex(AssertionError,'detector'):
                mp.tasks.vision.HandLandmarker.create_from_options(None)
        self.assertEqual(main.MODEL_PATH,original)
        self.assertFalse(asset.exists())


if __name__=='__main__':unittest.main()
