"""Small public-data smoke tests for the command-line verification script."""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent


class VerificationCliTest(unittest.TestCase):
    def test_reads_selected_data_directory_and_writes_selected_output(self):
        with tempfile.TemporaryDirectory() as directory:
            location = Path(directory)
            for seed in (17, 29, 43):
                pd.DataFrame({
                    "video": [1, 1, 2, 2],
                    "y_c1": [0, 1, 0, 1],
                    "y_c2": [0, 1, 0, 1],
                    "y_c3": [0, 1, 0, 1],
                    "p_c1": [0.1, 0.9, 0.2, 0.8],
                    "p_c2": [0.1, 0.9, 0.2, 0.8],
                    "p_c3": [0.1, 0.9, 0.2, 0.8],
                }).to_csv(location / f"locked_test_predictions_seed{seed}.csv", index=False)
            output = location / "verified.csv"
            run = subprocess.run(
                [sys.executable, str(ROOT / "verify_bootstrap.py"),
                 "--data-dir", str(location), "--out", str(output)],
                text=True, capture_output=True,
            )
            self.assertEqual(run.returncode, 0, run.stderr)
            result = pd.read_csv(output)
            self.assertEqual(len(result), 9)
            self.assertTrue((result["pooled_auc"] == 1.0).all())
            self.assertTrue((result["within_pair_fraction"] == 0.5).all())


if __name__ == "__main__":
    unittest.main()
