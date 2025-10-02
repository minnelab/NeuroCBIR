import unittest
import h5py
import numpy as np
from pathlib import Path
import random

class TestRandomHDF5File(unittest.TestCase):

    def setUp(self):
        # Set this to the directory containing your HDF5 files
        self.hdf5_dir = Path("/mimer/NOBACKUP/groups/biomedicalimaging-kth/felixnie/batched_OASIS3")
        self.hdf5_files = list(self.hdf5_dir.glob("*.h5"))
        self.assertTrue(self.hdf5_files, "❌ No HDF5 files found in directory.")
        self.test_file = random.choice(self.hdf5_files)

    def test_hdf5_structure_and_integrity(self):
        print(f"\n🔍 Testing file: {self.test_file.name}")
        with h5py.File(self.test_file, "r") as f:
            # Check expected datasets exist
            for name in ["images", "segmentations", "ids"]:
                self.assertIn(name, f, f"❌ Missing dataset: {name}")

            images = f["images"][:]
            segmentations = f["segmentations"][:]
            ids = f["ids"][:]

            # Check shapes and alignment
            self.assertEqual(images.shape[0], segmentations.shape[0], "❌ Mismatch in image/segmentation sample count.")
            self.assertEqual(images.shape[0], len(ids), "❌ Mismatch in image/id count.")

            # Sanity checks on values
            self.assertFalse(np.isnan(images).any(), "❌ Found NaNs in images.")
            self.assertFalse(np.isnan(segmentations).any(), "❌ Found NaNs in segmentations.")
            self.assertTrue(images.ndim == 4, "❌ Expected 4D images (batch, channels, height, width).")
            self.assertTrue(segmentations.ndim == 4, "❌ Expected 4D segmentations.")

            # Print a sample for debug
            print(f"✅ Loaded file: {self.test_file.name}")
            print(f"  images shape: {images.shape}")
            print(f"  segmentations shape: {segmentations.shape}")
            print(f"  ids[0]: {ids[0]}")

if __name__ == "__main__":
    unittest.main()
