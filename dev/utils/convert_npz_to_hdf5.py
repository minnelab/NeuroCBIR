# import numpy as np
# import h5py
# import os
# from pathlib import Path
# import argparse

# def convert_npz_to_hdf5(npz_path: Path, delete_after=False):
#     hdf5_path = npz_path.with_suffix('.h5')
#     print(f"Converting {npz_path} → {hdf5_path}")

#     # try:
#     with np.load(npz_path, allow_pickle=True) as data:
#         images = data["images"]
#         segmentations = data["segmentations"]
#         ids = data["ids"]

#     with h5py.File(hdf5_path, "w") as f:
#         # Create datasets with gzip compression
#         f.create_dataset("images", data=images, compression="gzip", compression_opts=4, chunks=True)
#         f.create_dataset("segmentations", data=segmentations, compression="gzip", compression_opts=4, chunks=True)

#         # Ensure all IDs are native Python strings (not NumPy Unicode)
#         ids_utf8 = np.array([str(s) for s in ids], dtype=object)

#         # Define HDF5-compatible variable-length UTF-8 dtype
#         utf8_dtype = h5py.string_dtype(encoding='utf-8')

#         # Write to HDF5
#         f.create_dataset("ids", data=ids_utf8, dtype=utf8_dtype)


#     # except Exception as e:
#     #     print(f"❌ Failed to convert {npz_path.name}: {e}")
#     #     return

#     if delete_after:
#         try:
#             os.remove(npz_path)
#             print(f"✅ Deleted original file: {npz_path.name}")
#         except Exception as e:
#             print(f"⚠️ Could not delete {npz_path.name}: {e}")

# def batch_convert_npz_to_hdf5(folder: str, delete_after=False):
#     folder = Path(folder)
#     npz_files = sorted(folder.glob("*.npz"))
#     if not npz_files:
#         print("No .npz files found in", folder)
#         return

#     for npz_file in npz_files:
#         convert_npz_to_hdf5(npz_file, delete_after)

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Convert .npz files to .h5 (HDF5) files with compression")
#     parser.add_argument("folder", type=str, help="Folder containing .npz files")
#     parser.add_argument("--delete", action="store_true", help="Delete .npz files after conversion")
#     args = parser.parse_args()

#     batch_convert_npz_to_hdf5(args.folder, delete_after=args.delete)
