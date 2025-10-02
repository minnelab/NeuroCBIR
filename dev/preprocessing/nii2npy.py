import os
import numpy as np
import nibabel as nib
from tqdm import tqdm
from preprocessing.load_dataset import list_files_with_extension, BrainMRIDataset, get_label


def convert_nii_to_npy(image_paths, image_ids, output_dir, pe=None):
    os.makedirs(output_dir, exist_ok=True)
    
    for path, img_id in tqdm(zip(image_paths, image_ids), total=len(image_paths), desc="Converting to .npy"):
        try:
            # Load NIfTI image
            image = np.array(nib.load(path).get_fdata(), dtype=np.float32)

            # Add positional encoding if needed
            if pe is not None:
                image = np.concatenate([image, pe], axis=0)

            # Save as .npy
            np.save(os.path.join(output_dir, f"{img_id}.npy"), image)
        except Exception as e:
            print(f"Failed to process {path}: {e}")


if __name__ == '__main__':


    # Loading MRI  paths
    dataset_path = "/home/maia-user/Dataset/OASIS3/"
    file_paths, file_names = list_files_with_extension(dataset_path, extension="align_norm+cropped.nii.gz")

    raw_image_paths = np.array([os.path.join(dataset_path, file_path, file_name) for file_path, file_name in zip(file_paths, file_names)])
    raw_image_ids = np.array([file_path.split('/')[1] for file_path in file_paths])  # e.g., "OAS30001"

    # Optional PE
    # pos_en = positionalencoding2d(4, 128, 128).detach().cpu().numpy()
    pos_en = None

    convert_nii_to_npy(raw_image_paths, raw_image_ids, output_dir="/home/maia-user/Dataset/OASIS3_NPY/", pe=pos_en)
