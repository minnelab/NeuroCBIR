import os
import numpy as np
import nibabel as nib
from tqdm import tqdm
from load_dataset import list_files_with_extension

def brain_crop(brain_data, desired_shape=[160, 176, 208]):
    # Define target shape (160,160,192)
    target_shape = np.array(desired_shape)

    # Compute new crop boundaries
    # start = center - (target_shape // 2)
    start = [48, 36, 8]
    end = start + target_shape

    # Ensure boundaries don't go out of bounds
    end = np.minimum(end, brain_data.shape)

    # Apply cropping
    cropped_data = brain_data[start[0]:end[0], start[1]:end[1], start[2]:end[2]]    
    return cropped_data

def convert_nii_to_npy(image_paths, seg_paths, image_ids, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for img_path, seg_path, img_id in tqdm(zip(image_paths, seg_paths, image_ids), total=len(image_paths), desc="Converting to .npy"):
        try:
            # Load and preprocess image
            image = np.array(nib.load(img_path).get_fdata(), dtype=np.float32)
            image = np.flip(image.transpose((1, 2, 0)), 1)
            image = (image * 255).clip(0, 255).astype(np.uint8)

            # Load and preprocess segmentation
            seg = np.array(nib.load(seg_path).get_fdata(), dtype=np.float32)
            seg = brain_crop(seg, desired_shape=[160, 176, 208])
            seg = np.flip(seg.transpose((1, 2, 0)), 1).astype(np.uint8)

            # Save both in a dictionary
            image_seg = {
                "image": image,
                "seg": seg
            }

            # Optional: print for debugging
            # print(f"[{img_id}] image: {image.shape}, seg: {seg.shape}, dtype: {seg.dtype}")

            # Save to .npy file
            np.save(os.path.join(output_dir, f"{img_id}.npy"), image_seg)

        except Exception as e:
            print(f"❌ Failed to process {img_path}: {e}")


if __name__ == '__main__':


    # Loading MRI  paths

    dataset_path = "/home/maia-user/Dataset/OASIS3/"
    file_paths, file_names = list_files_with_extension(dataset_path, extension="align_norm+cropped.nii.gz")

    raw_image_paths = np.array([os.path.join(dataset_path, file_path, file_name) for file_path, file_name in zip(file_paths, file_names)])
    raw_image_ids = np.array([file_path.split('/')[1] for file_path in file_paths])  # e.g., "OAS30001"

    file_paths, file_names = list_files_with_extension(dataset_path, extension="align_aseg.nii.gz")
    raw_seg_paths = np.array([os.path.join(dataset_path, file_path, file_name) for file_path, file_name in zip(file_paths, file_names)])

    convert_nii_to_npy(raw_image_paths, raw_seg_paths, raw_image_ids, output_dir="/home/maia-user/Dataset/OASIS3_NPY_UINT/")
