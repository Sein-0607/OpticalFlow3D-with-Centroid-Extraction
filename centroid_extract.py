import nibabel as nib
import numpy as np
import cv2
from scipy.ndimage import affine_transform

# Step 1: Load fixed and moving volumes
fixed_nifti_path = "PATH/TO/FIXED/VOLUME"
labeled_nifti_path = "PATH/TO/FIXED/VOLUME_LABEL"

fixed_image = nib.load(fixed_nifti_path)
fixed_image_data = fixed_image.get_fdata()
labeled_image = nib.load(labeled_nifti_path)
labeled_image_data = labeled_image.get_fdata()

moving_nifti_path = "PATH/TO/MOVING/VOLUME"
moving_labeled_nifti_path = "PATH/TO/MOVING/VOLUME_LABEL"

moving_image = nib.load(moving_nifti_path)
moving_image_data = moving_image.get_fdata()
moving_labeled_image = nib.load(moving_labeled_nifti_path)
moving_labeled_image_data = moving_labeled_image.get_fdata()

# Step 2: Extract centroids for each label
labels_of_interest = [label index_0, label index_1...]

fixed_centroids = {}
moving_centroids = {}

for label in labels_of_interest:
    fixed_label_coords = np.where(labeled_image_data == label)
    fixed_centroid = np.mean(np.array(fixed_label_coords), axis=1)

    moving_label_coords = np.where(moving_labeled_image_data == label)
    moving_centroid = np.mean(np.array(moving_label_coords), axis=1)

    # Check for outliers in either fixed or moving centroids
    if all(0 <= coord <= 256 for coord in fixed_centroid) and all(0 <= coord <= 256 for coord in moving_centroid):
        fixed_centroids[label] = fixed_centroid
        moving_centroids[label] = moving_centroid

