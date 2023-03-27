from typing import Union, Tuple, Dict
import pydicom as dcm
import numpy as np
import nibabel as nib
import json
import os
import cv2 as cv
from preprocess.windowing import Preprocessing, crop_pad_vol
from config import *
import random
import shutil
from annotation_format import COCOFormat


class SetEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, set):
            return list(obj)
        return json.JSONEncoder.default(self, obj)


def save_metadata_json(data: Dict):
    output_file = f'{OUTPUT_PATH}/metadata.json'
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    if os.path.exists(output_file):
        os.remove(output_file)
    with open(output_file, 'w') as outfile:
        json.dump(data, outfile, cls=SetEncoder)


def split_data(metadata: Dict, ratio: float = 0.5,) -> None:
    image_path = os.path.join(OUTPUT_PATH, "images")
    mask_path = os.path.join(OUTPUT_PATH, "masks")

    train_img_path = os.path.join(TRAINING_PATH, "images")
    train_msk_path = os.path.join(TRAINING_PATH, "masks")
    os.makedirs(train_img_path, exist_ok=True)
    os.makedirs(train_msk_path, exist_ok=True)

    test_img_path = os.path.join(TESTING_PATH, "images")
    test_msk_path = os.path.join(TESTING_PATH, "masks")
    os.makedirs(test_img_path, exist_ok=True)
    os.makedirs(test_msk_path, exist_ok=True)

    random.seed(42)
    random.shuffle(PATIENTS)

    train_data = PATIENTS[:int(ratio*len(PATIENTS))]
    test_data = PATIENTS[int(ratio*len(PATIENTS)):]
    train_filenames = [filename for filename in os.listdir(
        image_path) if filename.split("_")[0] in train_data]
    test_filenames = [filename for filename in os.listdir(
        mask_path) if filename.split("_")[0] in test_data]
    for tr_fname in train_filenames:
        shutil.move(os.path.join(image_path, tr_fname),
                    os.path.join(train_img_path, tr_fname))
        shutil.move(os.path.join(mask_path, tr_fname),
                    os.path.join(train_msk_path, tr_fname))
    for ts_fname in test_filenames:
        shutil.move(os.path.join(image_path, ts_fname),
                    os.path.join(test_img_path, ts_fname))
        shutil.move(os.path.join(mask_path, ts_fname),
                    os.path.join(test_msk_path, ts_fname))

    if not os.listdir(image_path) and not os.listdir(mask_path):
        os.rmdir(image_path)
        os.rmdir(mask_path)


class Patient:
    def __init__(
        self,
        patient_id: str,
        ct_dir: str,
        mask_dir: str,
        output_path: str,
        mask_extension: str = ".nii.gz",
        image_crop: Union[tuple, None] = None,
        window: Union[str, None] = "soft_tissue"
    ) -> None:

        self.patient_id = patient_id
        # on the os.path.join , S is included after the homogenization of the dataset
        # TODO -> Change the directory that is hardcoded
        self.ct_dir = ct_dir
        self.mask_dir = mask_dir

        self.ct_path = os.path.join(ct_dir, patient_id)
        self.mask_path = os.path.join(mask_dir, patient_id+mask_extension)
        self.annotation_dict = COCOFormat()
        self.patient_metadata = self.get_patient_metadata()
        self.ct_metadata = self.load_volume_parameters()
        self.patient_volume = self.get_volume()
        self.patient_masks = self.get_mask()
        self.output_path = output_path
        self.dicom_output = os.path.join(self.output_path, "images")
        self.mask_output = os.path.join(self.output_path, "masks")
        self.image_crop = image_crop
        self.preprocessing = Preprocessing(
            window=window,
            metadata=self.ct_metadata
        )

    def get_patient_metadata(self,) -> Union[dcm.FileDataset, None]:
        """
        Each CT scan contains its own DIRFILE that contains 
        information for patient's metadata and the CT volume
        """
        if os.path.exists(os.path.join(self.ct_path, "DIRFILE")):
            metadata = dcm.dcmread(os.path.join(self.ct_path, "DIRFILE"))
        else:
            return None
        return metadata

    def get_filenames(self,) -> Union[list, None]:
        """
        This function is not completely right since we added hardcoded the second element
        TODO -> check a generalized method for getting the slice folder
        """
        try:
            if self.patient_metadata:
                ct_folder = os.path.join(
                    self.ct_path, self.patient_metadata.DirectoryRecordSequence[0].ReferencedFileID[1])
                return [os.path.join(ct_folder, i) for i in os.listdir(ct_folder)
                        if i != "DIRFILE" and i != ".DS_Store"]
        except ValueError as e:
            raise ValueError("Failed to load patient metadata")

    def dicom_extraction(self, only_location: bool = False) -> list:
        dicom_files = []
        for filename in self.get_filenames():
            sl = dcm.dcmread(filename)
            dicom_files.append(sl)
        if only_location:
            return sorted([float(x.SliceLocation) for x in dicom_files])

        return sorted(dicom_files, key=lambda x: float(x.SliceLocation))

    def get_volume(self, ) -> np.ndarray:
        return np.array([sl.pixel_array for sl in self.dicom_extraction()]).transpose(1, 2, 0).astype(np.float32)

    def get_modality(self) -> str:
        return self.patient_metadata.DirectoryRecordSequence[0].Modality

    def get_mask(self) -> np.ndarray:
        # print(f"{self.mask_path}, \n {self.ct_path}")
        mask = nib.load(self.mask_path).get_fdata()
        mask = np.swapaxes(mask, 1, 0).astype(np.uint8)
        mask[mask > 0] = True
        mask[mask == 0] = False
        return mask

    def get_shape(self) -> tuple:
        if self.image_crop:
            return (
                self.image_crop[0],
                self.image_crop[1],
                int(
                    self.patient_metadata.DirectoryRecordSequence[0].NumberOfSeriesRelatedInstances)
            )
        else:
            return (
                self.patient_metadata.DirectoryRecordSequence[0].Rows,
                self.patient_metadata.DirectoryRecordSequence[0].Columns,
                int(
                    self.patient_metadata.DirectoryRecordSequence[0].NumberOfSeriesRelatedInstances)
            )

    def load_volume_parameters(self) -> dict:
        sl = dcm.dcmread(self.get_filenames()[0])
        return {
            "SliceThickness": sl.SliceThickness,
            "PixelSpacing": list(sl.PixelSpacing),
            "ImageOrientation": list(sl.ImageOrientationPatient),
            "RescaleIntercept": sl.RescaleIntercept,
            "RescaleSloce": sl.RescaleSlope,
        }

    def get_segmented_region(self) -> dict:
        mask_size = self.patient_masks.shape
        sl_locations = self.dicom_extraction(only_location=True)
        indexes = []
        locations = []
        # Loop through the slices of the mask image
        for z, loc in zip(range(mask_size[2]), sl_locations):
            slice = self.patient_masks[:, :, z]
            if slice.any():
                if len(np.nonzero(slice)[0]) < 100:
                    # If the pixel size of the segmented region
                    # is less than 100 the dont pass it
                    pass
                indexes.append(z)
                locations.append(loc)
        print(len(indexes), len(locations))
        return {
            "Indexes": indexes,
            "SliceLocation": locations
        }

    def data_ROI_only(self) -> Tuple[np.ndarray, np.ndarray]:
        indexes = self.get_segmented_region()["Indexes"]
        msk = []
        sl = []
        for i in indexes:
            msk.append(self.patient_masks[:, :, i])
            sl.append(self.patient_volume[:, :, i])

        masks, vol = np.array(msk).transpose(
            1, 2, 0), np.array(sl).transpose(1, 2, 0)
        vol = self.preprocessing.windowing.volume_windowing(vol)
        if self.image_crop:
            return crop_pad_vol(vol, masks, self.image_crop)
        else:
            return vol, masks

    def get_centroids(self):
        """
        TODO-> Fix the doc
        TODO-> Change the labeling method. In this specific case we do not have any other label for detection,
            thus, we hardcoded the label 0 for all the cases, where 0 is the region that contains
            the bone marrow. 
        Get the centroid of each contour using the openCV module
        centroid: List of length (N,2), where N is the number of bounding boxes in the image. 
            The array should contain the (cX,cY) coordinates of each bounding box.
        bbox: np.array of shape (N,4), where N is the number of bounding boxes in the image. 
            The array should contain the (x_1,y_1,x_2,y_2) coordinates of each bounding box.
        label: list of shape (N,) that contains the label intex of each bounding box,
            since we have only one class we set all the labels to the same value (e.g "0")
        """
        slices, masks = self.data_ROI_only()
        print(f"Slices shape : {slices.shape}, Masks shape: {masks.shape}")
        centroids = {}
        for sl in range(slices.shape[2]):
            labels = []
            centroids[sl] = {}
            contours, hierarchy = cv.findContours(
                masks[:, :, sl], cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
            box_size = (20, 20)
            centroids[sl]["centroid"] = []
            centroids[sl]["bbox"] = []
            for cnt in contours:
                M = cv.moments(cnt)
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                # centroids[sl]["centroid"].append((cX, cY))
                x1, y1, x2, y2 = self.get_bounding_box(
                    cX, cY, box_size=box_size)
                centroids[sl]["centroid"].append([cX, cY])
                centroids[sl]["bbox"].append([x1, y1, x2, y2])
                labels.append(0)
        return centroids

    def get_bounding_box(self, cX: float, cY: float, box_size: tuple = (40, 40)) -> Tuple[int, int, int, int]:
        """
        Function that returns the `top left` and `bottom right` corner of the bounding box
        """
        b_width, b_height = box_size
        x1 = cX - b_width
        x2 = cX + b_width
        y1 = cY - b_height
        y2 = cY + b_height

        return (x1, y1, x2, y2)

    def extract_json_file(self, ) -> dict:
        """
        Extraction of annotation in JSON file (like COCO annotations)
        """
        patient_metadata = {}
        patient_metadata["metadata"] = self.load_volume_parameters(
        )
        patient_metadata["slice_of_interest"] = self.get_segmented_region(
        )
        patient_metadata["centroids"] = self.get_centroids()
        return patient_metadata

    def save_volume_with_ROI_only(self, save_metadata: bool = True, slice_by_slice: bool = False,) -> None:

        os.makedirs(self.dicom_output, exist_ok=True)
        os.makedirs(self.mask_output, exist_ok=True)

        vol, masks = self.data_ROI_only()
        if slice_by_slice:
            for i in range(vol.shape[2]):
                np.save("{}/{}_{:03d}.npy".format(self.dicom_output,
                        self.patient_id, i), vol[:, :, i])
                np.save("{}/{}_{:03d}.npy".format(self.mask_output,
                        self.patient_id, i), masks[:, :, i])
        else:
            np.save(f"{self.dicom_output}/{self.patient_id}.npy", vol)
            np.save(f"{self.dicom_output}/{self.patient_id}.npy", masks)
