from typing import Union
import pydicom as dcm
from pydicom import dicomdir
import numpy as np
import nibabel as nib
import os
import cv2 as cv
from preprocess.windowing import Preprocessing


class Patient:
    def __init__(
        self,
        patient_id,
        ct_dir,
        mask_dir,
        output_path,
        mask_extension=".nii.gz",
    ) -> None:

        self.patient_id = patient_id
        # on the os.path.join , S is included after the homogenization of the dataset
        # TODO -> Change the directory that is hardcoded
        self.ct_dir = ct_dir
        self.mask_dir = mask_dir

        self.ct_path = os.path.join(ct_dir, patient_id)
        self.mask_path = os.path.join(mask_dir, patient_id+mask_extension)
        self.patient_metadata = self.get_patient_metadata()
        self.ct_metadata = self.load_volume_parameters()
        self.patient_volume = self.get_volume()
        self.patient_masks = self.get_mask()
        self.output_path = output_path

    def get_patient_metadata(self,) -> Union[dcm.FileDataset,None]:
        """
        Each CT scan contains its own DIRFILE that contains 
        information for patient's metadata and the CT volume
        """
        if os.path.exists(os.path.join(self.ct_path, "DIRFILE")):
            metadata = dcm.dcmread(os.path.join(self.ct_path, "DIRFILE"))
        else:
            return None
        return metadata

    def get_filenames(self,) -> Union[list,None]:
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

    def dicom_extraction(self, only_location=False) -> list:
        dicom_files = []
        for filename in self.get_filenames():
            sl = dcm.dcmread(filename)
            # print(f"Image Position {sl.ImagePositionPatient}, Slice location: {sl.SliceLocation}")
            dicom_files.append(sl)
        # print(sl)
        if only_location:
            return sorted([float(x.SliceLocation) for x in dicom_files])

        return sorted(dicom_files, key=lambda x: float(x.SliceLocation))

    def get_volume(self) -> np.ndarray:
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

    def get_segmented_region(self):
        mask_size = self.patient_masks.shape
        sl_locations = self.dicom_extraction(only_location=True)
        indexes = []
        locations = []
        # Loop through the slices of the mask image
        for z,loc in zip(range(mask_size[2]), sl_locations):
            # print(self.patient_masks)
            slice = self.patient_masks[:, :, z]
            if slice.any():
                if len(np.nonzero(slice)[0]) < 100:
                    # If the pixel size of the segmented region
                    # is less than 100 the dont pass it
                    pass
                indexes.append(z)
                locations.append(loc)
        print(len(indexes),len(locations))
        return {
            "Indexes":indexes,
            "SliceLocation":locations
                }

    def data_ROI_only(self):
        indexes = self.get_segmented_region()["Indexes"]
        masks = []
        slices = []
        for i in indexes:
            masks.append(self.patient_masks[:, :, i])
            slices.append(self.patient_volume[:, :, i])

        return np.array(slices).transpose(1, 2, 0), np.array(masks).transpose(1, 2, 0)

    def get_centroids(self):
        """
        Get the centroid of each contour using the openCV module
        """
        slices, masks = self.data_ROI_only()
        print(f"Slices shape : {slices.shape}, Masks shape: {masks.shape}")
        centroids = {}
        for sl in range(slices.shape[2]):
            centroids[sl] = []
            contours, hierarchy = cv.findContours(
                masks[:, :, sl], cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
            box_size = (20, 20)
            for cnt in contours:
                M = cv.moments(cnt)
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                # centroids[sl]["centroid"].append((cX, cY))
                x1, y1, x2, y2 = self.get_bounding_box(
                    cX, cY, box_size=box_size)
                centroids[sl].append({
                    "centroid": {
                        "x": cX,
                        "y": cY,
                    },
                    "bbox": {
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2
                    },

                })

        return centroids

    def get_bounding_box(self, cX, cY, box_size=(40, 40)) -> tuple:
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
        patient_metadata = {}
        patient_metadata["metadata"] = self.load_volume_parameters(
        )
        patient_metadata["slice_of_interest"] = self.get_segmented_region(
        )
        patient_metadata["centroids"] = self.get_centroids()

        return patient_metadata

    def save_volume_with_ROI_only(self, save_metadata=True) -> None:
        new_volume_path = os.path.join(self.output_path, "ct")
        new_mask_path = os.path.join(self.output_path, "mask")
        os.makedirs(new_volume_path, exist_ok=True)
        os.makedirs(new_mask_path, exist_ok=True)
        vol, masks = self.data_ROI_only()
        np.save(f"{new_volume_path}/{self.patient_id}.npy", vol)
        np.save(f"{new_mask_path}/{self.patient_id}.npy", masks)
