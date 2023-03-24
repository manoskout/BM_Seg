from data_preperation import Patient
from visualization import visualize_boxes, visualize_windowing
from preprocess.windowing import Preprocessing
import os
import json
import numpy as np
import cv2 as cv

# GLOBAL Definition
DICOM_PATH = "../data/ct/"
ROI_PATH = "../data/roi/"
OUTPUT_PATH = "../data/output"
# Get the patient names
PATIENTS = [i for i in os.listdir(
    DICOM_PATH) if i != "DIRFILE" and i != ".DS_Store"]
print("The available patients are {pat}".format(pat=PATIENTS))


class SetEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, set):
            return list(obj)
        return json.JSONEncoder.default(self, obj)


def save_metadata_json(data):
    output_file = f'{OUTPUT_PATH}/metadata.json'
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    if os.path.exists(output_file):
        os.remove(output_file)
    with open(output_file, 'w') as outfile:
        json.dump(data, outfile, cls=SetEncoder)


def main():

    patients_metadata = {}
    os.makedirs(os.path.join(OUTPUT_PATH), exist_ok=True)

    for patient_id in PATIENTS:
        pat = Patient(patient_id=patient_id,
                      ct_dir=DICOM_PATH, mask_dir=ROI_PATH, output_path=OUTPUT_PATH, image_crop=(384, 384))
        volume, masks = pat.data_ROI_only()

        patients_metadata[patient_id] = pat.extract_json_file()

        preprocessing = Preprocessing(
            window="soft_tissue",
            metadata=patients_metadata[patient_id]["metadata"]
        )
        break

        # pat.save_volume_with_ROI_only(slice_by_slice=True)
    save_metadata_json(patients_metadata)

    windowed_vol = preprocessing.windowing.volume_windowing(volume)
    visualize_boxes(volume, masks, patients_metadata[patient_id]["centroids"])

    visualize_windowing(
        volume=volume, windowed=windowed_vol)


if __name__ == "__main__":
    main()


# for i in range(volume.shape[2]):
    # ss = cv.ximgproc.segmentation.createSelectiveSearchSegmentation()
    # img = cv.cvtColor(volume[:, :, i].astype(
    #     np.uint8), cv.COLOR_GRAY2BGR)
    # # img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # ss.setBaseImage(img)

    # # Set the search mode for Selective Search algorithm
    # ss.switchToSelectiveSearchQuality()

    # # Generate region proposals using Selective Search algorithm
    # rects = ss.process()

    # # Print the total number of region proposals
    # print('Total Region Proposals:', len(rects))
    # # Display the top 100 region proposals on the original image
    # for i, rect in enumerate(rects[:100]):
    #     x, y, w, h = rect
    #     cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)

    # cv.imshow('Region Proposals', img)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
