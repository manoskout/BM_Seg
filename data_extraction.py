from utils.data_preperation import Patient
from utils.visualization import visualize_boxes, visualize_windowing
from preprocess.windowing import Preprocessing
import os
import json

# GLOBAL Definition
DICOM_PATH = "/Users/manoskout/Desktop/BM_Seg/data/ct/"
ROI_PATH = "/Users/manoskout/Desktop/BM_Seg/data/roi/"
OUTPUT_PATH = "/Users/manoskout/Desktop/BM_Seg/data/output"
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
    with open(f'{OUTPUT_PATH}/metadata.json', 'w') as outfile:
        json.dump(data, outfile, cls=SetEncoder)


def main():

    patients_metadata = {}

    for patient_id in PATIENTS:
        pat = Patient(patient_id=patient_id,
                      ct_dir=DICOM_PATH, mask_dir=ROI_PATH, output_path=OUTPUT_PATH)
        volume, masks = pat.data_ROI_only()

        patients_metadata[patient_id] = pat.extract_json_file()
        # print(patients_metadata[patient_idq]["centroids"])
        # print(type(volume))
        preprocessing = Preprocessing(
            # volume=volume,
            window="soft_tissue",
            metadata=patients_metadata[patient_id]["metadata"]
        )
        break
        # pat.save_volume_with_ROI_only()
        # break
    save_metadata_json(patient_id)

    visualize_boxes(volume, masks, patients_metadata[patient_id]["centroids"])

    windowed_vol = preprocessing.windowing.volume_windowing(volume)

    visualize_windowing(
        volume=volume, windowed=windowed_vol)


if __name__ == "__main__":
    main()
