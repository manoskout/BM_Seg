from data_preperation import Patient, split_data
from visualization import visualize_boxes
import os
from config import *
from typing import Dict, Tuple
from annotation_format import COCOFormat
print("The available patients are {pat}".format(pat=PATIENTS))


def main():

    patients_metadata = {}
    os.makedirs(os.path.join(OUTPUT_PATH), exist_ok=True)
    categories = [
        {
            "supercategory": "bone",
            "id": 1,
            "name": "bone marrow"
        }
    ]
    annotationsCOCO = COCOFormat()

    # Get the general information from the dataset
    pat = Patient(patient_id=PATIENTS[0],
                  ct_dir=DICOM_PATH, mask_dir=ROI_PATH,
                  output_path=OUTPUT_PATH, image_crop=None,
                  window="soft_tissue")
    metadata = pat.patient_metadata

    annotationsCOCO.init_categories(
        categories=categories)
    annotationsCOCO.init_dataset_info(
        description=metadata["Description"],
        date_created=metadata["SeriesDate"]
    )

    # Append all the data
    for patient_id in PATIENTS:

        pat = Patient(patient_id=patient_id,
                      ct_dir=DICOM_PATH, mask_dir=ROI_PATH,
                      output_path=OUTPUT_PATH, image_crop=INPUT_SIZE,
                      window=WINDOWING,
                      )
        vol, masks, indexes = pat.data_ROI_only()

        patients_metadata[patient_id] = pat.extract_json_file()
        # random number
        ann_uid = 5240
        for i in range(vol.shape[2]):
            curr_fname = "{}_{:03d}.npy".format(patient_id, i)
            annotationsCOCO.set_image(
                image_id=indexes[i], height=vol.shape[0], width=vol.shape[1], file_name=curr_fname)
            for bbox in patients_metadata[patient_id]["centroids"][i]["bbox"]:
                annotationsCOCO.set_annotation(
                    # the category id is hardcoded to 1 (bone marrow)
                    image_id=indexes[i], category_id=1, bbox=bbox, annotation_id=ann_uid
                )
                ann_uid += 1
        pat.save_volume_with_ROI_only(slice_by_slice=True)
        print(f"[Info - {patient_id}] Data extraction finished")
    print(f"[Info] All the patients are extracted")

    split_sets = split_data(ratio=0.5, valid=True, val_ratio=0.2)
    print(f"[Info] Image split is done")
    if split_sets["valid"]:
        annotationsCOCO.split_annotation(
            split_sets["train"],
            split_sets["test"],
            train_path=os.path.join(
                TRAINING_PATH, "images/_annotations.coco.json"),
            test_path=os.path.join(
                TESTING_PATH, "images/_annotations.coco.json"),
            valid_data=split_sets["valid"],
            valid_path=os.path.join(
                VALIDATION_PATH, "images/_annotations.coco.json")
        )
    else:
        annotationsCOCO.split_annotation(
            split_sets["train"],
            split_sets["test"],
            train_path=os.path.join(
                TRAINING_PATH, "images/_annotations.coco.json"),
            test_path=os.path.join(
                TESTING_PATH, "images/_annotations.coco.json")
        )
    print(f"[Info] Annotation split is done")

    # print(annotationsCOCO.get_output())
    # print(f"Train : {split_sets['train']}, Test : {split_sets['test']}")
    # annotationsCOCO.create_COCOJSON(
    #     os.path.join(OUTPUT_PATH, "annotations.json"))


if __name__ == "__main__":
    main()
