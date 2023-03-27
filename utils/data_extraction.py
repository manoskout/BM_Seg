from data_preperation import Patient, split_data, save_metadata_json
from visualization import visualize_boxes
import os
from config import *
from typing import Dict, Tuple

print("The available patients are {pat}".format(pat=PATIENTS))


def main():

    patients_metadata = {}
    os.makedirs(os.path.join(OUTPUT_PATH), exist_ok=True)

    for patient_id in PATIENTS:
        pat = Patient(patient_id=patient_id,
                      ct_dir=DICOM_PATH, mask_dir=ROI_PATH,
                      output_path=OUTPUT_PATH, image_crop=None,
                      window="soft_tissue")
        volume, masks = pat.data_ROI_only()

        patients_metadata[patient_id] = pat.extract_json_file()
        pat.save_volume_with_ROI_only(slice_by_slice=True)
    split_data(patients_metadata, ratio=0.5, )
    save_metadata_json(patients_metadata)

    # visualize_boxes(volume, masks, patients_metadata[patient_id]["centroids"])


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
