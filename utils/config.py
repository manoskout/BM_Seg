import os

DICOM_PATH = "../data/ct/"
ROI_PATH = "../data/roi/"

OUTPUT_PATH = "../data/output"
TRAINING_PATH = "../data/output/train"
TESTING_PATH = "../data/output/test"
VALIDATION_PATH = "../data/output/valid"


INPUT_SIZE = (600, 600)

WINDOWING = "soft_tissue"

PATIENTS = [i for i in os.listdir(
    DICOM_PATH) if i != "DIRFILE" and i != ".DS_Store"]
