import os

DICOM_PATH = "../data/ct/"
ROI_PATH = "../data/roi/"
OUTPUT_PATH = "../data/output"
TRAINING_PATH = "../data/output/train"
TESTING_PATH = "../data/output/test"

PATIENTS = [i for i in os.listdir(
    DICOM_PATH) if i != "DIRFILE" and i != ".DS_Store"]