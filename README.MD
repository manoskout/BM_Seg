# CT Scan Preprocessing

This code provides a tool for preprocessing CT scan data, including windowing and visualizing the results.

## Installation

To use this code, you need to have the necessary dependencies installed. You can install them using pip:

```shell
pip install -r requirements.txt
```

## Usage
The main functionality of this code is provided by the `main()` function in the `ct_preprocessing.py` file. You can modify this file to suit your needs, and then run it to preprocess your CT scan data.

You can modify the following global variables to specify the paths to your data:

- **DICOM_PATH**: the path to the directory containing the DICOM files
- **ROI_PATH**: the path to the directory containing the ROI files
- **OUTPUT_PATH**: the path to the directory where you want to save the output files

You can also modify the following lines to customize the preprocessing:

```python
preprocessing = Preprocessing(
    window="soft_tissue",
    metadata=patients_metadata[patient_id]["metadata"]
)
```
By default, this code uses the *'soft_tissue'* windowing setting. You can change this by modifying the window parameter. In the code block below it all the available choices.
```json
{
    "abdomen": {"wl": 60, "ww": 400},
    "angio": {"wl": 300, "ww": 600},
    "bone": {"wl": 400, "ww": 1800},
    "temporal_bones": {"wl": 600, "ww": 2800},
    "soft_tissue": {"wl": 50, "ww": 250},
    "brain": {"wl": 40, "ww": 80},
    "mediastinum": {"wl": 50, "ww": 350},
    "lungs": {"wl": -600, "ww": 1500}
}

```
You can also modify the metadata parameter to provide any additional metadata that you want to use in the preprocessing.