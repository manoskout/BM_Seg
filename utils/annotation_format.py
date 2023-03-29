import json
from typing import Dict, Union
import numpy as np
from datetime import date
from data_preperation import save_metadata_json


class COCOFormat():

    def __init__(self, licenses: Union[list, None] = None, ) -> None:
        self.classes = None
        self.licenses = licenses
        self.images = []
        self.annotations = []

    def init_categories(self, categories: list) -> None:
        self.classes = categories

    def init_dataset_info(self,
                          description: str = "None",
                          url: str = "None",
                          version: str = "None",
                          contributor: str = "None",
                          date_created: str = str(date.today())
                          ) -> None:

        self.info = {
            "description": description,
            "url": url,
            "version": version,
            "year": date_created.split("-")[0],
            "contributor": contributor
        }

    # def init_classes(self, classes: list):
    #     output = []
    #     for i, category in classes:

    #         if type(category) == tuple:
    #             category, supercategory = category
    #         else:
    #             supercategory = None
    #         output.append({
    #             "id": i,
    #             "name": category,
    #             "supercategory": str(supercategory).lower()
    #         })
    #     self.classes = output

    def bbox_conversion(self, bbox: Union[np.ndarray, list]) -> list:
        x1, y1, x2, y2 = bbox
        return [x1, y1, x2-x1, y2-y1]

    def set_image(self,
                  image_id: int,
                  width: int,
                  height: int,
                  file_name: str,
                  ):
        """
        {
            "id": int,
            "width": int,
            "height": int
            "file_name": str,
            "license": int,
        }
        """
        self.images.append({
            "id": image_id,
            "license": None,
            "file_name": file_name,
            "width": width,
            "height": height,
        })

    def set_annotation(self,
                       annotation_id: int,
                       image_id: int,
                       category_id: int,
                       bbox: list,
                       segmentation: Union[list, None] = None,
                       area: Union[int, None] = None,
                       iscrowd: Union[int, None] = None,
                       convert_bbox=True):
        """
        The annotation contains list of each individual object annotation from every single image in the dataset
        {
            "id": str
            "image_id": str
            "category_id": str
            "bbox": list (e.g [x1,y1, heigh,width])
            "segmentation": RLE or [polygon]
            "area" : int (is the area of the bounding box, the integer is a pixel value )
            "iscrowd": 0 or 1 (if we wave segmentation)
        }
        The bbox in COCO is the x,y co-ordicate of the top left and the heigh and width
        """
        self.curr_ann_id = annotation_id
        self.annotations.append({
            "id": annotation_id,
            "image_id": image_id,
            "category_id": category_id,
            "bbox": self.bbox_conversion(bbox) if convert_bbox else bbox,
            #  FUTURE WORK
            # "segmentation": [],
            # "area": ,
            # "iscrowd" : 0,
        })

    def get_output(self) -> dict:
        final = {}
        final["info"] = self.info
        final["licenses"] = self.licenses
        final["categories"] = self.classes
        final["images"] = self.images
        final["annotations"] = self.annotations
        return final

    def create_COCOJSON(self, path: str) -> None:
        save_metadata_json(data=self.get_output(), output_file=path)

    def split_annotation(self, training_data, testing_data, train_path, test_path):
        final = self.get_output()
        train = {}
        test = {}

        test["info"] = final["info"]
        train["info"] = final["info"]

        test["licenses"] = final["licenses"]
        train["licenses"] = final["licenses"]

        test["categories"] = final["categories"]
        train["categories"] = final["categories"]
        test["images"] = [image for image in final["images"]
                          if image["file_name"].split("_")[0] in testing_data]

        train["images"] = [image for image in final["images"]
                           if image["file_name"].split("_")[0] in training_data]

        test_imgIDs = [image["id"] for image in test["images"]]
        train_imgIDs = [image["id"] for image in train["images"]]

        test["annotations"] = [annotation for annotation in final["annotations"]
                               if annotation["image_id"] in test_imgIDs]
        train["annotations"] = [annotation for annotation in final["annotations"]
                                if annotation["image_id"] in train_imgIDs]

        save_metadata_json(train, output_file=train_path)
        save_metadata_json(test, output_file=test_path)
