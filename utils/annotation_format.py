import json
from typing import Dict, Union


class COCOFormat():

    def __init__(self, info: dict, classes: list = [], licenses: Union[list, None] = None) -> None:
        self.info = info
        self.licenses = licenses
        self.categories = self.set_categories(classes)
        self.images = []
        self.annotations = []

    def set_categories(self, categories) -> list:
        output = []
        for i, category in categories:

            if type(category) == tuple:
                category, supercategory = category
            else:
                supercategory = None
            output.append({
                "id": i,
                "name": category,
                "supercategory": str(supercategory).lower()
            })
        return output

    def set_image(self,):
        """
        {
            "id": int,
            "width": int,
            "height": int
            "file_name": str,
            "license": int,
        }
        """
        pass

    def set_annotation(self,):
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
        pass
