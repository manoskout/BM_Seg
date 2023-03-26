import torch.nn as nn
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead
import numpy as np
from torch.utils.data import Dataset, DataLoader


# Define the dataset


class CTSliceDataset(Dataset):
    def __init__(self, image_files, targets):
        self.image_files = image_files
        self.targets = targets

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        image_file = self.image_files[index]
        image = np.load(image_file)
        target = self.targets[index]
        return image, target

# Define the model


class FasterRCNNModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # Define the backbone
        backbone = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            weights="DEFAULT")
        backbone.out_channels = 256
        # Define the anchor generator
        anchor_sizes = ((32, 64, 128, 256, 512),)
        aspect_ratios = ((0.5, 1.0, 2.0),)
        anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)

        # Define the region proposal network (RPN)
        rpn_head = RPNHead(
            256, anchor_generator.num_anchors_per_location()[0])
        rpn_pre_nms_top_n_train = 2000
        rpn_pre_nms_top_n_test = 1000
        rpn_post_nms_top_n_train = 2000
        rpn_post_nms_top_n_test = 1000
        rpn_nms_thresh = 0.7
        rpn_fg_iou_thresh = 0.7
        rpn_bg_iou_thresh = 0.3
        rpn_batch_size_per_image = 256
        rpn_positive_fraction = 0.5
        box_roi_pool = torchvision.ops.MultiScaleRoIAlign(
            featmap_names=['0'], output_size=7, sampling_ratio=2)
        box_head = torchvision.models.detection.faster_rcnn.TwoMLPHead(
            256 * 7 * 7, 1024)
        box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
            1024, num_classes)
        roi_heads = torchvision.models.detection.faster_rcnn.RoIHeads(
            box_roi_pool, box_head, box_predictor,
            rpn_fg_iou_thresh, rpn_bg_iou_thresh, rpn_batch_size_per_image, rpn_positive_fraction,
            rpn_pre_nms_top_n_train, rpn_pre_nms_top_n_test, rpn_post_nms_top_n_train, rpn_post_nms_top_n_test,
            rpn_nms_thresh
        )

        # Define the final model
        self.model = FasterRCNN(backbone, num_classes=num_classes,
                                rpn_anchor_generator=anchor_generator, rpn_head=rpn_head, roi_heads=roi_heads)
        # self.model.backbone.apply(weight_init)
        # self.model.rpn.apply(weight_init)
        # self.model.roi_heads.apply(weight_init)

    def forward(self, x, y=None):
        if self.training:
            return self.model(x, y)
        else:
            return self.model(x)
