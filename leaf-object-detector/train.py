import os
from xml.etree import ElementTree

import torch
import torchvision
from PIL import Image
from torch.utils.data import Dataset
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, fasterrcnn_resnet50_fpn


class LeafDataset(Dataset):
    def __init__(self, data_dir, transforms=None):
        self.data_dir = data_dir
        self.imgs = [i for i in (sorted(os.listdir(data_dir))) if i[-3:] == 'jpg']
        self.labels = [i for i in (sorted(os.listdir(data_dir))) if i[-3:] == 'xml']
        self.transforms = transforms

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        # Load image
        img = Image.open(f'{self.data_dir}/{self.imgs[idx]}')
        # Load PASCAL VOC format label
        label = ElementTree.parse(f'{self.data_dir}/{self.labels[idx]}')
        root = label.getroot()
        bboxes = []
        for box in root.iter('object'):
            y_min = int(box.find("bndbox/ymin").text)
            x_min = int(box.find("bndbox/xmin").text)
            y_max = int(box.find("bndbox/ymax").text)
            x_max = int(box.find("bndbox/xmax").text)
            bboxes.append([x_min, y_min, x_max, y_max])
        bboxes = torch.as_tensor(bboxes, dtype=torch.float32)
        labels = torch.ones((len(bboxes),), dtype=torch.float32)  # only one class - leaf
        target = {
            'boxes': bboxes,
            'labels': labels,
            'image_id': idx,
        }
        # Transforms
        if self.transforms is not None:
            img, target = self.transforms(img)

        return img, target


if __name__ == '__main__':
    ds = LeafDataset(data_dir='/Users/tyler/Desktop/leaves_dataset')

    model = fasterrcnn_resnet50_fpn(pretrained=True)
    classes = 2  # one for the background
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, classes)

    backend = torchvision.models.mobilenet_v2(pretrained=True).features
    backend.out_channels = 1280

