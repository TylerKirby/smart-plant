import argparse
import os
from xml.etree import ElementTree

import torch
import torchvision
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, fasterrcnn_resnet50_fpn
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.transforms import ToTensor, RandomHorizontalFlip, Compose
from utils.engine import evaluate, train_one_epoch, collate_fn


class LeafDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.imgs = [i for i in (sorted(os.listdir(data_dir))) if i[-3:] == 'jpg']
        self.labels = [i for i in (sorted(os.listdir(data_dir))) if i[-3:] == 'xml']
        self.transforms = ToTensor()

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
        labels = torch.ones((len(bboxes),), dtype=torch.int64)  # only one class - leaf
        target = {
            'boxes': bboxes,
            'labels': labels,
        }
        return self.transforms(img), target


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', '-p')
    args = parser.parse_args()
    # Create model
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    classes = 2  # one for the background
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, classes)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    print(f'Using {device} for training')

    # Currently we have only 36 examples. We'll use 30 for training and 6 for validation
    dataset = LeafDataset(data_dir=args.path)
    train_set, val_set = torch.utils.data.random_split(dataset, [30, 6])
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, collate_fn=collate_fn)

    # Training loop
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)
    num_epochs = 10
    for e in range(num_epochs):
        # train for one epoch
        train_one_epoch(model, optimizer, train_loader, device, e, print_freq=10)
        # update learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        print('entering eval')
        print(len(val_loader))
        evaluate(model, val_loader, device=device)
        # save model
        if e % 10 == 0:
            torch.save({
                'epoch': e,
                'model_state_dict': model.state_dict()},
                f'model_checkpoints/midas_1.1_' + str(e) + 'EPOCH_checkpoint.pt')
