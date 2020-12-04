import argparse

import torch
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, fasterrcnn_resnet50_fpn
from torchvision.transforms import ToTensor


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight_path', '-w', default='/Users/tyler/Projects/smart-plant/leaf-object-detector/leaf_od_weights.pt')
    parser.add_argument('--img_path', '-i', default='/Users/tyler/Desktop/leaves_dataset/IMG_0786.jpg')
    args = parser.parse_args()

    model = fasterrcnn_resnet50_fpn(pretrained=True)
    classes = 2  # one for the background
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, classes)

    checkpoint = torch.load(args.weight_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    transforms = ToTensor()
    img = Image.open(args.img_path)
    img = transforms(img)
    # put the model in evaluation mode
    model.eval()
    with torch.no_grad():
        prediction = model([img])
    img = Image.fromarray(img.mul(255).permute(1, 2, 0).byte().numpy())

    CONF_THRESH = 0.15
    plt.imshow(img)

    current_axis = plt.gca()

    colors = plt.cm.hsv(np.linspace(0, 1, 9)).tolist()  # Set the colors for the bounding boxes
    color = colors[0]
    for i in range(len(prediction[0]['scores'])):
        obj = prediction[0]
        score = prediction[0]['scores'][i]
        if score > CONF_THRESH:
            label = obj['labels'][i]
            score = obj['scores'][i]
            boxes = obj['boxes'][i]
            xmin, ymin, xmax, ymax = boxes
            xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
            label = '{}'.format(label)
            # color = colors[class_to_color[label]]
            current_axis.add_patch(
                plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, color='yellow', fill=False, linewidth=2))
    plt.savefig('example.png')
