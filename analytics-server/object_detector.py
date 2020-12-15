import base64
import io

import torch
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from PIL import Image
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, fasterrcnn_resnet50_fpn
from torchvision.transforms import ToTensor


class ObjectDetector:
    def __init__(self):
        self.model = fasterrcnn_resnet50_fpn(pretrained=True)
        classes = 2  # one for the background
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, classes)
        checkpoint = torch.load('leaf_od_weights.pt', map_location=torch.device('cpu'))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

    def detect(self, img):
        transforms = ToTensor()
        img = transforms(img)
        with torch.no_grad():
            prediction = self.model([img])
        img = Image.fromarray(img.mul(255).permute(1, 2, 0).byte().numpy())

        conf_thres = 0.15
        plt.imshow(img)
        current_axis = plt.gca()
        leaf_count = 0
        for i in range(len(prediction[0]['scores'])):
            obj = prediction[0]
            score = prediction[0]['scores'][i]
            if score > conf_thres:
                leaf_count += 1
                label = obj['labels'][i]
                score = obj['scores'][i]
                boxes = obj['boxes'][i]
                xmin, ymin, xmax, ymax = boxes
                xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
                current_axis.add_patch(
                    plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, color='yellow', fill=False, linewidth=2))
        plt_bytes = io.BytesIO()
        plt.savefig(plt_bytes, format='jpg')
        plt_bytes.seek(0)
        plt_base64 = base64.b64encode(plt_bytes.read())
        return leaf_count, "data:image/jpeg;base64," + plt_base64.decode('UTF-8')
