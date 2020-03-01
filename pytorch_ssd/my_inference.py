import torch
from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd, create_mobilenetv1_ssd_predictor
import numpy as np
import cv2

class PytorchSSD():
    _defaults = {
        "model_path": './models/mobilenet-v1-ssd-mp-0_675.pth',  # 权重放置
        "label_path": './models/voc-model-labels.txt',  # class_names
        "net_type": 'mb1-ssd'
    }

    def __init__(self):
        self.__dict__.update(self._defaults) # set up default values
        self.class_names = [name.strip() for name in open(self.label_path).readlines()]
        self.model = self.get_model()

    def get_model(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        net_type = 'mb1-ssd'  # sys.argv[1]
        model_path = './models/mobilenet-v1-ssd-mp-0_675.pth'  # sys.argv[2]
        label_path = './models/voc-model-labels.txt'  # sys.argv[3]

        class_names = [name.strip() for name in open(label_path).readlines()]

        net = create_mobilenetv1_ssd(len(class_names), is_test=True)
        net.load(model_path)
        predictor = create_mobilenetv1_ssd_predictor(net, candidate_size=200, device=device)
        return predictor

    def detect(self,image):
        if not isinstance(image,np.ndarray):
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        boxes, labels, probs = self.model.predict(image, 10, 0.4)

        predict_names = [ self.class_names[lb] for lb in labels]

        return boxes,predict_names,probs

