from models import *
import torch
from PIL import Image
import cv2
import numpy as np
from utils.utils import *
import torchvision.transforms as transforms
import torch.nn.functional as F
import datetime
import os
import random

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator




def pad_to_square(img, pad_value):
    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # Add padding
    img = F.pad(img, pad, "constant", value=pad_value)

    return img, pad


def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image

class PytorchYolo():

    '''
    如果需要更换yolotiny
    在model_path,model_def处修改对应路径即可
    '''
    _defaults = {
        "model_path": './weights/yolov3.weights', # 权重放置
        "classes_path": './data/coco.names', # class_names
        "model_def":'./config/yolov3.cfg', # 模型结构图
        "score": 0.3,
        "iou": 0.45,
        "model_image_size": 416,
        "gpu_num": 1,
        "conf_thres":0.8,
        "nms_thres":0.4
    }

    def __init__(self,**kwargs):
        '''
        初始化模型
        :param kwargs:
        '''
        self.__dict__.update(self._defaults) # set up default values
        self.__dict__.update(kwargs) # and update with user overrides
        self.device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classes = load_classes(self.classes_path)

        self.model =self.get_model()

    def get_model(self):
        '''
        创建Darknet版本的模型
        :return:
        '''
        model = Darknet(self.model_def, img_size=self.model_image_size).to(self.device)
        if self.model_path.endswith(".weights"):
            # Load darknet weights
            model.load_darknet_weights(self.model_path)
        else:
            # Load checkpoint weights
            model.load_state_dict(torch.load(self.model_path))
        return model


    def detect_image(self,image,display=True):
        '''
        检测单张图片，输入时PIL格式的图片，
        如果不是，需要进行转换
        :param image:
        :return: tensor([res1,res2...]),
            res1: [x1,y1,x2,y2,conf,cls_conf,cls_pred]
        '''
        prev_time = time.time()
        if isinstance(image,np.ndarray):
            image = Image.fromarray(image)

        org_w,org_h = image.size

        # 预处理
        img = self.preprocess(image)
        Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        input_imgs = Variable(img.type(Tensor))

        '''
            nms最后输出的是[res1,res2...],
            res1: [x1,y1,x2,y2,conf,cls_conf,cls_pred]
        '''
        with torch.no_grad():
            # 原始的输出坐标是(center x, center y, width, height)
            detections = self.model(input_imgs)
            # nms中会转换成(x1, y1, x2, y2)
            detections = non_max_suppression(detections, self.conf_thres, self.nms_thres)[0]

        # Log progress
        current_time = time.time()
        inference_time = datetime.timedelta(seconds=current_time - prev_time)
        prev_time = current_time
        print("\t+ Inference Time: %s" % (inference_time))


        # 将结果从416*416的输入图像映射回原图上
        if detections is not None:
            detections = rescale_boxes(detections, self.model_image_size, (org_h,org_w))
            unique_labels = detections[:, -1].cpu().unique()

        if  display:
            self.display(detections,image)

        return detections


    def display(self,detections,image):
        # 可视化
        plt.figure()
        fig, ax = plt.subplots(1)
        ax.imshow(image)
        for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
            print("\t+ Label: %s, Conf: %.5f" % (self.classes[int(cls_pred)], cls_conf.item()))

            box_w = x2 - x1
            box_h = y2 - y1

            color = [random.random() for i in range(3)]
            color.append(1)
            color = tuple(color)
            # Create a Rectangle patch
            bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor="none")
            # Add the bbox to the plot
            ax.add_patch(bbox)
            # Add label
            plt.text(
                x1,
                y1,
                s=self.classes[int(cls_pred)],
                color="white",
                verticalalignment="top",
                bbox={"color": color, "pad": 0},
            )

        plt.axis("off")
        plt.gca().xaxis.set_major_locator(NullLocator())
        plt.gca().yaxis.set_major_locator(NullLocator())
        plt.show()

        # 考虑是否保存
        # plt.savefig(f"./dog.png", bbox_inches="tight", pad_inches=0.0)
        plt.close()



    def preprocess(self,image):
        '''
        模型预处理：
            1、转成pytorch的tensor
            2、pad
            3、resize
            4、添加维度，pytorch必须按照[batch,channel,w,h]的格式进行输入
        :param image:
        :return:
        '''
        img = transforms.ToTensor()(image)
        img, _ = pad_to_square(img, 0)
        img = resize(img, self.model_image_size)
        img.unsqueeze_(0)
        return img


if __name__ == '__main__':
    print(os.getcwd())
    image = Image.open('./data/samples/dog.jpg')
    pty = PytorchYolo()

    pty.detect_image(image)
