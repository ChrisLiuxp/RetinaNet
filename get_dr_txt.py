#-------------------------------------#
#       mAP所需文件计算代码
#       具体教程请查看Bilibili
#       Bubbliiiing
#-------------------------------------#
import cv2
import numpy as np
import colorsys
import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from retinanet import RetinaNet
from nets.retinanet import Retinanet
from PIL import Image,ImageFont, ImageDraw
from utils.utils import non_max_suppression, bbox_iou, decodebox, letterbox_image, retinanet_correct_boxes
from nets.decode import FCOSDecoder
from tqdm import tqdm


def preprocess_input(image):
    image /= 255
    mean=(0.406, 0.456, 0.485)
    std=(0.225, 0.224, 0.229)
    image -= mean
    image /= std
    return image

class mAP_RetinaNet(RetinaNet):
    #---------------------------------------------------#
    #   检测图片
    #---------------------------------------------------#
    def detect_image(self,image_id,image):
        self.confidence = 0.01
        self.iou = 0.5
        f = open("./input/detection-results/"+image_id+".txt","w")
        image_shape = np.array(np.shape(image)[0:2])

        crop_img = np.array(letterbox_image(image, self.image_size))
        photo = np.array(crop_img,dtype = np.float32)
        photo = np.transpose(preprocess_input(photo), (2, 0, 1))
        images = []
        images.append(photo)
        images = np.asarray(images)

        with torch.no_grad():
            images = torch.from_numpy(images)
            if self.cuda:
                images = images.cuda()
            # _, regression, classification, anchors = self.net(images)
            #
            # regression = decodebox(regression, anchors, images)
            # detection = torch.cat([regression,classification],axis=-1)

            cls_heads, reg_heads, center_heads, batch_positions = self.net(images)
            decode = FCOSDecoder(image_shape[0],image_shape[1])
            batch_scores, batch_classes, batch_pred_bboxes, batch_pred_cls = decode(
                cls_heads, reg_heads, center_heads, batch_positions)

            detection = torch.cat([batch_pred_bboxes, batch_pred_cls], axis=-1)

            batch_detections = non_max_suppression(detection, len(self.class_names),
                                                    conf_thres=self.confidence,
                                                    nms_thres=0.3)
        try:
            batch_detections = batch_detections[0].cpu().numpy()
        except:
            return image

        top_index = batch_detections[:,4] > self.confidence
        top_conf = batch_detections[top_index,4]
        top_label = np.array(batch_detections[top_index,-1],np.int32)
        top_bboxes = np.array(batch_detections[top_index,:4])
        top_xmin, top_ymin, top_xmax, top_ymax = np.expand_dims(top_bboxes[:,0],-1),np.expand_dims(top_bboxes[:,1],-1),np.expand_dims(top_bboxes[:,2],-1),np.expand_dims(top_bboxes[:,3],-1)

        # 去掉灰条
        boxes = retinanet_correct_boxes(top_ymin,top_xmin,top_ymax,top_xmax,np.array(self.image_size),image_shape)

        for i, c in enumerate(top_label):
            predicted_class = self.class_names[c]
            score = str(top_conf[i])

            top, left, bottom, right = boxes[i]
            f.write("%s %s %s %s %s %s\n" % (predicted_class, score[:6], str(int(left)), str(int(top)), str(int(right)),str(int(bottom))))

        f.close()
        return

retinanet = mAP_RetinaNet()
image_ids = open('VOCdevkit/VOC2007/ImageSets/Main/test.txt').read().strip().split()

if not os.path.exists("./input"):
    os.makedirs("./input")
if not os.path.exists("./input/detection-results"):
    os.makedirs("./input/detection-results")
if not os.path.exists("./input/images-optional"):
    os.makedirs("./input/images-optional")


for image_id in tqdm(image_ids):
    image_path = "./VOCdevkit/VOC2007/JPEGImages/"+image_id+".jpg"
    image = Image.open(image_path)
    # 开启后在之后计算mAP可以可视化
    # image.save("./input/images-optional/"+image_id+".jpg")
    retinanet.detect_image(image_id,image)
    # print(image_id," done!")


print("Conversion completed!")