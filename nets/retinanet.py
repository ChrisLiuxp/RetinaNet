import torch.nn as nn
import torch.nn.functional as F  
import torch
import math
from nets.resnet import resnet18,resnet34,resnet50,resnet101,resnet152
from utils.anchors import Anchors
from nets.head import FCOSClsCenterHead, FCOSRegHead, FCOSPositions

class PyramidFeatures(nn.Module):
    def __init__(self, C3_size, C4_size, C5_size, feature_size=256):
        super(PyramidFeatures, self).__init__()

        self.P5_1 = nn.Conv2d(C5_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P5_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        self.P4_1 = nn.Conv2d(C4_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P4_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        self.P3_1 = nn.Conv2d(C3_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P3_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        self.P6 = nn.Conv2d(C5_size, feature_size, kernel_size=3, stride=2, padding=1)

        self.P7_1 = nn.ReLU()
        self.P7_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=2, padding=1)

    def forward(self, inputs):
        C3, C4, C5 = inputs
        _, _, h4, w4 = C4.size()
        _, _, h3, w3 = C3.size()

        P5_x = self.P5_1(C5)
        P5_upsampled_x = F.interpolate(P5_x, size=(h4, w4))
        P5_x = self.P5_2(P5_x)

        P4_x = self.P4_1(C4)
        P4_x = P5_upsampled_x + P4_x
        P4_upsampled_x = F.interpolate(P4_x, size=(h3, w3))
        P4_x = self.P4_2(P4_x)

        P3_x = self.P3_1(C3)
        P3_x = P3_x + P4_upsampled_x
        P3_x = self.P3_2(P3_x)

        P6_x = self.P6(C5)

        P7_x = self.P7_1(P6_x)
        P7_x = self.P7_2(P7_x)

        return [P3_x, P4_x, P5_x, P6_x, P7_x]


class RegressionModel(nn.Module):
    def __init__(self, num_features_in, num_anchors=9, feature_size=256):
        super(RegressionModel, self).__init__()

        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()

        self.output = nn.Conv2d(feature_size, num_anchors * 4, kernel_size=3, padding=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.act3(out)

        out = self.conv4(out)
        out = self.act4(out)

        out = self.output(out)

        # out is B x C x W x H, with C = 4*num_anchors
        out = out.permute(0, 2, 3, 1)

        return out.contiguous().view(out.shape[0], -1, 4)


class ClassificationModel(nn.Module):
    def __init__(self, num_features_in, num_anchors=9, num_classes=80, prior=0.01, feature_size=256):
        super(ClassificationModel, self).__init__()

        self.num_classes = num_classes
        self.num_anchors = num_anchors

        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()

        self.output = nn.Conv2d(feature_size, num_anchors * num_classes, kernel_size=3, padding=1)
        self.output_act = nn.Sigmoid()

    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.act3(out)

        out = self.conv4(out)
        out = self.act4(out)

        out = self.output(out)
        out = self.output_act(out)

        # out is B x C x W x H, with C = n_classes + n_anchors
        out1 = out.permute(0, 2, 3, 1)

        batch_size, width, height, channels = out1.shape

        out2 = out1.view(batch_size, width, height, self.num_anchors, self.num_classes)

        return out2.contiguous().view(x.shape[0], -1, self.num_classes)

class Resnet(nn.Module):
    def __init__(self, phi, load_weights=False):
        super(Resnet, self).__init__()
        self.edition = [resnet18,resnet34,resnet50,resnet101,resnet152]
        model = self.edition[phi](load_weights)
        del model.avgpool
        del model.fc
        self.model = model

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        feat1 = self.model.layer2(x)
        feat2 = self.model.layer3(feat1)
        feat3 = self.model.layer4(feat2)

        return [feat1,feat2,feat3]

class Retinanet(nn.Module):

    def __init__(self, num_classes, phi, pretrain_weights=False):
        super(Retinanet, self).__init__()
        self.pretrain_weights = pretrain_weights
        self.backbone_net = Resnet(phi,pretrain_weights)
        fpn_sizes = {
            0: [128, 256, 512],
            1: [128, 256, 512],
            2: [512, 1024, 2048],
            3: [512, 1024, 2048],
            4: [512, 1024, 2048],
        }[phi]

        self.fpn = PyramidFeatures(fpn_sizes[0], fpn_sizes[1], fpn_sizes[2])
        # self.regressionModel = RegressionModel(256)
        # self.classificationModel = ClassificationModel(256, num_classes=num_classes)
        # self.anchors = Anchors()
        # self._init_weights()
        self.num_classes = num_classes
        self.cls_head = FCOSClsCenterHead(256,
                                          self.num_classes,
                                          num_layers=4,
                                          prior=0.01)
        self.regcenter_head = FCOSRegHead(256, num_layers=4)

        self.strides = torch.tensor([8, 16, 32, 64, 128], dtype=torch.float)
        self.positions = FCOSPositions(self.strides)

        self.scales = nn.Parameter(torch.FloatTensor([1., 1., 1., 1., 1.]))

    # def _init_weights(self):
    #     if not self.pretrain_weights:
    #         print("_init_weights")
    #         for m in self.modules():
    #             if isinstance(m, nn.Conv2d):
    #                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
    #                 m.weight.data.normal_(0, math.sqrt(2. / n))
    #             elif isinstance(m, nn.BatchNorm2d):
    #                 m.weight.data.fill_(1)
    #                 m.bias.data.zero_()
    #
    #     print("_init_classificationModel")
    #     prior = 0.01
    #     self.classificationModel.output.weight.data.fill_(0)
    #     self.classificationModel.output.bias.data.fill_(-math.log((1.0 - prior) / prior))
    #     print("_init_regressionModel")
    #     self.regressionModel.output.weight.data.fill_(0)
    #     self.regressionModel.output.bias.data.fill_(0)


    # def forward(self, inputs):
    #
    #     p3, p4, p5 = self.backbone_net(inputs)
    #
    #     features = self.fpn([p3, p4, p5])
    #
    #     regression = torch.cat([self.regressionModel(feature) for feature in features], dim=1)
    #
    #     classification = torch.cat([self.classificationModel(feature) for feature in features], dim=1)
    #
    #     anchors = self.anchors(features)
    #
    #     return features, regression, classification, anchors
    def forward(self, inputs):
        self.batch_size, _, _, _ = inputs.shape
        device = inputs.device

        p3, p4, p5 = self.backbone_net(inputs)

        features = self.fpn([p3, p4, p5])


        self.fpn_feature_sizes = []
        cls_heads, reg_heads, center_heads = [], [], []
        for feature, scale in zip(features, self.scales):
            # 每层特征图的大小，如64×64
            self.fpn_feature_sizes.append([feature.shape[3], feature.shape[2]])
            cls_outs, center_outs = self.cls_head(feature)
            # [N,num_classes,H,W] -> [N,H,W,num_classes]
            cls_outs = cls_outs.permute(0, 2, 3, 1).contiguous()
            cls_heads.append(cls_outs)

            reg_outs = self.regcenter_head(feature)
            # [N,4,H,W] -> [N,H,W,4]
            reg_outs = reg_outs.permute(0, 2, 3, 1).contiguous()
            reg_outs = reg_outs * scale
            reg_heads.append(reg_outs)
            # [N,1,H,W] -> [N,H,W,1]
            center_outs = center_outs.permute(0, 2, 3, 1).contiguous()
            center_heads.append(center_outs)

        del features

        self.fpn_feature_sizes = torch.tensor(
            self.fpn_feature_sizes).to(device)

        batch_positions = self.positions(self.batch_size,
                                         self.fpn_feature_sizes)

        # if input size:[B,3,640,640]
        # features shape:[[B, 256, 80, 80],[B, 256, 40, 40],[B, 256, 20, 20],[B, 256, 10, 10],[B, 256, 5, 5]]
        # cls_heads shape:[[B, 80, 80, 80],[B, 40, 40, 80],[B, 20, 20, 80],[B, 10, 10, 80],[B, 5, 5, 80]]
        # reg_heads shape:[[B, 80, 80, 4],[B, 40, 40, 4],[B, 20, 20, 4],[B, 10, 10, 4],[B, 5, 5, 4]]
        # center_heads shape:[[B, 80, 80, 1],[B, 40, 40, 1],[B, 20, 20, 1],[B, 10, 10, 1],[B, 5, 5, 1]]
        # batch_positions shape:[[B, 80, 80, 2],[B, 40, 40, 2],[B, 20, 20, 2],[B, 10, 10, 2],[B, 5, 5, 2]]

        return cls_heads, reg_heads, center_heads, batch_positions