import numpy as np
import math


def generate_anchor_base(base_size=16, ratios=[0.5, 1, 2],
                         anchor_scales=[8, 16, 32]):
    anchor_base = np.zeros((len(ratios) * len(anchor_scales), 4),
                           dtype=np.float32)
    for i in range(len(ratios)):
        for j in range(len(anchor_scales)):
            h = base_size * anchor_scales[j] * np.sqrt(ratios[i])
            w = base_size * anchor_scales[j] * np.sqrt(1. / ratios[i])

            index = i * len(anchor_scales) + j
            anchor_base[index, 0] = - h / 2.
            anchor_base[index, 1] = - w / 2.
            anchor_base[index, 2] = h / 2.
            anchor_base[index, 3] = w / 2.
    return anchor_base


def _enumerate_shifted_anchor(anchor_base, feat_stride, height, width):
    # 计算网格中心点
    shift_x = np.arange(0, width * feat_stride, feat_stride)
    shift_y = np.arange(0, height * feat_stride, feat_stride)
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shift = np.stack((shift_x.ravel(), shift_y.ravel(),
                      shift_x.ravel(), shift_y.ravel(),), axis=1)

    # 每个网格点上的9个先验框
    A = anchor_base.shape[0]
    K = shift.shape[0]
    anchor = anchor_base.reshape((1, A, 4)) + \
             shift.reshape((K, 1, 4))
    # 所有的先验框
    anchor = anchor.reshape((K * A, 4)).astype(np.float32)
    return anchor


def fun_iou(box1, box2):
    '''
    box:[x1, y1, x2, y2]
    '''
    in_h = min(box1[3], box2[3]) - max(box1[1], box2[1])
    in_w = min(box1[2], box2[2]) - max(box1[0], box2[0])
    inner = 0 if in_h < 0 or in_w < 0 else in_h * in_w
    union = (box1[2] - box1[0]) * (box1[3] - box1[1]) + \
            (box2[2] - box2[0]) * (box2[3] - box2[1]) - inner
    # print(inner)
    iou = inner / union
    return iou


def fun_cn(box1, box2):
    l = box1[0] - box2[0]
    t = box1[1] - box2[1]
    r = box2[2] - box1[0]
    b = box2[3] - box1[1]
    cn = math.sqrt((min(l, r) / max(l, r)) * (min(t, b) / max(t, b)))
    return cn


def fun_iou_cn(zuo_x, zuo_y, width, height):
    # 全部点的中心点坐标
    center_x_c = 197.49033
    center_y_c = 138.98067
    width_c = 181.01935
    height_c = 362.0387

    point_center = (center_x_c, center_y_c, center_x_c + width_c, center_y_c + height_c)
    point = (zuo_x + width / 2, zuo_y + height / 2, width, height)
    cn = fun_cn(point, point_center)
    point = (zuo_x, zuo_y, zuo_x + width, zuo_y + height)
    iou = fun_iou(point, point_center)
    return iou, cn


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    nine_anchors = generate_anchor_base()
    print(nine_anchors)
    height, width, feat_stride = 20, 20, 32
    anchors_all = _enumerate_shifted_anchor(nine_anchors, feat_stride, height, width)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.ylim(-300, 800)
    plt.xlim(-300, 900)
    shift_x = np.arange(0, width * feat_stride, feat_stride)
    shift_y = np.arange(0, height * feat_stride, feat_stride)
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    plt.scatter(shift_x, shift_y, color="y")
    box_widths = anchors_all[:, 2] - anchors_all[:, 0]
    box_heights = anchors_all[:, 3] - anchors_all[:, 1]

    # hang = (10,9)
    # lie = (9,10)
    points = [(5, 7), (15, 9), (6, 10), (10, 9)]
    rect = plt.Rectangle([anchors_all[10 * 180 + 9 * 9 + 1, 0], anchors_all[10 * 180 + 9 * 9 + 1, 1]],
                         box_widths[10 * 180 + 9 * 9 + 1], box_heights[10 * 180 + 9 * 9 + 1], color="r",
                         fill=False)

    ax.add_patch(rect)
    for i in range(height):
        for j in range(width):
            # print(i*180+j*9)
            if (i, j) in points:
                #######################################
                # 这两种形状做对比
                # good = i * 180 + j * 9 + 1
                good = i * 180 + j * 9 + 4
                #######################################
                rect = plt.Rectangle([anchors_all[good, 0], anchors_all[good, 1]],
                                     box_widths[good], box_heights[good], color="b",
                                     fill=False)
                iou, centerness = fun_iou_cn(anchors_all[good, 0], anchors_all[good, 1], box_widths[good],
                                             box_heights[good])
                print(i, j)
                print('iou:', iou, '   centerness:', centerness)
                ax.add_patch(rect)

    # 1881,1882,1883,1884,1885,1886,1887,1888,1889
    # for i in [1785,1885]:
    #     rect = plt.Rectangle([anchors_all[i, 0],anchors_all[i, 1]],box_widths[i],box_heights[i],color="r",fill=False)
    #     ax.add_patch(rect)

    plt.show()