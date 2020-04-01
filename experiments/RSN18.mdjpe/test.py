"""
@author: Jacky LUO
@date:  2020.03.24
"""

from config import cfg
from network import RSN
from lib.utils.transforms import get_affine_transform
from dataset.attribute import MDJPE

import os
from PIL import Image
import cv2 as cv
import numpy as np
from collections import OrderedDict

import torch
import torchvision.transforms as transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


def get_results(outputs, centers, scales, kernel, shifts):
    scales *= 200
    nr_img = outputs.shape[0]
    preds = np.zeros((nr_img, cfg.DATASET.KEYPOINT.NUM, 2))
    maxvals = np.zeros((nr_img, cfg.DATASET.KEYPOINT.NUM, 1))
    for i in range(nr_img):
        score_map = outputs[i].copy()
        # score_map = score_map / 255 + 0.5  # not understand well!
        score_map = score_map / 255
        kps = np.zeros((cfg.DATASET.KEYPOINT.NUM, 2))
        scores = np.zeros((cfg.DATASET.KEYPOINT.NUM, 1))
        border = 10
        dr = np.zeros((cfg.DATASET.KEYPOINT.NUM,
                       cfg.OUTPUT_SHAPE[0] + 2 * border, cfg.OUTPUT_SHAPE[1] + 2 * border))
        dr[:, border: -border, border: -border] = outputs[i].copy()
        # post-Gaussian filter
        for w in range(cfg.DATASET.KEYPOINT.NUM):
            dr[w] = cv.GaussianBlur(dr[w], (kernel, kernel), 0)
        for w in range(cfg.DATASET.KEYPOINT.NUM):
            for j in range(len(shifts)):
                if j == 0:
                    lb = dr[w].argmax()
                    y, x = np.unravel_index(lb, dr[w].shape)
                    dr[w, y, x] = 0
                    x -= border
                    y -= border
                lb = dr[w].argmax()
                py, px = np.unravel_index(lb, dr[w].shape)
                dr[w, py, px] = 0
                px -= border + x
                py -= border + y
                ln = (px ** 2 + py ** 2) ** 0.5
                if ln > 1e-3:
                    x += shifts[j] * px / ln
                    y += shifts[j] * py / ln
            x = max(0, min(x, cfg.OUTPUT_SHAPE[1] - 1))
            y = max(0, min(y, cfg.OUTPUT_SHAPE[0] - 1))
            kps[w] = np.array([x * 4 + 2, y * 4 + 2])
            scores[w, 0] = score_map[w, int(round(y) + 1e-9), int(round(x) + 1e-9)]
        kps[:, 0] = kps[:, 0] / cfg.INPUT_SHAPE[1] * scales[i][0] + centers[i][0] - scales[i][0] * 0.5
        kps[:, 1] = kps[:, 1] / cfg.INPUT_SHAPE[0] * scales[i][1] + centers[i][1] - scales[i][1] * 0.5
        preds[i] = kps
        maxvals[i] = scores

    return preds, maxvals


def box_to_center_scale(box, model_image_width, model_image_height):
    """convert a box to center, scale information required for pose transformation
    Parameters
    ----------
    box : list of tuple
        list of length 2 with two tuples of floats representing
        bottom left and top right corner of a box
    model_image_width : int
    model_image_height : int
    Returns
    -------
    (numpy array, numpy array)
        Two numpy arrays, coordinates for the center of the box and the scale of the box
    """
    center = np.zeros((2), dtype=np.float32)

    bottom_left_corner = box[0]
    top_right_corner = box[1]
    box_width = top_right_corner[0] - bottom_left_corner[0]
    box_height = top_right_corner[1] - bottom_left_corner[1]
    bottom_left_x = bottom_left_corner[0]
    bottom_left_y = bottom_left_corner[1]
    center[0] = bottom_left_x + box_width * 0.5
    center[1] = bottom_left_y + box_height * 0.5

    aspect_ratio = model_image_width * 1.0 / model_image_height
    pixel_std = MDJPE.PIXEL_STD  # 200

    if box_width > aspect_ratio * box_height:
        box_height = box_width * 1.0 / aspect_ratio
    elif box_width < aspect_ratio * box_height:
        box_width = box_height * aspect_ratio
    scale = np.array(
        [box_width * 1.0 / pixel_std, box_height * 1.0 / pixel_std],
        dtype=np.float32)
    if center[0] != -1:
        scale = scale * 1.25
    # scale[0] *= (1 + MDJPE.TEST.X_EXTENTION)
    # scale[1] *= (1 + MDJPE.TEST.Y_EXTENTION)

    return center, scale


def get_person_detection_boxes(model, img, thres):
    """人体检测框"""

    tforms = transforms.Compose([transforms.ToTensor()])
    timg = tforms(img)
    pred = model([timg])
    pred_classes = [COCO_INSTANCE_CATEGORY_NAMES[i]
                    for i in list(pred[0]['labels'].numpy())]
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])]
                  for i in list(pred[0]['boxes'].detach().numpy())]
    # 按得分高 --> 低排列
    pred_score = list(pred[0]['scores'].detach().numpy())
    if not pred_score:
        return []
    pred_t = [pred_score.index(x) for x in pred_score if x > thres][-1]
    pred_boxes = pred_boxes[:pred_t + 1]
    pred_classes = pred_classes[:pred_t + 1]

    person_boxes = []
    m_area = 0
    for idx, box in enumerate(pred_boxes):
        if pred_classes[idx] == 'person':
            area = (box[1][0] - box[0][0]) * (box[1][1] - box[0][1])
            if area > m_area:
                m_area = area
                if len(person_boxes):
                    person_boxes.pop(-1)
                person_boxes.append(box)

    return person_boxes


def get_person_keypoints(model, image, center, scale):
    rotation = 0

    # pose estimation transformation
    trans = get_affine_transform(center, scale, rotation, cfg.INPUT_SHAPE)
    model_input = cv.warpAffine(
        image,
        trans,
        (int(cfg.INPUT_SHAPE[1]), int(cfg.INPUT_SHAPE[0])),
        flags=cv.INTER_LINEAR)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg.INPUT.MEANS,
                             std=cfg.INPUT.STDS),
    ])

    # pose estimation inference
    model_input = transform(model_input).unsqueeze(0)
    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        # compute output heatmap
        output = model(model_input)
        preds, maxvals = get_results(
            output.clone().cpu().numpy(),
            np.asarray([center]),
            np.asarray([scale]),
            cfg.TEST.GAUSSIAN_KERNEL,
            cfg.TEST.SHIFT_RATIOS
        )

        return preds.squeeze(), maxvals.squeeze()


def visualize(img, joints, score=None):
    pairs = [
        [0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9],
        [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18],
        [18, 19], [19, 20], [20, 21], [21, 22], [22, 23], [23, 24], [24, 25], [25, 26], [26, 27],
        [27, 28], [28, 29], [29, 30], [30, 31], [31, 32], [32, 33], [33, 34], [34, 35], [35, 36],
        [36, 37], [37, 38], [38, 39], [39, 40], [40, 41], [41, 42], [42, 43], [43, 44], [44, 45],
        [45, 46], [46, 47], [47, 48], [48, 49], [49, 50], [50, 51], [51, 52], [52, 53], [53, 54],
        [54, 55], [55, 56], [56, 57], [57, 58], [58, 59], [59, 60], [60, 61], [61, 62], [0, 64],
        [62, 64], [63, 64]
    ]

    if score:
        cv.putText(img, score, (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1.2,
                   (128, 255, 0), 2)

    def draw_line(img, p1, p2, c):
        if p1[0] > 0 and p1[1] > 0 and p2[0] > 0 and p2[1] > 0:
            cv.line(img, tuple(map(int, p1)), tuple(map(int, p2)), c, 3)

    for idx, pair in enumerate(pairs):
        if idx < 31:
            draw_line(img, joints[pair[0]], joints[pair[1]], c=(255, 0, 0))
        elif idx < 62:
            draw_line(img, joints[pair[0]], joints[pair[1]], c=(0, 255, 0))
        else:
            draw_line(img, joints[pair[0]], joints[pair[1]], c=(0, 0, 255))

    for i in range(cfg.DATASET.KEYPOINT.NUM):
        if joints[i, 0] > 0 and joints[i, 1] > 0:
            cv.circle(img, tuple(map(int, joints[i, :2])), 8, (255, 255, 255), -1)

    return img


def visualize_v2(img, joints, scores, thresh=0.1):
    """改版后的可视化效果"""
    extra_pairs = [[63, 64], [62, 64], [0, 64]]
    pairs = [
        [0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9],
        [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18],
        [18, 19], [19, 20], [20, 21], [21, 22], [22, 23], [23, 24], [24, 25], [25, 26], [26, 27],
        [27, 28], [28, 29], [29, 30], [30, 31], [31, 32], [32, 33], [33, 34], [34, 35], [35, 36],
        [36, 37], [37, 38], [38, 39], [39, 40], [40, 41], [41, 42], [42, 43], [43, 44], [44, 45],
        [45, 46], [46, 47], [47, 48], [48, 49], [49, 50], [50, 51], [51, 52], [52, 53], [53, 54],
        [54, 55], [55, 56], [56, 57], [57, 58], [58, 59], [59, 60], [60, 61], [61, 62],
    ]
    num_range = list(range(joints.shape[0]))
    whole_dict = OrderedDict(zip(num_range, scores.tolist()))
    remain_dict = {k: v for k, v in whole_dict.items() if v > thresh}

    keys = list(remain_dict.keys())
    groups = []
    for i in range(len(keys) - 1):
        groups.append([keys[i], keys[i + 1]])

    def draw_line(img, p1, p2, c):
        if p1[0] > 0 and p1[1] > 0 and p2[0] > 0 and p2[1] > 0:
            cv.line(img, tuple(map(int, p1)), tuple(map(int, p2)), c, 3)

    # 求数组交集
    pairs = [v for v in pairs if v in groups]
    for idx, pair in enumerate(pairs):
        draw_line(img, joints[pair[0]], joints[pair[1]], c=(0, 255, 0))
        if pair[1] <= 31:
            draw_line(img, joints[pair[0]], joints[pair[1]], c=(255, 0, 0))

    # 额外的三个连接
    for pair in extra_pairs:
        if joints[pair[0]][0] > 0 and joints[pair[0]][1] > 0 and joints[pair[1]][0] > 0 and joints[pair[1]][1] > 0:
            draw_line(img, joints[pair[0]], joints[pair[1]], c=(0, 0, 255))

    # 按照阈值可视化
    keys = set(np.asarray(pairs).reshape(-1).tolist())
    for i in keys:
        if joints[i, 0] > 0 and joints[i, 1] > 0:
            # cv.putText(img, str(i), (int(joints[i, 0]), int(joints[i, 1])),
            #            cv.FONT_HERSHEY_SIMPLEX, 0.8, (128, 255, 0), 2)
            cv.circle(img, tuple(map(int, joints[i, :2])), 8, (255, 255, 255), -1)

    return img


if __name__ == '__main__':
    img = cv.imread("7.jpg")
    pil_img = Image.fromarray(img)

    # 加载人体检测模型, 后期可以替换成轻量级人体检测模型
    det_model = fasterrcnn_resnet50_fpn(pretrained=True)
    det_model.eval()

    # 加载人体关键点模型
    # pose_model = RSN(cfg)
    # device = torch.device(cfg.MODEL.DEVICE)
    # pose_model.to(device)

    # model_file = cfg.MODEL.WEIGHT
    # if os.path.exists(model_file):
    #     state_dict = torch.load(
    #         model_file, map_location=lambda storage, loc: storage)
    #     state_dict = state_dict['model']
    #     pose_model.load_state_dict(state_dict)
    model_file = cfg.MODEL.WEIGHT
    if os.path.exists(model_file):
        pose_model = torch.load(model_file, map_location=lambda storage, loc: storage)
    boxes = get_person_detection_boxes(det_model, pil_img, thres=0.8)
    center, scale = box_to_center_scale(boxes[0], cfg.INPUT_SHAPE[1], cfg.INPUT_SHAPE[0])
    pose_preds, pose_scores = get_person_keypoints(pose_model, img, center, scale)

    # 以下可以存储为.json的格式...
    print("Keypoints Scores:")
    print(pose_scores)

    # 可视化
    img = visualize_v2(img, pose_preds, pose_scores, thresh=0.45)
    cv.imwrite("result.png", img)
