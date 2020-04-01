"""
@author: Jacky LUO
@date:  2020.03.27
"""

from easydict import EasyDict as edict


class COCO:
    NAME = 'COCO'

    KEYPOINT = edict()
    KEYPOINT.NUM = 17
    KEYPOINT.FLIP_PAIRS = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12],
                           [13, 14], [15, 16]]
    KEYPOINT.UPPER_BODY_IDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    KEYPOINT.LOWER_BODY_IDS = [11, 12, 13, 14, 15, 16]
    KEYPOINT.LOAD_MIN_NUM = 1

    INPUT_SHAPE = (256, 192)  # height, width
    OUTPUT_SHAPE = (64, 48)
    WIDTH_HEIGHT_RATIO = INPUT_SHAPE[1] / INPUT_SHAPE[0]

    PIXEL_STD = 200
    COLOR_RGB = False

    TRAIN = edict()
    TRAIN.BASIC_EXTENTION = 0.05
    TRAIN.RANDOM_EXTENTION = True
    TRAIN.X_EXTENTION = 0.6
    TRAIN.Y_EXTENTION = 0.8
    TRAIN.SCALE_FACTOR_LOW = -0.25
    TRAIN.SCALE_FACTOR_HIGH = 0.25
    TRAIN.SCALE_SHRINK_RATIO = 0.8
    TRAIN.ROTATION_FACTOR = 45
    TRAIN.PROB_ROTATION = 0.5
    TRAIN.PROB_FLIP = 0.5
    TRAIN.NUM_KEYPOINTS_HALF_BODY = 3
    TRAIN.PROB_HALF_BODY = 0.3
    TRAIN.X_EXTENTION_HALF_BODY = 0.6
    TRAIN.Y_EXTENTION_HALF_BODY = 0.8
    TRAIN.ADD_MORE_AUG = False
    TRAIN.GAUSSIAN_KERNELS = [(15, 15), (11, 11), (9, 9), (7, 7), (5, 5)]

    TEST = edict()
    TEST.FLIP = True
    TEST.X_EXTENTION = 0.01 * 9.0
    TEST.Y_EXTENTION = 0.015 * 9.0
    TEST.SHIFT_RATIOS = [0.25]
    TEST.GAUSSIAN_KERNEL = 5


class MPII:
    NAME = 'MPII'

    KEYPOINT = edict()
    KEYPOINT.NUM = 16
    KEYPOINT.FLIP_PAIRS = [[0, 5], [1, 4], [2, 3], [10, 15], [11, 14], [12, 13]]
    KEYPOINT.UPPER_BODY_IDS = [7, 8, 9, 10, 11, 12, 13, 14, 15]
    KEYPOINT.LOWER_BODY_IDS = [0, 1, 2, 3, 4, 5, 6]
    KEYPOINT.LOAD_MIN_NUM = 1

    INPUT_SHAPE = (256, 256)  # height, width
    OUTPUT_SHAPE = (64, 64)
    WIDTH_HEIGHT_RATIO = INPUT_SHAPE[1] / INPUT_SHAPE[0]

    PIXEL_STD = 200
    COLOR_RGB = False

    TRAIN = edict()
    TRAIN.BASIC_EXTENTION = 0.0
    TRAIN.RANDOM_EXTENTION = False
    TRAIN.X_EXTENTION = 0.25
    TRAIN.Y_EXTENTION = 0.25
    TRAIN.SCALE_FACTOR_LOW = -0.25
    TRAIN.SCALE_FACTOR_HIGH = 0.25
    TRAIN.SCALE_SHRINK_RATIO = 1.0
    TRAIN.ROTATION_FACTOR = 60
    TRAIN.PROB_ROTATION = 0.5
    TRAIN.PROB_FLIP = 0.5
    TRAIN.NUM_KEYPOINTS_HALF_BODY = 8
    TRAIN.PROB_HALF_BODY = 0.5
    TRAIN.X_EXTENTION_HALF_BODY = 0.6
    TRAIN.Y_EXTENTION_HALF_BODY = 0.6
    TRAIN.ADD_MORE_AUG = False
    TRAIN.GAUSSIAN_KERNELS = [(15, 15), (11, 11), (9, 9), (7, 7), (5, 5)]

    TEST = edict()
    TEST.FLIP = True
    TEST.X_EXTENTION = 0.25
    TEST.Y_EXTENTION = 0.25
    TEST.SHIFT_RATIOS = [0.25]
    TEST.GAUSSIAN_KERNEL = 9


class MDJPE:
    """美到家数据集"""
    NAME = 'MDJPE'

    KEYPOINT = edict()
    KEYPOINT.NUM = 65
    # KEYPOINT.FLIP_PAIRS = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12],
    #                        [13, 14], [15, 16]]
    # KEYPOINT.UPPER_BODY_IDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    # KEYPOINT.LOWER_BODY_IDS = [11, 12, 13, 14, 15, 16]

    # 31, 63, 64点位没有对称点
    KEYPOINT.FLIP_PAIRS = [
        [0, 62], [1, 61], [2, 60], [3, 59], [4, 58],
        [5, 57], [6, 56], [7, 55], [8, 54], [9, 53],
        [10, 52], [11, 51], [12, 50], [13, 49], [14, 48],
        [15, 47], [16, 46], [17, 45], [18, 44], [19, 43],
        [20, 42], [21, 41], [22, 40], [23, 39], [24, 38],
        [25, 37], [26, 36], [27, 35], [28, 34], [29, 33],
        [30, 32]
    ]
    KEYPOINT.UPPER_BODY_IDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
                               18, 19, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
                               56, 57, 58, 59, 60, 61, 62, 63, 64]
    KEYPOINT.LOWER_BODY_IDS = [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37,
                               38, 39, 40, 41, 42]
    KEYPOINT.LOAD_MIN_NUM = 1

    INPUT_SHAPE = (384, 288)  # or (256, 192)
    OUTPUT_SHAPE = (96, 72)  # or (64, 48)
    WIDTH_HEIGHT_RATIO = INPUT_SHAPE[1] / INPUT_SHAPE[0]

    PIXEL_STD = 200
    COLOR_RGB = False

    TRAIN = edict()
    TRAIN.BASIC_EXTENTION = 0.05
    TRAIN.RANDOM_EXTENTION = True
    TRAIN.X_EXTENTION = 0.6
    TRAIN.Y_EXTENTION = 0.8
    TRAIN.SCALE_FACTOR_LOW = -0.25
    TRAIN.SCALE_FACTOR_HIGH = 0.25
    TRAIN.SCALE_SHRINK_RATIO = 0.8
    TRAIN.ROTATION_FACTOR = 45
    TRAIN.PROB_ROTATION = 0.5
    TRAIN.PROB_FLIP = 0.5
    TRAIN.NUM_KEYPOINTS_HALF_BODY = 20  # 这个参数啥意思？
    TRAIN.PROB_HALF_BODY = 0.3
    TRAIN.X_EXTENTION_HALF_BODY = 0.6
    TRAIN.Y_EXTENTION_HALF_BODY = 0.8
    TRAIN.ADD_MORE_AUG = False
    TRAIN.GAUSSIAN_KERNELS = [(15, 15), (11, 11), (9, 9), (7, 7), (5, 5)]

    TEST = edict()
    TEST.FLIP = True
    TEST.X_EXTENTION = 0.01 * 9.0
    TEST.Y_EXTENTION = 0.015 * 9.0
    TEST.SHIFT_RATIOS = [0.25]
    TEST.GAUSSIAN_KERNEL = 5


def load_dataset(name):
    if name == 'COCO':
        dataset = COCO()
    elif name == 'MPII':
        dataset = MPII()
    elif name == 'MDJPE':
        dataset = MDJPE()
    return dataset
