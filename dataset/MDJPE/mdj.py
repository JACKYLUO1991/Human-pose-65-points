"""
@author: Jacky LUO
@date:  2020.03
"""
import os
import cv2 as cv
import numpy as np
# import json

# https://zhuanlan.zhihu.com/p/70598884
from dataset.JointsDataset import JointsDataset
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


class MDJDataset(JointsDataset):
    """自定义数据读取方式"""

    def __init__(self, DATASET, stage, transform=None):
        super().__init__(DATASET, stage, transform)
        self.cur_dir = os.path.split(os.path.realpath(__file__))[0]

        self.train_gt_file = 'train.json'
        self.train_gt_path = os.path.join(self.cur_dir, 'annotations',
                                          self.train_gt_file)

        self.val_gt_file = 'test.json'
        self.val_gt_path = os.path.join(self.cur_dir, 'annotations',
                                        self.val_gt_file)

        self.data = self._get_data()
        self.data_num = len(self.data)

    def _get_data(self):
        data = list()

        if self.stage == 'train':
            coco = COCO(self.train_gt_path)
        elif self.stage == 'val':
            coco = COCO(self.val_gt_path)
            self.val_gt = coco
        else:
            pass

        for aid, ann in coco.anns.items():
            img_id = ann['image_id']
            if img_id not in coco.imgs:
                continue
            if ann['iscrowd']:
                continue

            img_name = coco.imgs[img_id]['file_name']
            prefix = 'test' if 'test' in img_name else 'train'
            img_path = os.path.join(self.cur_dir, 'images', prefix, img_name)

            bbox = np.array(ann['bbox'])
            area = ann['area']
            joints = np.array(ann['keypoints']).reshape((-1, 3))
            headRect = np.array([0, 0, 1, 1], np.int32)

            center, scale = self._bbox_to_center_and_scale(bbox)

            # 代码部分冗余
            if np.sum(joints[:, -1] > 0) < self.kp_load_min_num or ann['num_keypoints'] != 65:
                continue

            d = dict(aid=aid,
                     area=area,
                     bbox=bbox,
                     center=center,
                     headRect=headRect,
                     img_id=img_id,
                     img_name=img_name,
                     img_path=img_path,
                     joints=joints,
                     scale=scale)

            data.append(d)

        return data

        # else:
        #     if self.stage == 'val':
        #         det_path = self.val_det_path
        #     else:
        #         det_path = self.test_det_path
        #     dets = json.load(open(det_path))
        #
        #     for det in dets:
        #         if det['image_id'] not in coco.imgs or det['category_id'] != 1:
        #             continue
        #
        #         img_id = det['image_id']
        #         img_name = 'COCO_val2014_000000%06d.jpg' % img_id
        #         img_path = os.path.join(self.cur_dir, 'images', 'val2014',
        #                                 img_name)
        #
        #         bbox = np.array(det['bbox'])
        #         center, scale = self._bbox_to_center_and_scale(bbox)
        #         joints = np.zeros((self.keypoint_num, 3))
        #         score = det['score']
        #         headRect = np.array([0, 0, 1, 1], np.int32)
        #
        #         d = dict(bbox=bbox,
        #                  center=center,
        #                  headRect=headRect,
        #                  img_id=img_id,
        #                  img_name=img_name,
        #                  img_path=img_path,
        #                  joints=joints,
        #                  scale=scale,
        #                  score=score)
        #
        #         data.append(d)

    def _bbox_to_center_and_scale(self, bbox):
        x, y, w, h = bbox

        center = np.zeros(2, dtype=np.float32)
        center[0] = x + w / 2.0
        center[1] = y + h / 2.0

        scale = np.array([w * 1.0 / self.pixel_std, h * 1.0 / self.pixel_std],
                         dtype=np.float32)

        return center, scale

    def evaluate(self, pred_path):
        pred = self.val_gt.loadRes(pred_path)
        coco_eval = COCOeval(self.val_gt, pred, iouType='keypoints')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

    def visualize(self, img, joints, score=None):
        # pairs = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12],
        #          [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3],
        #          [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]
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
        color = np.random.randint(0, 256, (self.keypoint_num, 3)).tolist()

        for i in range(self.keypoint_num):
            if joints[i, 0] > 0 and joints[i, 1] > 0:
                cv.circle(img, tuple(joints[i, :2]), 2, tuple(color[i]), 2)
        if score:
            cv.putText(img, score, (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1.2,
                       (128, 255, 0), 2)

        def draw_line(img, p1, p2):
            c = (0, 0, 255)
            if p1[0] > 0 and p1[1] > 0 and p2[0] > 0 and p2[1] > 0:
                cv.line(img, tuple(p1), tuple(p2), c, 2)

        for pair in pairs:
            draw_line(img, joints[pair[0]], joints[pair[1]])

        return img


if __name__ == '__main__':
    from dataset.attribute import load_dataset

    dataset = load_dataset('MDJPE')
    mdj = MDJDataset(dataset, 'val')
    for i in range(len(mdj)):
        print(i)
        try:
            mdj[i]
        except:
            print(mdj[i]['img_path'])
