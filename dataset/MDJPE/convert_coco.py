import traceback
import datetime
import json
import os
import argparse
import shutil

import cv2 as cv
import numpy as np
from interval import Interval
from imutils import paths

__CLASS__ = ['__background__', 'human']


def argparser():
    parser = argparse.ArgumentParser("define argument parser for pycococreator!")
    parser.add_argument("-r", "--root_path", default="./images", help="path of root directory")
    parser.add_argument("-p", "--phase_folder", default="test2020", choices=["train2020", "test2020"],
                        help="datasets path of [train2020, test2020]")
    parser.add_argument("-po", "--have_points", default=True, help="if have points we will deal it!")
    return parser.parse_args()


def create_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)


def process(args):
    annotations = {}

    root_path = args.root_path
    phase_folder = args.phase_folder

    # coco annotations info.
    annotations["info"] = {
        "description": "MDJPE dataset convert to COCO format",
        "url": "http://www.meidaojia.com/",
        "version": "1.0",
        "year": 2020,
        "contributor": "Jacky LUO",
        "date_created": "2020/03/23"
    }

    # coco annotations licenses.
    annotations["licenses"] = [{
        "url": "https://www.apache.org/licenses/LICENSE-2.0.html",
        "id": 1,
        "name": "Apache License 2.0"
    }]

    # coco annotations categories.
    annotations["categories"] = []
    for cls, clsname in enumerate(__CLASS__):
        if clsname == '__background__':
            continue
        annotations["categories"].append(
            {
                "supercategory": "object",
                "id": cls,  # 0 or 1
                "name": clsname
            }
        )
        for catdict in annotations["categories"]:
            if catdict["name"] == 'human' and args.have_points:
                catdict["keypoints"] = [str(i) for i in range(65)]  # 65个关键点
                catdict["skeleton"] = [[]]

    # 重新生成路径
    image_paths = os.path.join(root_path, phase_folder)
    json_paths = image_paths.replace("images", "jsons")
    phase = 'train' if 'train' in phase_folder else 'test'
    images_folder = os.path.join(root_path, phase)  # "./images/train"
    create_dir(images_folder)

    print("convert datasets {} to coco format!".format(phase))
    annotations["images"] = []
    annotations["annotations"] = []

    idx = 0
    for image_path, gt_path in zip(paths.list_images(image_paths), paths.list_files(json_paths)):
        # print(image_path, gt_path)
        # 读取对应的json文件
        mdjpe = json.load(open(gt_path, 'r'))
        try:
            points_group = mdjpe['points']
            box = mdjpe['rect'][0]
        except:
            continue

        # 人体检测框
        x1 = box[0]
        y1 = box[1]
        bw = box[2]
        bh = box[3]
        img = cv.imread(image_path)
        height, width, _ = img.shape

        file_name = 'COCO_' + phase + '_' + str(idx).zfill(4) + '.jpg'
        newfilename = os.path.join(images_folder, file_name)
        # os.rename(image_path, newfilename)
        shutil.copyfile(image_path, newfilename)

        annotations["images"].append(
            {
                "license": 1,
                "file_name": file_name,
                "coco_url": "",
                "height": height,
                "width": width,
                "date_captured": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "flickr_url": "",
                "id": idx
            }
        )
        # coco annotations annotations.
        annotations["annotations"].append(
            {
                "id": idx,
                "image_id": idx,
                "category_id": 1,
                "segmentation": [[]],
                "area": bw * bh,
                "bbox": [x1, y1, bw, bh],
                "iscrowd": 0,
            }
        )

        if args.have_points:
            catdict = annotations["annotations"][idx]
            if __CLASS__[catdict["category_id"]] == 'human':
                # 关键点处理
                points = []
                spoints = 0
                for p in points_group:
                    if p:
                        if p[0] in Interval(0, width) and p[1] in Interval(0, height):  # 关键可见
                            points.extend([p[0], p[1], 2])
                        else:
                            points.extend([p[0], p[1], 1])  # 关键点不可见
                        spoints += 1
                    else:
                        points.extend([0, 0, 0])  # 关键点未标注
                catdict["keypoints"] = np.array(points).flatten().tolist()
                catdict["num_keypoints"] = spoints  # v > 0的个数

        if (idx + 1) % 100 == 0:
            print("processing {} ...".format(idx + 1))
        idx += 1

    json_path = os.path.join("./", phase + ".json")
    with open(json_path, "w") as f:
        json.dump(annotations, f)


if __name__ == '__main__':
    print("begining to convert customer format to coco format!")
    args = argparser()
    try:
        process(args)
    except Exception as e:
        traceback.print_exc()
    print("successful to convert customer format to coco format")
