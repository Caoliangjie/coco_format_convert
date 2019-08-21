# -*- coding: utf-8 -*-
import os
import json
import numpy as np
import pandas as pd
import glob
import cv2
import os
import shutil
from IPython import embed
from tqdm import tqdm
import argparse
#0为背景
parser = argparse.ArgumentParser(description='convert object label')
parser.add_argument('keyframe_dir', metavar='DIR',
                    help='path to frame dir')
parser.add_argument('--mode', type=str, choices=['train', 'val', 'test'])

args = parser.parse_args()
obj_name = open('objects_en.txt','r')
obj_list = [line.rstrip() for line in obj_name]
print(obj_list)
class Csv2CoCo:

    def __init__(self,image_dir,total_annos):
        self.images = []
        self.annotations = []
        self.categories = []
        self.img_id = 0
        self.ann_id = 0
        self.image_dir = image_dir
        self.total_annos = total_annos

    def save_coco_json(self, instance, save_path):
        json.dump(instance, open(save_path, 'w'), ensure_ascii=False, indent=2)  # indent=2 更加美观显示

    # 由txt文件构建COCO
    def to_coco(self, keys):
        self._init_categories()
        for key in keys:
            self.images.append(self._image(key))
            shapes = self.total_annos[key]
            for shape in shapes:
                bboxi = []
                for cor in shape[:-1]:
                    bboxi.append(int(cor))
                label = shape[-1]
                #print('label',label)
                annotation = self._annotation(bboxi,label)
                print('annotation',annotation)
                self.annotations.append(annotation)
                self.ann_id += 1
            self.img_id += 1
        instance = {}
        instance['info'] = 'spytensor created'
        instance['license'] = ['license']
        instance['images'] = self.images
        instance['annotations'] = self.annotations
        instance['categories'] = self.categories
        return instance
    def to_coco_test(self, keys):
        self._init_categories()
        for key in keys:
            self.images.append(self._image(key))
            shapes = self.total_annos[key]
            for shape in shapes:
                bboxi = []
                for cor in shape[:-1]:
                    bboxi.append(int(cor))
                #label = shape[-1]
                #print('label',label)
                #annotation = self._annotation(bboxi,label)
                print('annotation',annotation)
                #self.annotations.append(annotation)
                #self.ann_id += 1
            self.img_id += 1
        instance = {}
        instance['info'] = 'spytensor created'
        instance['license'] = ['license']
        instance['images'] = self.images
        #instance['annotations'] = self.annotations
        instance['categories'] = self.categories
        return instance

    # 构建类别
    def _init_categories(self):
        for k in obj_list:#classname_to_id.items():
            category = {}
            category['id'] = obj_list.index(k)
            category['name'] = k
            self.categories.append(category)

    # 构建COCO的image字段
    def _image(self, path):
        image = {}
        print(path)
        img = cv2.imread(self.image_dir + path)
        image['height'] = img.shape[0]
        image['width'] = img.shape[1]
        image['id'] = self.img_id
        image['file_name'] = path
        return image

    # 构建COCO的annotation字段
    def _annotation(self, shape,label):
        # label = shape[-1]
        points = shape[:4]
        annotation = {}
        annotation['id'] = self.ann_id
        annotation['image_id'] = self.img_id
        annotation['category_id'] = int(obj_list.index(label))
        annotation['segmentation'] = self._get_seg(points)
        annotation['bbox'] = self._get_box(points)
        annotation['iscrowd'] = 0
        annotation['area'] = 1.0
        return annotation

    # COCO的格式： [x1,y1,w,h] 对应COCO的bbox格式
    def _get_box(self, points):
        min_x = points[0]
        min_y = points[1]
        max_x = points[2]
        max_y = points[3]
        return [min_x, min_y, max_x - min_x, max_y - min_y]
    # segmentation
    def _get_seg(self, points):
        min_x = points[0]
        min_y = points[1]
        max_x = points[2]
        max_y = points[3]
        h = max_y - min_y
        w = max_x - min_x
        a = []
        a.append([min_x,min_y, min_x,min_y+0.5*h, min_x,max_y, min_x+0.5*w,max_y, max_x,max_y, max_x,max_y-0.5*h, max_x,min_y, max_x-0.5*w,min_y])
        return a
   

if __name__ == '__main__':
    csv_file = "{}.csv".format(args.mode)
    image_dir = args.keyframe_dir#"/home/sda/videonet/train/image/train/"
    #print('image_dir',image_dir)
    saved_coco_path = "./"
    # 整合csv格式标注文件
    total_csv_annotations = {}
    annotations = pd.read_csv(csv_file,header=None).values
    for annotation in annotations:
        #print(annotation[0].split(os.sep)[-2]+'/'+annotation[0].split(os.sep)[-1])
        key = annotation[0].split(os.sep)[-2]+'/'+annotation[0].split(os.sep)[-1]
        value = np.array([annotation[1:]])
        if key in total_csv_annotations.keys():
            total_csv_annotations[key] = np.concatenate((total_csv_annotations[key],value),axis=0)
        else:
            total_csv_annotations[key] = value
    # 按照键值划分数据
    total_keys = list(total_csv_annotations.keys())
    in_keys = total_keys
    print("{}_n:".format(args.mode), len(in_keys))#, 'val_n:', len(val_keys))
    # 创建必须的文件夹
    if not os.path.exists('%scoco/annotations/'%saved_coco_path):
        os.makedirs('%scoco/annotations/'%saved_coco_path)
    if not os.path.exists('%scoco/train2017/'%saved_coco_path):
        os.makedirs('%scoco/train2017/'%saved_coco_path)
    if not os.path.exists('%scoco/val2017/'%saved_coco_path):
        os.makedirs('%scoco/val2017/'%saved_coco_path)
    if not os.path.exists('%scoco/test2017/'%saved_coco_path):
        os.makedirs('%scoco/test2017/'%saved_coco_path)
    # 把训练集转化为COCO的json格式
    for file in in_keys:
      if not os.path.exists('{}coco/{}2017/{}'.format(saved_coco_path,args.mode,file.split('/')[0])):
        #print(file.split('/')[0])
        os.makedirs('{}coco/{}2017/{}'.format(saved_coco_path,args.mode,file.split('/')[0]))
        if not os.path.exists("{}coco/{}2017/{}".format(saved_coco_path,args.mode,file)):
          shutil.copy(image_dir+file,"{}coco/{}2017/{}".format(saved_coco_path,args.mode,file))
      elif os.path.exists('{}coco/{}2017/{}'.format(saved_coco_path,args.mode,file.split('/')[0])):
        if not os.path.exists("{}coco/{}2017/{}".format(saved_coco_path,args.mode,file)):
          shutil.copy(image_dir+file,"{}coco/{}2017/{}".format(saved_coco_path,args.mode,file))
    l2c = Csv2CoCo(image_dir=image_dir,total_annos=total_csv_annotations)
    if args.mode != 'test':
      instance = l2c.to_coco(in_keys)
    elif args.mode == 'test':
      instance = l2c.to_coco_test(in_keys)
    l2c.save_coco_json(instance, '{}coco/annotations/instances_{}2017.json'.format(saved_coco_path,args.mode))

