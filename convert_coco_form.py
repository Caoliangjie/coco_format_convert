import os
import json
import cv2
import time
import argparse
import csv
parser = argparse.ArgumentParser(description='convert object label')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('keyframe_dir', metavar='DIR',
                    help='path to frame dir')
parser.add_argument('--mode', type=str, choices=['train', 'val', 'test'])

args = parser.parse_args()

root = os.path.join(args.data, args.mode) 

video_list = open(os.path.join(root, 'videolist.txt'),'r')

output_list = open('%s_list.txt' % args.mode,'w')

obj_name = open('objects.txt','r')
obj_list = [line.rstrip() for line in obj_name]
print(obj_list)
train_category = []
video_count = 0
start = time.time()
key_frame_count = 0
#output_folder = '%s_label' % args.mode
#if not os.path.exists(output_folder):
 #   os.makedirs(output_folder)
with open('val.csv','w') as f1:
 for vid in video_list:
    label = json.load(open(os.path.join(root, 'label', 'sample_' + vid.rstrip().split('.')[0] + '.json'), 'r'))
    writer = csv.writer(f1)
    for shot in label['shots']:
        keyframe = shot['keyframe']
        image_path = os.path.join(args.keyframe_dir, args.mode, vid[:-1], '%05d.jpg'% keyframe)
        for target in shot['targets'] :
          if target['category'] == 0:
           if os.path.isfile(image_path):
                        xmin = target['bbox']['x']
                        xmax = xmin + target['bbox']['width']
                        ymin = target['bbox']['y']
                        ymax = ymin + target['bbox']['height']
                        cls_id = obj_list[int(target['tag'])]
                        writer.writerow([image_path,xmin,xmax,ymin,ymax,cls_id])
