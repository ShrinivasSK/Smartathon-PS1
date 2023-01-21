from ultralytics import YOLO
import torch
import os
import cv2
import numpy as np
from tqdm import tqdm
import pandas as pd
from imgaug.augmentables.bbs import BoundingBox

MODEL_PATH="./checkpoints/model.pt"
AUG_MODEL_PATH="./checkpoints/model_aug.pt"
BASE_IMAGE_PATH="./yolodata/images/test/"

## Loading All the Test Images
files=os.listdir(BASE_IMAGE_PATH)
all_images=[]
for file in tqdm(files):
    img=np.array(cv2.imread(BASE_IMAGE_PATH+file))
    all_images.append(img)


## Obtaining Model Predictions in a batch-wise manner
model = YOLO(MODEL_PATH) 
model_aug=YOLO(AUG_MODEL_PATH)
all_res=[]
for i in range(0,len(files),1):
    results=model(all_images[i:i+1])
    results_aug=model_aug(all_images[i:i+1])
    all_res.append([(results,results_aug)])

## Obtain Class ID To name map from train data
df=pd.read_csv('./yolodata/train.csv')
classes=df['class'].values
names=df['name'].values
i=0
class_to_name={}
while len(class_to_name)!=11:
    class_to_name[int(str(classes[i]).split('.')[0])]=names[i]
    i+=1

## Converting the Model Predictions to required format 
data=[]
image_paths=set()
for id,both_pred in enumerate(all_res):
    res=both_pred[0][0][0].boxes.boxes.cpu().numpy()
    res_aug=both_pred[0][1][0].boxes.boxes.cpu().numpy()
    bboxes=[]
    for pred in res:
        bboxes.append((BoundingBox(
            x2=pred[2],x1=pred[0],y1=pred[1],y2=pred[3],label=pred[-1]
        ),pred[-2]))
    bboxes_aug=[]
    for pred in res_aug:
        bboxes_aug.append((BoundingBox(
            x2=pred[2],x1=pred[0],y1=pred[1],y2=pred[3],label=pred[-1]
        ),pred[-2]))
    final_res=[]
    for i in range(len(res)):
        cnt=0
        for j in range(len(res_aug)):
            if bboxes[i][0].label==bboxes_aug[j][0].label:
                iou=bboxes_aug[j][0].iou(bboxes[i][0])
                if iou>=0.7:
                    if bboxes[i][1]<=bboxes_aug[j][1]:
                        cnt+=1
                        break
        if cnt==0:
            final_res.append(bboxes[i])
    for i in range(len(res_aug)):
        cnt=0
        for j in range(len(res)):
            if bboxes_aug[i][0].label==bboxes[j][0].label:
                iou=bboxes[j][0].iou(bboxes_aug[i][0])
                if iou>=0.5:
                    if bboxes_aug[i][1]<=bboxes[j][1]:
                        cnt+=1
                        break
        if cnt==0:
            final_res.append(bboxes_aug[i])
    for pred in final_res:
        pred=pred[0]
        if pred.label>=6:
            pred.label+=1
        val={}
        val['class']=int(pred.label)
        val['image_path']=str(files[id])
        image_paths.add(str(files[id]))
        val['name']=str(class_to_name[pred.label])
        val['xmax']=int((pred.x2/640)*960)
        val['xmin']=int((pred.x1/640)*960)
        val['ymax']=int((pred.y2/640)*540)
        val['ymin']=int((pred.y1/640)*540)

        data.append(val)

## Appending empty result for images that do not have any predictions
df_test=pd.read_csv('./yolodata/test.csv')
image_paths_all=df_test['image_path'].values
for image_path in image_paths_all:
    if image_path not in image_paths:
        val={}
        val['image_path']=str(image_path)
        data.append(val)

## Generating the final submission file
sub_df=pd.DataFrame(data)
sub_df.to_csv('submission_aug.csv',index=False)