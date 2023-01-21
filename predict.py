from ultralytics import YOLO
import torch
import os
import cv2
import numpy as np
from tqdm import tqdm
import pandas as pd

MODEL_PATH="./checkpoints/model.pt"
BASE_IMAGE_PATH="./yolodata/images/test/"

## Loading All the Test Images
files=os.listdir(BASE_IMAGE_PATH)
all_images=[]
for file in tqdm(files):
    img=np.array(cv2.imread(BASE_IMAGE_PATH+file))
    all_images.append(img)

## Obtaining Model Predictions in a batch-wise manner
model = YOLO(MODEL_PATH)  ## Loading the Finetuned Model
all_res=[]
for i in range(0,len(files),10):
    results=model(all_images[i:i+10])
    all_res.extend(results)

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
for id,res in enumerate(all_res):
    res=res.boxes.boxes.cpu().numpy()
    for pred in res:        
        if pred[-1]>=6: ## To account for one class removal from the train data
            pred[-1]+=1

        val={}
        val['class']=int(pred[-1])
        val['image_path']=str(files[id])
        image_paths.add(str(files[id]))
        val['name']=str(class_to_name[pred[-1]])
        ## Scaling Output to Appropriate Size
        val['xmax']=int((pred[2]/640)*960)  
        val['xmin']=int((pred[0]/640)*960)
        val['ymax']=int((pred[3]/640)*540)
        val['ymin']=int((pred[1]/640)*540)

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
sub_df.to_csv('submission.csv',index=False)