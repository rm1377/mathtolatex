
import numpy as np 

import torch 
import torchvision 
import matplotlib.pyplot as plt 
import glob 
import pandas as pd
import json 
import os 
import random 
import cv2 






class Resize(object):
  def __init__(self,size):
    self.size = size if isinstance(size,tuple) else tuple(size)
  def __call__(self,x):
    temp = np.zeros((self.size[0],self.size[1],3),dtype=x.dtype)
    h,w, c = x.shape
    r = max([w/self.size[1],h/self.size[0]])
    x = cv2.resize(x,(int(w/r),int(h/r)))
    px = self.size[1] - int(w/r) 
    py = self.size[0] - int(h/r)
    temp[py//2:py//2+x.shape[0],px//2:px//2+x.shape[1],:] = x
    return temp






def bind_objects(frame,thresh_img, minArea, plot=False):
    '''Draws bounding boxes and detects when cars are crossing the line
    frame: numpy image where boxes will be drawn onto
    thresh_img: numpy image after subtracting the background and all thresholds and noise reduction operations are applied
    '''
    cnts,_ = cv2.findContours(thresh_img,1,2)                #this line is for opencv 2.4, and also now for OpenCV 4.4, so this is the current one
    #cnts = sorted(cnts,key = cv2.contourArea,reverse=False)
    #frame = cv2.drawContours(frame, cnts, -1, (0,255,0), 3)
    cnt_id         = 1
    cur_centroids  = []
    boxes = [] 
    max_area = (0,0)
    for idx,c in enumerate(cnts):
    
        if cv2.contourArea(c) < minArea and False:           #ignore contours that are smaller than this area
            continue

        x,y,w,h   = cv2.boundingRect(c)
        if w*h>max_area[1]:
          max_area = (idx, w*h)

        box = np.array([x,y,x+w,y+h],int)
        boxes.append(box) 
        if plot:
            cv2.rectangle(frame,(box[0],box[1]),(box[2],box[3]),(255,0,0),3)
            cv2.rectangle(thresh_img,(box[0],box[1]),(box[2],box[3]),(255,0,0),3)
    boxes = np.array(boxes)
    return boxes[max_area[0]]



DATASET_PATH = '/media/altex/DataDrive/Datasets/Im2Latex'
FORMULA_PATH = DATASET_PATH + '/im2latex_formulas.lst.norm'
IMAGES_PATH =  DATASET_PATH + '/formula_images'
TRAIN_PATH =  DATASET_PATH + '/im2latex_validate.lst' 
SAVE_PATH =  DATASET_PATH + '/formula_images_preprocess'

if not os.path.exists(SAVE_PATH):
  os.makedirs(SAVE_PATH) 

data_list = []

with open(TRAIN_PATH,'r',encoding="ISO-8859-1") as f:
  line = f.readline()
  while line:
      data_list.append(line.split('\n')[0])
      line = f.readline() 
                

save_file = open(TRAIN_PATH+'.filtered','w')
print("num images : ",len(data_list))
for idx in range(len(data_list)):
  formula_idx, image_name, render_type = data_list[idx].split(' ')
  img_path = os.path.join(IMAGES_PATH,image_name+'.png')
  formula_idx, image_name, render_type = data_list[idx].split(' ')

  try:
    img = cv2.imread(img_path)
    thresh_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel3 = np.ones([11,11],np.uint8)
    thresh_img = thresh_img.max() - thresh_img
    thresh_img = cv2.dilate(thresh_img,kernel3,iterations=5)
    #thresh_img = cv2.erode(thresh_img,kernel2,iterations=3)
    boxes = bind_objects(img,thresh_img, 100, plot=False)
    img = img[boxes[1]:boxes[3],boxes[0]:boxes[2],:]
    save_img_path = os.path.join(SAVE_PATH,image_name+'.png')
    cv2.imwrite(save_img_path,img)
    save_file.write(data_list[idx])
    save_file.write('\n')


  except Exception as e:
    print(img_path)
