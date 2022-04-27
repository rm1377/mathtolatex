
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




class LatexVocab(object):
    def __init__(self, vocab_path):
        token2id = {}
        id2token = {} 
        with open(vocab_path, 'r') as f:
            line = f.readline()
            k = 0
            while line:
                token = line.split('\n')[0]
                token2id[token] = k 
                id2token[k] = token 
                line = f.readline() 
                k += 1 
        self.id2token = id2token 
        self.token2id = token2id 


    def text2seq(self, text, max_size=None):
        text = text.split(' ')
        seq = []
        for ch in text:
            try:
                seq.append(self.token2id[ch])
            except:
                seq.append(self.token2id['_UNK_'])
        seq = [self.token2id['_START_']] + seq + [self.token2id['_END_']]
        if max_size is not None:
            temp = [self.token2id['_PAD_']]*max_size
            temp[:len(seq)] = seq 
            seq = temp
            seq = seq[:max_size]
        return seq 
    def seq2text(self,seq):
        seq = [self.id2token[id] for id in seq]
        text = ''
        for s in seq:
            if s=='_START_':
              continue
            if s=='_END_' or s=='_PAD_':
              break
            text += s + ' '
        return text[:-1]

    def __len__(self):
        return len(self.id2token)


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
    for c in cnts:
        if cv2.contourArea(c) < minArea:           #ignore contours that are smaller than this area
            continue

        x,y,w,h   = cv2.boundingRect(c)
        

        box = np.array([x,y,x+w,y+h],int)
        boxes.append(box) 
        if plot:
            cv2.rectangle(frame,(box[0],box[1]),(box[2],box[3]),(255,0,0),3)
            cv2.rectangle(thresh_img,(box[0],box[1]),(box[2],box[3]),(255,0,0),3)
    boxes = np.array(boxes)
    return boxes



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


class Im2LatexDataset(torch.utils.data.Dataset):
    def __init__(self, images_dir,data_path, formula_path=None, vocab_path=None, max_size=None,transforms=None, num_caches=1000):
        super(Im2LatexDataset,self).__init__()
        self.root = images_dir
        
        self.data_list = []
        self.formulas = None
        self.vocab = None
        formulas = []
        if formula_path is not None:
            with open(formula_path,'r',encoding="ISO-8859-1") as f:
                line = f.readline()
                while line:
                    formulas.append(line.split('\n')[0])
                    
                    line = f.readline() 
            self.formulas = np.array(formulas)

        if vocab_path is not None:
            self.vocab = LatexVocab(vocab_path)
        self.max_size = max_size

        with open(data_path,'r',encoding="ISO-8859-1") as f:
            line = f.readline()
            while line:
                self.data_list.append(line.split('\n')[0])
                line = f.readline() 
                
        self.data_list = np.array(self.data_list)
        self.cached_images = [None]*self.data_list.shape[0]
        self.cached_labels = [None]*self.data_list.shape[0]
        
        self.num_caches_limits = num_caches
        self.cached = 0
        self.transforms = transforms 

    
    
    def __len__(self):
        return self.data_list.shape[0]
    
    def bind_objects(self,frame,thresh_img, minArea, plot=False):
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
        for c in cnts:
            if cv2.contourArea(c) < minArea:           #ignore contours that are smaller than this area
                continue

            x,y,w,h   = cv2.boundingRect(c)


            box = np.array([x,y,x+w,y+h],int)
            boxes.append(box) 
            if plot:
                cv2.rectangle(frame,(box[0],box[1]),(box[2],box[3]),(255,0,0),3)
                cv2.rectangle(thresh_img,(box[0],box[1]),(box[2],box[3]),(255,0,0),3)
        boxes = np.array(boxes)
        return boxes
    

    def __getitem__(self,idx):

        if self.cached_images[idx] is not None:
          img = self.cached_images[idx] 
          formula_txt, formula_seq = self.cached_labels[idx]
          
        else:
          formula_idx, image_name, render_type = self.data_list[idx].split(' ')
          img_path = os.path.join(self.root,image_name+'.png')
          formula_idx, image_name, render_type = self.data_list[idx].split(' ')
          formula_txt = self.formulas[int(formula_idx)]
          formula_seq = np.array(self.vocab.text2seq(formula_txt,self.max_size))
          img = cv2.imread(img_path)
        #   try:
        #     img = cv2.imread(img_path)
        #     thresh_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #     kernel3 = np.ones([11,11],np.uint8)
        #     thresh_img = thresh_img.max() - thresh_img
        #     thresh_img = cv2.dilate(thresh_img,kernel3,iterations=5)
        #     #thresh_img = cv2.erode(thresh_img,kernel2,iterations=3)
        #     boxes = self.bind_objects(img,thresh_img, 100, plot=False)[0]
        #     img = img[boxes[1]:boxes[3],boxes[0]:boxes[2],:]
        #     if img.shape[0]:
        #       pass
        #   except:
        #     img = np.zeros((100,100,3),dtype=np.uint8)

          
          if self.cached<self.num_caches_limits:
            self.cached_images[idx] = img 
            self.cached_labels[idx] = (formula_txt, formula_seq)
            self.cached += 1 
        if self.transforms is not None:
          for trf in self.transforms:
            img = trf(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img.reshape([img.shape[0], img.shape[1],1])
        img = img.transpose(2, 0, 1)  # 
        img = np.ascontiguousarray(img)
        img = np.float32(img)
        img = torch.from_numpy(img).to(torch.float32)

        formula_seq = torch.from_numpy(formula_seq)
        return img, (formula_txt,formula_seq) 



class Normalize(object):
  def __init__(self,mean=[0.485, 0.456, 0.406], scale=[0.229, 0.224, 0.225]):
    self.mean = np.array(mean) 
    self.scale = np.array(scale) 
  def __call__(self,x):

    x = np.float32(x)/255 
    return x 

class RandomBGR2GRAY(object):
  def __init__(self,p=0.5):
    self.p = p
  def __call__(self,x):
    if np.random.uniform(0.,1.0)<self.p:
        x = cv2.cvtColor(x,cv2.COLOR_BGR2GRAY)
        x = cv2.cvtColor(x,cv2.COLOR_GRAY2BGR)
    return x 




class RandomCrop(object):
  def __init__(self,p=0.5):
    self.p = p
  def __call__(self,x):
    if np.random.uniform(0,1)<self.p:
      return x 
    k = np.random.randint(0,4) 
    h, w, _ = x.shape
    if k==0:
      xmin = np.random.randint(0,w//4)
      ymin = np.random.randint(0,h//4)
      xmax = w 
      ymax = h 
    elif k==1:
      xmin = 0 
      ymin = np.random.randint(0,h//4)
      xmax = np.random.randint(w//4,3*w//4) 
      ymax = h
    elif k==2:
      xmin = np.random.randint(0,w//4)
      ymin = 0
      xmax = w 
      ymax = np.random.randint(h//4,3*h//4) 
    elif k==3:
      xmin = 0
      ymin = 0
      xmax = np.random.randint(w//4,3*w//4) 
      ymax = np.random.randint(h//4,3*h//4)
    cropped = x[ymin:ymax,xmin:xmax,:]
      

    return cropped



