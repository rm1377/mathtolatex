import numpy as np 

import torch 
from torch import nn
from model import *
import torchvision 
import matplotlib.pyplot as plt 
import glob 
import pandas as pd
import json 
import os 
import random 
import cv2 
from tqdm import tqdm 
from utils import score_files
from data_utils import Im2LatexDataset,Resize, LatexVocab,Normalize
from pynvml import *
nvmlInit()
h = nvmlDeviceGetHandleByIndex(0)
info = nvmlDeviceGetMemoryInfo(h)
print(f'total    : {info.total}')
print(f'free     : {info.free}')
print(f'used     : {info.used}')

DATASET_PATH = '/media/altex/DataDrive/Datasets/Im2Latex'
FORMULA_PATH = DATASET_PATH + '/im2latex_formulas.lst.norm'
IMAGES_PATH =  DATASET_PATH + '/formula_images_preprocess'
VOCAB_PATH = DATASET_PATH + '/latex.vocab'
TRAIN_PATH =  DATASET_PATH + '/im2latex_train.lst.filtered' 
VAL_PATH = DATASET_PATH + '/im2latex_validate.lst.filtered' 
TEST_PATH = DATASET_PATH + '/im2latex_test.lst.filtered' 

MODEL_SAVE_PATH = 'ckpt-modelV0_1.pt'

EPOCHS = 200
BATCH_SIZE = 8
init_lr = 1e-4


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    formulas = [] 
    with open(FORMULA_PATH,'r',encoding="ISO-8859-1") as f:
        line = f.readline()
        while line:
            formulas.append(line.split('\n')[0])
            
            line = f.readline() 
    formulas = np.array(formulas)

    train_data = [] 

    with open(TRAIN_PATH,'r',encoding="ISO-8859-1") as f:
        line = f.readline()
        while line:
            train_data.append(line.split('\n')[0])
            
            line = f.readline() 
    train_data = np.array(train_data)
    print(train_data.shape)
    train_data[1]

    ds_train = Im2LatexDataset(IMAGES_PATH,TRAIN_PATH,FORMULA_PATH,VOCAB_PATH,MAX_LEN,transforms=[Resize([IMG_HSIZE,IMG_WSIZE]),Normalize()]) # create dataset
    train_loader = torch.utils.data.DataLoader(ds_train,batch_size=BATCH_SIZE,shuffle=True,drop_last=True,num_workers=8) # creat dataloader

    ds_val = Im2LatexDataset(IMAGES_PATH,VAL_PATH,FORMULA_PATH,VOCAB_PATH,MAX_LEN,transforms=[Resize([IMG_HSIZE,IMG_WSIZE]),Normalize()]) # create dataset
    val_loader = torch.utils.data.DataLoader(ds_val,batch_size=BATCH_SIZE,shuffle=False,drop_last=True,num_workers=2) # creat dataloader

    images, labels = next(iter(train_loader))
    # grid_imgs = torchvision.utils.make_grid(images,nrow=4) 
    # grid_imgs.size()
    # np_grid = grid_imgs.permute([1,2,0]).numpy()
    # plt.figure(figsize=(20,20))
    # plt.axis('off') 
    # print([cls for cls in labels[:2]])
    # plt.imshow(np_grid) 
    # plt.show()


    decoder_model = DecoderRNN(HiddenSize,len(ds_train.vocab), EmbedSize, AttnSize).to(device) 
    encoder_model = CNNModel(HiddenSize).to(device)



    try:
        ckpt = torch.load(MODEL_SAVE_PATH.replace('.pt','-best.pt'))
        encoder_model.load_state_dict(ckpt['encoder_model'])
        decoder_model.load_state_dict(ckpt['decoder_model'])
        print("load model from checkpoint")
    except Exception as e:
        print('There is no checkpoint in ',MODEL_SAVE_PATH)
        ckpt = {'epoch': 0,
                'best_fitness': 0,
                'encoder_model': None,
                'decoder_model': None,
                'encoder_optimizer': None ,
                'decoder_optimizer': None ,
                'train_loss':[],
                'train_accuracy':[],
                'val_loss':[],
                'val_accuracy':[]
        }
    start_epoch = ckpt['epoch'] 
    best_fitness = ckpt['best_fitness']






    encoder_optimizer = torch.optim.Adam(encoder_model.parameters(),init_lr) 
    decoder_optimizer = torch.optim.Adam(decoder_model.parameters(),init_lr) 


    if ckpt['encoder_optimizer'] is not None:
        encoder_optimizer.load_state_dict(ckpt['encoder_optimizer'])
    for g in encoder_optimizer.param_groups:
        g['lr'] = init_lr

    if ckpt['decoder_optimizer'] is not None:
        decoder_optimizer.load_state_dict(ckpt['decoder_optimizer'])
    for g in decoder_optimizer.param_groups:
        g['lr'] = init_lr





    criterion = torch.nn.CrossEntropyLoss() 




    def forward_pass(images, labels):
        features = encoder_model(images)
        features = features.view([BATCH_SIZE,-1,HiddenSize])

        #states = decoder_model.initHidden()
        mean_features = torch.mean(features,1)
        states = [mean_features,mean_features]
        total_loss = 0
        predictions = []
        for t in range(1,labels.size(1)):
            outputs, states = decoder_model(labels[:,t-1], features,states) 
            mask = labels[:,t:t+1]!=ds_train.vocab.token2id['_PAD_']
            outputs_ = mask*outputs
            labels_t = labels[:,t] * mask[:,0]
            loss = criterion(outputs, labels_t)
            total_loss += loss 
            predict = torch.unsqueeze(torch.argmax(outputs,-1), -1)
            predictions.append(predict)
        predictions = torch.cat(predictions,-1) 
        total_loss = total_loss/labels.size(1)
        return predictions, total_loss 





    def evaluate(encoder_model,decoder_model,testloader):
        encoder_model.eval()
        decoder_model.eval()
        gt_path = 'gt_data.txt'
        pred_path = 'predictions.txt'
        prediction_file = open(pred_path,'w')
        gt_file = open(gt_path,'w')
        with torch.no_grad():
            data_loader = tqdm(testloader)
            mean_loss = 0
            mean_acc = 0
            
            for data in data_loader:
                images, labels = data
                labels = labels[1]
                images = images.to(torch.float).to(device)
                labels = labels.to(device)
                outputs, loss = forward_pass(images, labels)
                acc = (outputs == labels[:,1:]).sum()/(labels.size(0)*outputs.size(1))
                outputs = outputs.cpu().numpy()
                for i in range(outputs.shape[0]):
                    seq = list(outputs[i])
                    pred_text = ds_train.vocab.seq2text(seq)
                    prediction_file.write(pred_text)
                    prediction_file.write('\n') 
                    gt_seq = list(labels[i].cpu().numpy())
                    gt_text = ds_train.vocab.seq2text(gt_seq)
                    gt_file.write(gt_text)
                    gt_file.write('\n')

                mean_loss += loss.item()
                mean_acc += acc.item()

        prediction_file.close()
        gt_file.close()
        metrics = score_files(gt_path, pred_path)
        mean_loss /= len(testloader)
        mean_acc /= len(testloader)
        return mean_loss, mean_acc, metrics



    pationt = 0 
    print('training .... ')
    for epoch in range(start_epoch, EPOCHS):
        encoder_model.train()
        decoder_model.train()
        mean_loss = 0 
        mean_accuracy = 0 
        pbar = tqdm(train_loader)
        for itr,(images, labels) in enumerate(pbar):
            
            labels = labels[1]
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            encoder_model.zero_grad()
            decoder_model.zero_grad() 
            images = images.to(torch.float).to(device)
            labels = labels.to(device)
            outputs, loss = forward_pass(images, labels)
            acc = (outputs == labels[:,1:]).sum()/(labels.size(0)*outputs.size(1))

            loss.backward() 
            encoder_optimizer.step() 
            decoder_optimizer.step() 
            loss_value = loss.item() 
            mean_loss += 1/(itr+1)*(loss_value - mean_loss)
            mean_accuracy += 1/(itr+1)*(acc.item() - mean_accuracy)

            s = 'EPOCH[%d/%d] loss=%2.4f - accuracy=%2.4f'%(epoch,EPOCHS,mean_loss,mean_accuracy)

            pbar.set_description(s)
        ckpt['train_loss'].append(mean_loss)
        ckpt['train_accuracy'].append(mean_accuracy)

        val_loss, val_accuracy, metrics = evaluate(encoder_model, decoder_model, val_loader)
        ckpt['val_loss'].append(val_loss)
        ckpt['val_accuracy'].append(val_accuracy)
        print('\n val loss = ', val_loss,' accuracy = ',val_accuracy)
        print(metrics)

        ckpt['epoch']= epoch+1
        ckpt['best_fitness']= val_accuracy
        ckpt['encoder_model'] = encoder_model.state_dict()
        ckpt['decoder_model'] = decoder_model.state_dict()
        ckpt['encoder_optimizer']= None if epoch==EPOCHS-1 else encoder_optimizer.state_dict()
        ckpt['decoder_optimizer']= None if epoch==EPOCHS-1 else decoder_optimizer.state_dict()

        torch.save(ckpt, MODEL_SAVE_PATH.replace('.pt','-last.pt'))
        if pationt>10 and False:
            print("early stopping ... ")
            break
        if metrics['BLEU-4']>best_fitness:
            pationt = 0
            best_fitness = metrics['BLEU-4'] 
            ckpt['best_fitness'] = best_fitness
            print("save best at accuracy = ",best_fitness)
            torch.save(ckpt, MODEL_SAVE_PATH.replace('.pt','-best.pt'))
        else:
            pationt += 1



if __name__ == '__main__':
    main()