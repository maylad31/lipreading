#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2020 Imperial College London (Pingchuan Ma)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

""" TCN for lipreading"""

import os
import argparse
import numpy as np
from tqdm import tqdm
from lipreading.preprocess import *
from dataset import MyDataset, pad_packed_collate

import cv2
def get_preprocessing_pipelines():
    # -- preprocess for the video stream
    preprocessing = {}
    # -- LRW config
    crop_size = (88, 88)
    (mean, std) = (0.421, 0.165)
    preprocessing['train'] = Compose([
                                Normalize( 0.0,255.0 ),
                                RandomCrop(crop_size),
                                HorizontalFlip(0.5),
                                Normalize(mean, std) ])

    preprocessing['val'] = Compose([
                                Normalize( 0.0,255.0 ),
                                CenterCrop(crop_size),
                                Normalize(mean, std) ])

    preprocessing['test'] = preprocessing['val']

    return preprocessing
    
    

import torch
import torch.nn.functional as F

from lipreading.utils import load_json, save2npz
from lipreading.model import Lipreading
from dataloaders import get_data_loaders, get_preprocessing_pipelines


def load_args(default_config=None):
    parser = argparse.ArgumentParser(description='Pytorch Lipreading ')
    # -- dataset config
    parser.add_argument('--dataset', default='lrw', help='dataset selection')
    parser.add_argument('--num-classes', type=int, default=500, help='Number of classes')
    # -- directory
    parser.add_argument('--data-dir', default='./cropped', help='Loaded data directory')
    parser.add_argument('--label-path', type=str, default='./labels/500WordsSortedList.txt', help='Path to txt file with labels')
    parser.add_argument('--annonation-direc', default=None, help='Loaded data directory')
    # -- model config
    parser.add_argument('--backbone-type', type=str, default='resnet', choices=['resnet', 'shufflenet'], help='Architecture used for backbone')
    parser.add_argument('--relu-type', type = str, default = 'relu', choices = ['relu','prelu'], help = 'what relu to use' )
    parser.add_argument('--width-mult', type=float, default=1.0, help='Width multiplier for mobilenets and shufflenets')
    # -- TCN config
    parser.add_argument('--tcn-kernel-size', type=int, nargs="+", help='Kernel to be used for the TCN module')
    parser.add_argument('--tcn-num-layers', type=int, default=4, help='Number of layers on the TCN module')
    parser.add_argument('--tcn-dropout', type=float, default=0.2, help='Dropout value for the TCN module')
    parser.add_argument('--tcn-dwpw', default=False, action='store_true', help='If True, use the depthwise seperable convolution in TCN architecture')
    parser.add_argument('--tcn-width-mult', type=int, default=1, help='TCN width multiplier')
    # -- train
    parser.add_argument('--batch-size', type=int, default=1, help='Mini-batch size')
    # -- test
    parser.add_argument('--model-path', type=str, default='lrw_snv1x_tcn2x.pth.tar', help='Pretrained model pathname')
    # -- feature extractor
    parser.add_argument('--extract-feats', default=False, action='store_true', help='Feature extractor')
    parser.add_argument('--mouth-patch-path', type=str, default=None, help='Path to the mouth ROIs, assuming the file is saved as numpy.array')
    parser.add_argument('--mouth-embedding-out-path', type=str, default=None, help='Save mouth embeddings to a specificed path')
    # -- json pathname
    parser.add_argument('--config-path', type=str, default='lrw_snv1x_tcn2x.json', help='Model configuration with json format')

    args = parser.parse_args()
    return args


args = load_args()


def extract_feats(model):
    """
    :rtype: FloatTensor
    """
    model.eval()
    preprocessing_func = get_preprocessing_pipelines()['test']
    data = preprocessing_func(np.load(args.mouth_patch_path)['data'])  # data: TxHxW
    return model(torch.FloatTensor(data)[None, None, :, :, :].cuda(), lengths=[data.shape[0]])

actualword="building"
def evaluate(model,cap,out):
    preprocessing = get_preprocessing_pipelines()
    model.eval()
    data=np.load('cropped/crop.npz')['data']
    print(data.shape)
    data = preprocessing['test'](data)
    data, lengths, labels_np, = (np.array([data]), data.shape[0], "om")
    input = torch.FloatTensor(data)
    lengths = [input.size(1)]
    running_corrects = 0.
    f=open("500WordsSortedList.txt")
    l2=f.readlines()
    #print(l2)
    lt=[]
    with torch.no_grad():
        if True:
            
            logits = model(input.unsqueeze(1).cuda(), lengths=lengths)
            preds = F.softmax(logits, dim=1)
            preds=preds.cpu().numpy()
            preds=list(preds[0])
          
            le=len(preds)
            
            for i in range(le):
                if preds[i]>0.0001:
                    lt.append((preds[i],l2[i]))
    lt=sorted(lt,reverse=True)
    print(lt)
    st="1"+''.join(c for c in str(lt[0]) if c.isalpha()).rstrip("n")+" 2"+''.join(c for c in str(lt[1]) if c.isalpha()).rstrip("n")+" 3"+''.join(c for c in str(lt[2]) if c.isalpha()).rstrip("n")+" "
    for i in range(len(lt)):
        if actualword in str(lt[i][1]).lower():
            st=st+str(i+1)+''.join(c for c in str(lt[i]) if c.isalpha()).rstrip("n")
            
            
    while True:
        _,img=cap.read()
        #print("read")
        cv2.putText(img,st, (20,80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255))
        out.write(img)
        

def get_model():
    args_loaded = load_json( args.config_path)
    args.backbone_type = args_loaded['backbone_type']
    args.width_mult = args_loaded['width_mult']
    args.relu_type = args_loaded['relu_type']
    tcn_options = { 'num_layers': args_loaded['tcn_num_layers'],
                    'kernel_size': args_loaded['tcn_kernel_size'],
                    'dropout': args_loaded['tcn_dropout'],
                    'dwpw': args_loaded['tcn_dwpw'],
                    'width_mult': args_loaded['tcn_width_mult'],
                  }

    return Lipreading( num_classes=args.num_classes,
                       tcn_options=tcn_options,
                       backbone_type=args.backbone_type,
                       relu_type=args.relu_type,
                       width_mult=args.width_mult,
                       extract_feats=args.extract_feats).cuda()


def main():
    assert args.config_path.endswith('.json') and os.path.isfile(args.config_path), \
        "'.json' config path does not exist. Path input: {}".format(args.config_path)
    assert args.model_path.endswith('.tar') and os.path.isfile(args.model_path), \
        "'.tar' model path does not exist. Path input: {}".format(args.model_path)

    model = get_model()
    

    model.load_state_dict( torch.load(args.model_path)["model_state_dict"], strict=True)
    cap=cv2.VideoCapture("test.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, 20.0, (1280,720))

    if args.mouth_patch_path:
        save2npz( args.mouth_embedding_out_path, data = extract_feats(model).cpu().detach().numpy())
        return   
    # -- get dataset iterators
    
    evaluate(model,cap,out)
    cap.release()
    out.release()

if __name__ == '__main__':
    main()
