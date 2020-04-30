# -*- coding: utf-8 -*-

from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
#import time
import torch.nn.functional as F
import os
from PIL import Image
import copy, cv2
import sys, shutil, pickle
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from time import time
from collections import Counter
# Data augmentation and normalization for training
# Just normalization for validation


def datatransforms(mean, std, crop_size, resize_size):
    data_transforms = {
      'train': transforms.Compose([
        transforms.RandomResizedCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize( mean, std) #[0.00021798351, 0.00016647576, 0.00016200541], [5.786733e-05, 5.2953397e-05, 4.714992e-05] ) #mean, std) #[0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
      ]),
      'val': transforms.Compose([
        transforms.Resize(resize_size),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std) #[0.00021798351, 0.00016647576, 0.00016200541], [5.786733e-05, 5.2953397e-05, 4.714992e-05]) #mean, std)#[0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
      ]),
      'test': transforms.Compose([
        transforms.Resize(resize_size), #(resize_size, resize_size)),
#        transforms.ColorJitter(contrast=(50,51)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std) #[0.00021798351, 0.00016647576, 0.00016200541], [5.786733e-05, 5.2953397e-05, 4.714992e-05]) #mean, std)#[0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
      ]),

    }
    return data_transforms


data_dir = sys.argv[1]



mean = [0.485, 0.456, 0.406]#torch.tensor([0.485, 0.456, 0.406]) #[0.4616, 0.4006, 0.3602])
std = [0.229, 0.224, 0.225] #torch.tensor([0.229, 0.224, 0.225]) #[0.2287, 0.2160, 0.2085])
#mean = list(mean_std[0].data.cpu().numpy())
#std = list(mean_std[1].data.cpu().numpy())

crop_size = 224

resize_size = 224
data_transforms = datatransforms( mean, std, crop_size, resize_size)

phase = 'test'
BATCH_SIZE=16


class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """
    # override the __getitem__ method. this is the method dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns 
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path

image_datasets = {x: ImageFolderWithPaths(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in [phase]}

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=BATCH_SIZE,
                                             shuffle=True, num_workers=12)
              for x in [phase]}

dataset_sizes = {x: len(image_datasets[x]) for x in [phase]}

class_names = image_datasets[phase].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_model(path):
    model = torch.load(path)
    return model

def load_inputs_outputs(dataloaders):
    for inputs, labels in dataloaders[phase]:
        inputs = inputs.to(device)
        labels = labels.to(device)
    return inputs, labels

def convert_to_numpy(x):
    return x.data.cpu().numpy()


def load_tensor_inputs(paths, data_transforms):
    loader = data_transforms[phase]
    images = [loader(Image.open(path)) for path in paths]
    return torch.stack(images)

def eval_model(model, dataloaders):
    model.eval()   # Set model to evaluate mode
    running_corrects = 0
    output = []
    label = []
    total = 0
    all_times = []

    all_paths = []
    count =0
    start = time()
    for inputs, labels, paths in dataloaders[phase]:
        total+= len(paths)
        all_paths += paths
        inputs = inputs.to(device)

        if False:
           inpts = []
           for inp in inputs:
              mean = inp.mean([1,2])
              mean = mean.view(3, 1, 1).expand_as(inp)
              inp_dup = inp - mean
              inpts.append(inp_dup)

           inputs = torch.stack(inpts)
        
        labels = labels.to(device)
        outputs = model(inputs)
        outputs = F.softmax(outputs, dim=1)
        max_class = torch.argmax(outputs, 1)
        probs, outputs = torch.max(outputs, 1)
        probs_np = convert_to_numpy(probs)
        max_class = convert_to_numpy(max_class)
        outputs_np = convert_to_numpy(outputs)
        labels_np = convert_to_numpy(labels)
        output += (list(outputs_np))
        label += (list(labels_np))
        running_corrects += np.sum(outputs_np == labels_np)
        count = count +1
        all_times.append(time()-start)
        start = time()
        sys.stdout.write('count: {:d}/{:d}, average time:{:f} \r' \
                             .format(count*BATCH_SIZE, len(dataloaders[phase])*BATCH_SIZE, np.mean(np.array(all_times))/BATCH_SIZE ))
        sys.stdout.flush()
    accuracy = running_corrects*1.0/dataset_sizes[phase]
    print("\n")
#    print(confusion_matrix(label, output))
    return accuracy, label, output, all_paths
    
def load_inception_model(model_path, num_classes):

    model_ft = models.resnet50(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs,num_classes)
    model = model_ft.to(device)
    checkpoint = load_model(model_path)
    model.load_state_dict(checkpoint['state_dict'])
    return model

if __name__=="__main__":
    model_path = sys.argv[2]
    num_classes = int(sys.argv[3])
    output_dir = sys.argv[4]
    if not  os.path.exists(output_dir):
        os.makedirs(output_dir)
    print(model_path)
    model = load_inception_model(model_path, num_classes)    
    
    since = time()
    accuracy, labels, outputs, all_paths = eval_model(model, dataloaders)

    pickle.dump([all_paths,labels],  open('all_paths.pkl', 'wb'))
#    print(accuracy)
   # c = Counter(labels)
   # out = []
   # lbl = []
   # for idx, label in enumerate(labels):
   #     if c[label] != 1:
   #         out.append(outputs[idx])
   #         lbl.append(label)
    
   # outputs = out
   # labels = lbl
    accuracy = f1_score(labels, outputs, average='macro')
    print("f1 score:", accuracy)
    #print(confusion_matrix(label, output))
    #print(classification_report(label, output))
    last = time()
    total_time = last-since
    print("total time taken to process;", total_time, "per image:", total_time*1.0/len(outputs))
    pickle.dump([accuracy, labels, outputs],open(os.path.join(output_dir, os.path.basename(model_path)[:-8]+'_'+str(crop_size)+'_'+str(resize_size)+'_accuracy.pkl'),'wb'))
