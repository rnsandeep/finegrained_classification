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

def eval_model_variant(model, model2, dataloaders, brand_to_upc):
    model.eval()   # Set model to evaluate mode
    model2.eval()
    running_corrects = 0
    output = []
    label = []
    total = 0
    all_times = []

    all_paths = []
    count =0
    start = time()
    inputs_all = []
    for inputs, labels, paths in dataloaders[phase]:
        total+= len(paths)
        all_paths += paths
        inputs_all += inputs
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs_brands = model(inputs)
        outputs_variants = model2(inputs)
        outputs_brands = F.softmax(outputs_brands, dim=1)
        brands_probs, outputs_brands_max = torch.max(outputs_brands, 1)
        outputs_variants = F.softmax(outputs_variants, dim=1)
        variants_probs, outputs_variants_max = torch.max(outputs_variants, 1)

        brand_label = list(convert_to_numpy(outputs_brands_max))
        variants_label = list(convert_to_numpy(outputs_variants_max))
        changed_upcs = []
        for idx, ot in enumerate(brand_label):
            print(variants_probs[idx])
            if variants_probs[idx] < 0.90:
               upcs = brand_to_upc[ot]
               out = outputs_variants[idx][upcs]
               changed_upc = upcs[torch.argmax(out)]
               changed_upcs.append(changed_upc)
            else:
               changed_upcs.append(variants_label[idx])

        outputs_np = np.array(changed_upcs)
        labels_np = convert_to_numpy(labels)
        output += list(outputs_np)
        label += list(labels_np)
        running_corrects += np.sum(outputs_np == labels_np)
        count = count +1
        all_times.append(time()-start)
        start = time()
        sys.stdout.write('count: {:d}/{:d}, average time:{:f} \r' \
                             .format(count*BATCH_SIZE, len(dataloaders[phase])*BATCH_SIZE, np.mean(np.array(all_times))/BATCH_SIZE ))
        sys.stdout.flush()
    accuracy = running_corrects*1.0/dataset_sizes[phase]
    print("\n")
    return accuracy, label, output
    
def load_resnet50_model(model_path, num_classes):

    model_ft = models.resnet50(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs,num_classes)
    model = model_ft.to(device)
    checkpoint = load_model(model_path)
    model.load_state_dict(checkpoint['state_dict'])
    return model

if __name__=="__main__":

    brand_model_path = sys.argv[2]
    brand_variant_model_path = sys.argv[3]
    brand_num_classes = int(sys.argv[4])
    brand_variant_num_classes = int(sys.argv[5])


    output_dir = sys.argv[6]
    if not  os.path.exists(output_dir):
        os.makedirs(output_dir)

    brand_to_upc = pickle.load(open('brand_to_upc.pkl','rb'))

    brand_model = load_resnet50_model(brand_model_path, brand_num_classes)
    brand_variant_model = load_resnet50_model(brand_variant_model_path, brand_variant_num_classes)
    
    since = time()
    accuracy, labels, outputs = eval_model_variant(brand_model, brand_variant_model,  dataloaders, brand_to_upc)

    accuracy = f1_score(labels, outputs, average='macro')
    print("f1 score:", accuracy)
    #print(confusion_matrix(label, output))
    #print(classification_report(label, output))
    last = time()
    total_time = last-since
    print("total time taken to process;", total_time, "per image:", total_time*1.0/len(outputs))
    pickle.dump([accuracy, labels, outputs],open(os.path.join(output_dir, os.path.basename(brand_variant_model_path)[:-8]+'_'+str(crop_size)+'_'+str(resize_size)+'_accuracy.pkl'),'wb'))
