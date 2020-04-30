import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import sys, shutil



# Data augmentation and normalization for training
# Just normalization for validation


def datatransforms(crop_size):
    mean = [0.485, 0.456, 0.406] #list(mean_std[0].data.cpu().numpy())
    std = [0.229, 0.224, 0.225] #list(mean_std[1].data.cpu().numpy())
    print("mean and standard deviation:",mean,std)
    data_transforms = {
      'train': transforms.Compose([
        transforms.RandomResizedCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize( mean, std) #[0.00021798351, 0.00016647576, 0.00016200541], [5.786733e-05, 5.2953397e-05, 4.714992e-05] ) #mean, std) #[0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
      ]),
      'val': transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std) #[0.00021798351, 0.00016647576, 0.00016200541], [5.786733e-05, 5.2953397e-05, 4.714992e-05]) #mean, std)#[0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
      ]),
      'test': transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std) #[0.00021798351, 0.00016647576, 0.00016200541], [5.786733e-05, 5.2953397e-05, 4.714992e-05]) #mean, std)#[0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
      ]),

    }
    return data_transforms


data_dir = sys.argv[1] 
output_dir = sys.argv[2]

crop_size = 224

data_transforms = datatransforms(crop_size)

if not  os.path.exists(output_dir):
    os.makedirs(output_dir)

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train',  'test']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32,
                                             shuffle=True, num_workers=16)
              for x in ['train', 'test']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train',  'test']}
class_names = image_datasets['train'].classes
#print(image_datasets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device found:", device)


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

# Get a batch of training data
inputs, classes = next(iter(dataloaders['train']))


######################################################################
# Training the model
# ------------------
#
# Now, let's write a general function to train a model. Here, we will
# illustrate:
#
# -  Scheduling the learning rate
# -  Saving the best model
#
# In the following, parameter ``scheduler`` is an LR scheduler object from
# ``torch.optim.lr_scheduler``.

BATCH_SIZE=32
def train_model(model, criterion, optimizer, scheduler, num_epochs):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train']:#, 'val', 'test']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            count  = 0
            start = time.time()
            all_times = []
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()
                cls_nums = [torch.sum(labels==i) for i in range(len(class_names))]
                w = torch.FloatTensor(cls_nums)
                w = w/torch.sum(w)
                class_weights = 1/torch.FloatTensor(w).to(device)
 
                # forward
                # track history if only in train
                count  = count + 1
                with torch.set_grad_enabled(phase == 'train'):
                    if phase == 'train':
                       outputs = model(inputs)
                    else:
                       outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = nn.CrossEntropyLoss()(outputs, labels)
#                    loss = nn.CrossEntropyLoss(weight=class_weights)(outputs, labels)
#                    loss = criterion(outputs, labels, weight=class_weights)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                sample_time = time.time()-start   
                start = time.time()
                sys.stdout.write('count: {:d}/{:d}, average time:{:f} \r' \
                             .format(count*BATCH_SIZE, len(dataloaders[phase])*BATCH_SIZE, np.mean(np.array(all_times))/BATCH_SIZE ))
                sys.stdout.flush()


                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)


            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        save_checkpoint({
            'epoch': 25+ epoch + 1,
            #'arch': args.arch,
            'state_dict': model_ft.state_dict(),
            'best_prec1': epoch_acc,
            'optimizer' : optimizer_ft.state_dict(),
        }, False, os.path.join(output_dir, str(epoch)+'_checkpoint.pth.tar'))


    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)

    return model, best_acc, epoch



num_classes = int(sys.argv[3]) # no of classes
no_of_epochs = int(sys.argv[4]) # no of epochs to train



model_ft = models.resnet50(pretrained=True)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs,num_classes)

model_ft = model_ft.to(device)

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=60, gamma=0.1)

criterion = ''
model_ft, best_acc, epoch = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=no_of_epochs)

