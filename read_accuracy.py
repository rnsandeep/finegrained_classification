import pickle
import sys, os
from os import listdir
from os.path import isfile, join
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from collections import Counter
import pandas as pd

mypath = sys.argv[1]
onlyfiles = [os.path.join(mypath, f )for f in listdir(mypath) if isfile(join(mypath, f))]
max_acc = 0
mean_acc = 0
def transpose(labels):
    labels = np.array(labels)
    labels[np.where(labels==0)] = 2
    labels[np.where(labels==1)] = 0
    labels[np.where(labels==2)] = 1

    return list(labels)

def calculatePrecisionRecallAccuracy(labels, outputs):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for label, output in zip(labels, outputs):
        if label==output and label ==0:
            tn = tn+1
        elif label==output and label==1:
            tp = tp+1
        elif label!=output and label == 0:
            fp = fp+1
        else:
            fn = fn +1
    precision = tp*1.0/(tp+fp)
    recall = tp*1.0/(tp+fn)
    accuracy = (tp+tn)*1.0/(tp+fp+fn+tn)
#    print("tp:", tp, "fp:", fp, "fn:", fn)
    return  precision, recall, accuracy

onlyfiles.sort(key=lambda x: (int(os.path.basename(x).split('_')[0]), x) )

for f in onlyfiles:
    basename = os.path.basename(f)
    splits = basename.split('_')
    epoch = splits[0]
    scale = splits[2]
    resize = splits[3]
    accuracy, labels, outputs = pickle.load(open(f,'rb'))
#    accuracy = f1_score(labels, outputs, average='macro')
#    print("accuracy:", accuracy, f)

#    c = Counter(labels)
#    out = []
#    lbl = []
#    for idx, label in enumerate(labels):
#        if c[label] != 1:
#            out.append(outputs[idx])
#            lbl.append(label)

#    outputs = out
#    labels = lbl

    accuracy = f1_score(labels, outputs, average='macro')
    print("accuracy:", accuracy, f)

    y_true = pd.Series(labels, name="Actual")
    y_pred = pd.Series(outputs, name="Predicted")
    df_confusion = pd.crosstab(y_true, y_pred, margins=True)
                #confusion_matrix = ConfusionMatrix(y_true, y_pred)
    df_confusion.to_csv("confusion.csv")


    #outputs = transpose(outputs)
#    precision, recall, accuracy = calculatePrecisionRecallAccuracy(labels, outputs)
#    print(classification_report(labels, outputs))
#    matrix = confusion_matrix(outputs, labels)
#    print(matrix)
    if max_acc < accuracy:
      max_acc = accuracy
    mean_acc += accuracy
#    print("precision:%.3f"%precision, "recall:%.3f"%recall,"accuracy:%.3f"%accuracy, "epoch:", epoch, "scale;", scale, "resize:", resize)

print("maximum accuracy:", max_acc)
print("Mean_accuracy:", mean_acc/len(onlyfiles))
