import sys
import pickle

a = open(sys.argv[1], 'r').readlines()

classes = []

for l in a:
    splits = l.split('/')
    cls = splits[-2]
    image = splits[-1]
    classes.append(cls)

classes = list(set(classes))
    
label_dict = {}
for l in a:
    splits = l.split('/')
    cls = splits[-2]
    image = splits[-1]
    label_dict[image] = classes.index(cls)


pickle.dump(label_dict, open("labels.pkl", 'wb'))

