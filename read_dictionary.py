import pandas as pd
import sys
import os
import shutil
import pickle

dict = pd.read_csv("dictionary.csv")

print(dict.columns)
print(dict.head(10))



brands = list(dict['BRAND'].values)

brands_sub = list(dict['BRAND VARIANT'].values)

UPC = list(dict["UPC"].values)

upc_dict = {}
for idx in range(len(brands)):
    upc_dict[str(UPC[idx])] = brands[idx]

pickle.dump(upc_dict, open('upc_dict.pkl','wb'))

a = open(sys.argv[1], 'r').readlines()

output_path = 'output_brand'
if not  os.path.exists(output_path):
   os.makedirs(output_path)

datapath = sys.argv[2]

if False:
  for l in a:
    splits = l.split('/')
    cls = splits[1]
    image = splits[2]
    brand = upc_dict[cls].replace(' ', '_')

    if not os.path.exists(os.path.join(output_path, brand)):
        os.makedirs(os.path.join(output_path, brand))
 
    image_path = os.path.join(datapath, splits[1], image.strip())

    shutil.copy(image_path, os.path.join(output_path, brand))


    
    
       
    


