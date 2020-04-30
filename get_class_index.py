import pickle

paths, labels = pickle.load(open('all_paths.pkl','rb'))

brand_idx = {}

for idx, path in enumerate(paths):
    print(path, labels[idx])
    brand = path.split('/')[-2]
    label = labels[idx]
    brand_idx[label] = brand

brands = []
for i in range(258):
    brands.append(brand_idx[i])




pickle.dump(brands, open('brand_idx.pkl','wb'))
