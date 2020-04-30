import pickle

upc_dict = pickle.load(open('upc_dict.pkl','rb'))

brand_idx = pickle.load(open('brand_idx.pkl', 'rb'))
brand_variant_idx = pickle.load(open('brand_variant_idx.pkl','rb'))

brand_to_upc = {}
for upc in upc_dict:
    print(upc, upc_dict[upc])
    brand_variant = brand_variant_idx.index(upc)
    brand = brand_idx.index(upc_dict[upc].replace(' ', '_'))
    print(brand, brand_variant)
    if brand not in brand_to_upc:
        brand_to_upc[brand] = [brand_variant]
    else:
        brand_to_upc[brand] += [brand_variant]

upc_to_brand =  {}
for brand in brand_to_upc:
    upcs = brand_to_upc[brand]
    for upc in upcs:
        if upc not in upc_to_brand:
            upc_to_brand[upc] = brand

pickle.dump(upc_to_brand, open('upc_to_brand.pkl','wb'))


for brand in brand_to_upc:
    print(brand_to_upc[brand])

pickle.dump(brand_to_upc, open('brand_to_upc.pkl','wb'))



