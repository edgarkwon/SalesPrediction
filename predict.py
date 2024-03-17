import pandas as pd
import numpy as np
import pickle

# open csv file
df_brand = pd.read_csv('brand.csv')
df_area = pd.read_csv('area.csv')
df_traing = pd.read_csv('final.csv')

# import model file with pickle
with open('sales_model.pkl', 'rb') as file:
    model = pickle.load(file)

AREA_CODE = 3110056

# get area data based on area code
area_data = df_area[df_area['상권_코드'] == AREA_CODE]

# merge area data with brand data by repeatedly concatenating area data to each brand data
brand_data = df_brand
area_data = pd.concat([area_data]*len(brand_data), ignore_index=True)

# merge brand data and area data
data = pd.concat([brand_data, area_data], axis=1)

# replace cell with FALSE to 0 and TRUE to 1
data = data.replace('FALSE', 0)
data = data.replace('TRUE', 1)

# switch the column order to match the order of the training data
df_new_traing = df_traing.drop('EST_AMT_TOT', axis=1)
data = data[df_new_traing.columns.tolist()]

assert df_new_traing.columns.tolist() == data.columns.tolist()

# predict sales
prediction = model.predict(data)

# add prediction to the data
data['EST_AMT_TOT'] = prediction
data = data[df_traing.columns.tolist()]

# add remaining columns from brand data to data without duplicating the columns
for column in brand_data.columns.tolist():
    if column not in data.columns.tolist():
        data[column] = brand_data[column]

print(data.columns.tolist())

data.to_csv('output.csv', index=False, encoding='utf-8-sig')
