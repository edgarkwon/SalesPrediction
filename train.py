import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# open csv file
df = pd.read_csv('/home/gdsc_kaist/SalesPrediction/final.csv') 
df.dropna(inplace=True)

# find X and Y
X=df.drop('EST_AMT_TOT', axis=1)
Y=df['EST_AMT_TOT']

# replace cell with FALSE to 0 and TRUE to 1
X = X.replace('FALSE', 0)
X = X.replace('TRUE', 1)


# check for missing value
for column in X.columns:
    print(f'{column} has {X[column].isnull().sum()} missing value')
    

# split dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.2, random_state=42)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# train model
model = LinearRegression()
model.fit(X_train, y_train)

# test model
accuracy = model.score(X_test, y_test)
print(accuracy)

# save model
with open('/home/gdsc_kaist/SalesPrediction/sales_model.pkl', 'wb') as file:
    pickle.dump(model, file)



