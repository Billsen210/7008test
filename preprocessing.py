import pandas as pd

data = pd.read_csv('clean_data.csv')
data = data[data['age'] >= 1]
data = data[data['gender'] != 'Other']
data['gender'].replace({'Female': 0, 'Male': 1}, inplace=True)
data_encoded = pd.get_dummies(data, columns=['smoking_history'], prefix='smoking')

data_encoded.to_csv('/opt/data/df.csv', index=False)
