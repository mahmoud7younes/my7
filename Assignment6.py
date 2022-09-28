import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import category_encoders as cs
from sklearn.preprocessing import MinMaxScaler, StandardScaler

df = pd.read_csv('spaceship-Titanic.csv')
print(df.head())

HomePlanet = df['HomePlanet']
print(HomePlanet)
PassengerId = df['PassengerId']
print(PassengerId)
CryoSleep = df['CryoSleep']
print(CryoSleep)
Cabin = df['Cabin']
print(Cabin)
Destination = df['Destination']
print(Destination)
Age = df['Age']
print(Age)
RoomService = df['RoomService']
print(RoomService)
FoodCourt = df['FoodCourt']
print(FoodCourt)
ShoppingMall = df['ShoppingMall']
print(ShoppingMall)
Spa = df['Spa']
print(Spa)
VRDeck = df['VRDeck']
print(VRDeck)
CryoSleep = df['CryoSleep']
print(CryoSleep)

df.drop(columns=['HomePlanet'], inplace=True)
df['HomePlanet'] = HomePlanet

print(df.describe())
print(df.info())
print(df.isnull().sum())
print(df.isnull().mean() * 100)

print(df.drop(columns=['Cabin'], inplace=True))
print(df['Age'].fillna(df['Age'].mean(),inplace=True))
print(df['Age'].mode()[0])

df['Destination'].describe()
df.duplicated().any()
# df = df.append(df.iloc[0])
print(df.info())

print(sns.boxplot(df['Age']))

print(df['Age'].describe())

max_threshold = df['Age'].quantile(.98)
max_threshold

min_threshold = df['Age'].quantile(.1)
min_threshold

new_df = df[(df['Age'] < max_threshold) & (df['Age'] > min_threshold)]

print(sns.boxplot(new_df['Age']))

mapping = {"Europa":1, "Earth":0}
df['HomePlanet'] =df['HomePlanet'].map(mapping)
print(df.head())

df.loc[ df['PassengerId'] == 2_01, 'Age'] = np.nan
df.dropna(inplace = True)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(df, df['Transported'])
print(x_train)
print(x_test)
print(y_train)
print(y_test)

g= sns.FacetGrid(df, col= 'Transported')
g.map(plt.hist, 'Age', bins = 20)

g= sns.FacetGrid(df, col= 'Transported')
g.map(plt.hist, 'HomePlanet', bins = 20)





