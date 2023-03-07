import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import Dataset, DataLoader
df = pd.read_csv(r'D:\CNG514_Spring_2021\Project_22\heart.csv')

print(df.shape)

print(df.head())# preprocessing
print(df.isnull().sum())
print(df.max())
print(df.min())
df['Oldpeak']= df['Oldpeak'].clip(lower=0, upper=6.2)
print(df.dtypes)
print(df.describe())
print(df.Sex.value_counts())
print(df['HeartDisease'].value_counts())
print(df.ChestPainType.value_counts())

# %% Analyzing

numerical = ['Age','RestingBP','Cholesterol','MaxHR','Oldpeak']
categorical = ['Sex','ChestPainType','FastingBS','RestingECG','ST_Slope','ExerciseAngina','HeartDisease']

# Numerical Variables

sns.set(style='whitegrid', palette='deep', font_scale=1.1, rc={'figure.figsize':[8,5]})
df[numerical].hist(bins=15, figsize=(15,6), layout=(2,4))
plt.show()

sns.pairplot(df[numerical + ['HeartDisease']], hue='HeartDisease')
plt.show()

# Categorical Variables

fig, ax = plt.subplots(2, 3, figsize=(20,10))

for variable, subplot in zip(categorical, ax.flatten()):
    sns.countplot(x=df[variable], hue='HeartDisease', data=df, ax=subplot, palette='Set2')
    for label in subplot.get_xticklabels():
        label.set_rotation(90)
        plt.show()
 %%
# Heart disease frequency acc to sex

pd.crosstab(df.HeartDisease, df.Sex).plot(kind="bar", 
				figsize=(10,6), 
				color = ["salmon","lightblue"])


plt.title("Heart Disease Frequency for Sex")
plt.xlabel("0= No Disese, 1= Disease")
plt.ylabel("Amount")
plt.legend(["Female", "Male"])
plt.xticks(rotation=0);
plt.show()
Heart disease per chest pain
pd.crosstab(df.ChestPainType,df.HeartDisease).plot(kind="bar",
                                  figsize=(10,6),
                                  color=["salmon", "lightblue"])

plt.title("Heart Disease Frequency Per Chest Pain ")
plt.xlabel("Chest Pain")
plt.ylabel("Amount")
plt.legend(["No Disese", "Disese"])
plt.xticks(rotation=0);
plt.show()

# %% Categorical to Numerical conversion

df['Sex'].replace({'F':0,'M':1},inplace=True)
df['ChestPainType'].replace({'ASY':1,'ATA':2,'NAP':3,'TA':4},inplace=True)
df['RestingECG'].replace({'LVH':1,'Normal':2,'ST':3},inplace=True)
df['ExerciseAngina'].replace({'N':0,'Y':1},inplace=True)
df['ST_Slope'].replace({'Down':1,'Flat':2, 'Up':3},inplace=True)

# %%
corr_matrix = df.corr()

fig, ax = plt.subplots(figsize=(15,10))
ax = sns.heatmap(corr_matrix,
                annot=True,
                linewidths=0.5,
                fmt=".2f",
                cmap = "YlGnBu");

plt.show()

# %% Modelling

X = df.drop("HeartDisease", axis=1)
# y= np.array(df["HeartDisease"])
y= df["HeartDisease"]

# Normalization
scaler = MinMaxScaler()
scaler.fit(X)
scaled = scaler.fit_transform(X)
X = pd.DataFrame(scaled, columns=X.columns)

from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test =train_test_split(X, y, test_size=0.25, random_state=42)

num_folds = 10

acc_per_fold = []
loss_per_fold = []

# Merge inputs and targets
# inputs = np.concatenate((X_train, X_test), axis=0)
# targets = np.concatenate((y_train, y_test), axis=0)

# Define the K-fold Cross Validator
from sklearn.model_selection import KFold
kfold = KFold(n_splits=num_folds, shuffle=True)

# K-fold Cross Validation model evaluation
fold_no = 1
for train, test in kfold.split(X):
    from tensorflow import keras
    from keras.models import Sequential
    from keras import Input
    import tensorflow as tf
    from keras.utils.vis_utils import plot_model
    from keras.layers import Dense, SimpleRNN
    from keras.layers import LSTM
    from keras.layers import Dropout

    model = Sequential()
    model.add(Dense(20, activation='relu'))  # First hidden layer
    model.add(Dropout(0.25))
    model.add(Dense(16, activation='relu'))  # Second hidden layer
    model.add(Dropout(0.25))
    model.add(Dense(12, activation='relu'))  # Third hidden layer
    model.add(Dropout(0.25))
    model.add(Dense(1, activation='sigmoid'))  # Output layer

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['Accuracy'])
    feat_train, feat_test = X.iloc[train], X.iloc[test]
    targ_train, targ_test = y.iloc[train], y.iloc[test]

    from imblearn.over_sampling import SMOTE

    os = SMOTE(sampling_strategy='minority', random_state=1, k_neighbors=5)
    train_smote_X, train_smote_Y = os.fit_resample(feat_train, targ_train)
    feat_train = pd.DataFrame(data=train_smote_X, columns=feat_train.columns)
    targ_train = pd.DataFrame(data=train_smote_Y)

    history = model.fit(feat_train, targ_train, batch_size=30, epochs=100)

    # Generate a print
    print('------------------------------------------------------------------------')
    print(f'Training for fold {fold_no} ...')

    # Fit data to model

    # Generate generalization metrics
    scores = model.evaluate(feat_test, targ_test)
    print(
        f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1] * 100}%')
    acc_per_fold.append(scores[1] * 100)
    loss_per_fold.append(scores[0])

    # Increase fold number
    fold_no = fold_no + 1

# == Provide average scores ==
print('------------------------------------------------------------------------')
print('Score per fold')
for i in range(0, len(acc_per_fold)):
  print('------------------------------------------------------------------------')
  print(f'> Fold {i+1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}%')
print('------------------------------------------------------------------------')
print('Average scores for all folds:')
print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
print(f'> Loss: {np.mean(loss_per_fold)}')
print('------------------------------------------------------------------------')
