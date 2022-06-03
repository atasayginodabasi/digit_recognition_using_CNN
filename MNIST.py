import os
import pandas as pd
import seaborn as sns
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical

keras = tf.keras
# ----------------------------------------------------------------------------------------------------------------------

# Reading the data
print(os.listdir(r'C:/Users/ata-d/'))
train = pd.read_csv(r'C:/Users/ata-d/OneDrive/Masaüstü/ML/Datasets/mnist_train.csv')
test = pd.read_csv(r'C:/Users/ata-d/OneDrive/Masaüstü/ML/Datasets/mnist_test.csv')

# ----------------------------------------------------------------------------------------------------------------------

print('Shape of the train data:', train.shape)
print('Shape of the test data:', test.shape)

# Checking for the labels in the train dataset
train_unique = pd.DataFrame(train['label'].unique()).sort_values(by=0)

# Distribution of the Labels of the train dataset
'''''''''
plt.figure(figsize=(15, 8))
splot = sns.countplot(data=train, x='label',
                      order=train['label'].value_counts().index,
                      edgecolor=(0, 0, 0),
                      linewidth=2)

for p in splot.patches:
    splot.annotate(format(p.get_height(), '.1f'),
                   (p.get_x() + p.get_width() / 2., p.get_height()),
                   ha='center', va='center',
                   xytext=(0, 9),
                   textcoords='offset points')
plt.ylabel('Frequency of the Labels', fontsize=14)
plt.xlabel('Labels', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.title('Distribution of the Labels of the Train Dataset', fontsize=20)
'''''''''

# ----------------------------------------------------------------------------------------------------------------------

y = train['label']  # Label
X = train.drop(['label'], axis=1)  # Feature

# Some examples of the samples
'''''''''
for i in range(9):
    plt.subplot(330 + 1 + i)
    fig = X.iloc[i].values.reshape((28, 28))
    plt.imshow(fig)
    plt.axis('off')
plt.show()
'''''''''

# ----------------------------------------------------------------------------------------------------------------------

# Normalizing the dataset
X = X / 255.0

# CNN expects 3D inputs, so I will convert my data tto 3D form
y = y.values.reshape(-1, 1)
X = X.values.reshape(-1, 28, 28, 1)

print('Shape of the X matrix:', X.shape)
print('Shape of the y matrix:', y.shape)

# Encoding the labels
y = to_categorical(y, num_classes=10)

# ----------------------------------------------------------------------------------------------------------------------

# Train-Test split
trainX, valX, trainY, valY = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=13)

# ----------------------------------------------------------------------------------------------------------------------

# CNN Model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(trainX.shape[1:])))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))

# Output layer
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])  # Adam Optimizer

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', mode='auto', patience=2,
                                                  restore_best_weights=True)

hist = model.fit(trainX, trainY, epochs=10, batch_size=64, callbacks=[early_stopping],
                 verbose=1, validation_data=(valX, valY))
hist.history.keys()
model.summary()

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Training Loss of the Model', fontsize=30)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.legend(['Train Loss', 'Validation Loss'], loc='upper right')
plt.xlabel('Epochs', fontsize=20)
plt.ylabel('Losses', fontsize=20)

# plt.plot(hist.history['accuracy'])
# plt.plot(hist.history['val_accuracy'])
# plt.title('Training Accuracy of the Model', fontsize=30)
# plt.xticks(fontsize=18)
# plt.yticks(fontsize=18)
# plt.legend(['Train Accuracy', 'Validation Accuracy'], loc='upper right')
# plt.xlabel('Epochs', fontsize=20)
# plt.ylabel('Losses', fontsize=20)

model.evaluate(valX, valY)
# ----------------------------------------------------------------------------------------------------------------------

test = test / 255.0
test = test.values.reshape(-1, 28, 28, 1)

predictions = model.predict(test)
predictions = np.argmax(predictions, axis=1)

predictions = pd.DataFrame(predictions)

subs = pd.read_csv(r'C:/Users/ata-d/OneDrive/Masaüstü/ML/Datasets/mnist_sample_submission.csv')
subs = pd.concat([subs['ImageId'], predictions], axis=1)
subs.columns = ['ImageId', 'Label']
# subs.to_csv('C:/Users/ata-d/OneDrive/Masaüstü/Ata.csv', index=False)
