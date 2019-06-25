# coding: utf-8

import os
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
import keras
import numpy as np

# 各年代の画像をリスト化
Youth_files = os.listdir('C:/Users/itaku/Desktop/SampleData/Generation/10-20')
Thirty_files = os.listdir('C:/Users/itaku/Desktop/SampleData/Generation/20-40')
Fifty_files = os.listdir('C:/Users/itaku/Desktop/SampleData/Generation/40-60')
Seventy_files = os.listdir('C:/Users/itaku/Desktop/SampleData/Generation/60-')

# 配列Xに画像を入れる
X = []
for i in Youth_files:
    img = cv2.imread('C:/Users/itaku/Desktop/SampleData/Generation/10-20/'+i)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    X.append(rgb)

for i in Thirty_files:
    img = cv2.imread('C:/Users/itaku/Desktop/SampleData/Generation/20-40/'+i)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    X.append(rgb)

for i in Fifty_files:
    img = cv2.imread('C:/Users/itaku/Desktop/SampleData/Generation/40-60/'+i)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    X.append(rgb)

for i in Seventy_files:
    img = cv2.imread('C:/Users/itaku/Desktop/SampleData/Generation/60-/'+i)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    X.append(rgb)

# Yに各ラベルを入れる
Y = []
label_names = ['Youth', '20-40', '40-60', 'Old']
for i in range(len(Youth_files)):
    Y.append(0)

for i in range(len(Thirty_files)):
    Y.append(1)

for i in range(len(Fifty_files)):
    Y.append(2)

for i in range(len(Seventy_files)):
    Y.append(3)

# カスケードで顔抽出
cascade = cv2.CascadeClassifier('C:/Users/itaku/Desktop/opencv/data/haarcascades/haarcascade_frontalface_default.xml')
X_std = []

for i in range(len(X)):
    img = X[i]
    face_list = cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=1, minSize=(20, 20))
    for face in face_list:
        x, y, w, h = face
        face = img[y:y+h, x:x+w]
        face = cv2.resize(face, dsize=(100, 100))

    X_std.append(face)

print(len(X_std))
# 顔画像全表示
plt.figure(figsize=(10, 10))
for i in range(len(X_std)):
    plt.subplot(10, 11, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(X_std[i], cmap=plt.cm.binary)
    plt.xlabel(label_names[Y[i]])

plt.show()

# 正規化
for i in range(len(X_std)):
    X_std[i] = X_std[i]/255


# 学習用とテスト用
X_train, X_test, Y_train, Y_test = train_test_split(X_std, Y, test_size=0.2)
X_train = np.array(X_train)
X_train = X_train.reshape(-1, 100, 100, 3)
X_test = np.array(X_test)
X_test = X_test.reshape(-1, 100, 100, 3)
input_shape = X_train.shape[1:]


# モデル構築
model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), padding='same', input_shape=input_shape, activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(4, activation='softmax')
])

model.compile(optimizer=keras.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 学習実行
model.fit(X_train, Y_train, epochs=20)

test_loss, test_acc = model.evaluate(X_test, Y_test)
print('Test accuracy:', test_acc)
predictions = model.predict(X_test)

# 正解不正解画像表示定義


def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(label_names[predicted_label],
                                         100 * np.max(predictions_array),
                                         label_names[true_label]),
               color=color)

# 正解不正解ラベル表示定義


def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(4), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)
    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


# 結果表示
num_rows = 3
num_cols = 2
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plot_image(i, predictions, Y_test, X_test)
    plt.subplot(num_rows, 2*num_cols, 2*i+2)
    plot_value_array(i, predictions, Y_test)
plt.show()
