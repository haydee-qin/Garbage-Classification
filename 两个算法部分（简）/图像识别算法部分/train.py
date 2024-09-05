import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import random
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint

from sklearn.metrics import classification_report, log_loss, accuracy_score
from sklearn.model_selection import train_test_split


def all_files_path(rootDir):
    filepaths = []
    for root, dirs, files in os.walk(rootDir):  # 分别代表根目录、文件夹、文件
        for file in files:  # 遍历文件
            file_path = os.path.join(root, file)  # 获取文件绝对路径
            filepaths.append(file_path)  # 将文件路径添加进列表
        for dir in dirs:  # 遍历目录下的子目录
            dir_path = os.path.join(root, dir)  # 获取子目录路径
            all_files_path(dir_path)  # 递归调用
    return filepaths


# data_dir = './input/garbage-classification/Garbage classification/Garbage classification/'
data_dir = './input/垃圾图片库/'

Name = []
for file in os.listdir(data_dir):
    Name += [file]

print(Name)
print(len(Name))

N = []
for i in range(len(Name)):
    N += [i]

normal_mapping = dict(zip(Name, N))
reverse_mapping = dict(zip(N, Name))


def mapper(value):
    return reverse_mapping[value]


dataset = []
count = 0
for file in os.listdir(data_dir):
    path = os.path.join(data_dir, file)
    filedirs = all_files_path(path)
    for im in filedirs:
        try:
            image = load_img(im, grayscale=False, color_mode='rgb', target_size=(60, 60))
        except:
            print(im)
        image = img_to_array(image)
        image = image / 255.0
        dataset += [[image, count]]
    count = count + 1

n = len(dataset)
print(n)

num = []
for i in range(n):
    num += [i]
random.shuffle(num)
print(num[0:5])

data, labels = zip(*dataset)
data = np.array(data)
labels = np.array(labels)

train = data[num[0:(n // 10) * 8]]
trainlabel = labels[num[0:(n // 10) * 8]]

test = data[num[(n // 10) * 8:]]
testlabel = labels[num[(n // 10) * 8:]]

trainx = train
trainy = trainlabel
testx = test
testy = testlabel
print(trainx.shape)
print(testx.shape)
print(trainy.shape)
print(testy.shape)

datagen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True, rotation_range=20, zoom_range=0.2,
                             width_shift_range=0.2, height_shift_range=0.2, shear_range=0.1, fill_mode="nearest")

pretrained_model3 = tf.keras.applications.DenseNet201(input_shape=(60, 60, 3), include_top=False, weights='imagenet',
                                                      pooling='avg')
pretrained_model3.trainable = False

inputs3 = pretrained_model3.input
x3 = tf.keras.layers.Dense(128, activation='relu')(pretrained_model3.output)
outputs3 = tf.keras.layers.Dense(6, activation='softmax')(x3)
model = tf.keras.Model(inputs=inputs3, outputs=outputs3)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

filepath = 'weights.best.h5'
checkpoint = ModelCheckpoint(filepath=filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callback_list = [checkpoint]
his = model.fit(datagen.flow(trainx, trainy, batch_size=32), validation_data=(testx, testy), epochs=300,
                callbacks=callback_list)

y_pred = model.predict(testx)
pred = np.argmax(y_pred, axis=1)
# ground = np.argmax(testy,axis=1)
ground = testy
print(classification_report(ground, pred))
model.save('my_model.h5')
get_acc = his.history['accuracy']
value_acc = his.history['val_accuracy']
get_loss = his.history['loss']
validation_loss = his.history['val_loss']

epochs = range(len(get_acc))
plt.plot(epochs, get_acc, 'r', label='Accuracy of Training data')
plt.plot(epochs, value_acc, 'b', label='Accuracy of Validation data')
plt.title('Training vs validation accuracy')
plt.legend(loc=0)
plt.figure()
plt.show()

epochs = range(len(get_loss))
plt.plot(epochs, get_loss, 'r', label='Loss of Training data')
plt.plot(epochs, validation_loss, 'b', label='Loss of Validation data')
plt.title('Training vs validation loss')
plt.legend(loc=0)
plt.figure()
plt.show()

load_img("./input/垃圾图片库/有害垃圾/有害垃圾_蓄电池/img_蓄电池_1.jpeg",
         target_size=(150, 150))

image = load_img("./input/垃圾图片库/可回收物/可回收物_水杯/不锈钢杯子_可回收物/img_不锈钢杯子_1.jpeg",
                 target_size=(60, 60))

image = img_to_array(image)
image = image / 255.0
prediction_image = np.array(image)
prediction_image = np.expand_dims(image, axis=0)

prediction = model.predict(prediction_image)
value = np.argmax(prediction)
move_name = mapper(value)
print("Prediction is {}.".format(move_name))

print(test.shape)
pred2 = model.predict(test)
print(pred2.shape)

PRED = []
for item in pred2:
    value2 = np.argmax(item)
    PRED += [value2]

ANS = testlabel

accuracy = accuracy_score(ANS, PRED)
print(accuracy)
