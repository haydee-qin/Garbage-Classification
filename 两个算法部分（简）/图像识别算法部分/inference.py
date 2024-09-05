import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import random

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import load_img, img_to_array

from tensorflow.keras.models import load_model
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint

from sklearn.metrics import classification_report, log_loss, accuracy_score
from sklearn.model_selection import train_test_split

# data_dir = './input/garbage-classification/Garbage classification/Garbage classification/'
data_dir = './input/垃圾图片库/'
Name=[]
for file in os.listdir(data_dir):
    Name+=[file]

N=[]
for i in range(len(Name)):
    N+=[i]
    

normal_mapping=dict(zip(Name,N)) 
reverse_mapping=dict(zip(N,Name)) 

def mapper(value):
    return reverse_mapping[value]

# 从这里开始
# 加载模型
model = load_model('my_model.h5')
# 从本地读取一个图片，如果要用程序传入图片，则替换掉这一行。直接到img_to_array。
image=load_img("./input/垃圾图片库/可回收物/可回收物_水杯/不锈钢杯子_可回收物/img_不锈钢杯子_1.jpeg",target_size=(60,60))
# img转tensor
image=img_to_array(image) 
# 图片归一化
image=image/255.0
prediction_image=np.array(image)
prediction_image= np.expand_dims(image, axis=0)

# 送入模型推理
prediction=model.predict(prediction_image)
# 将推理结果转换为标签，这一步如果不理解可以print出来就知道什么意思了。结果是one-hot编码的。
value=np.argmax(prediction)
move_name=mapper(value)
# 输出推理结果
print("Prediction is {}.".format(move_name))

