import tensorflow as tf                                                               # библиотека Tensorflow
import keras                                                                          # библиотека Keras
from keras.layers import Input, Conv2D, MaxPooling2D, Activation, ReLU, Rescaling     # cлои библиотеки Keras
from keras.layers import BatchNormalization, Conv2DTranspose, Concatenate             # cлои библиотеки Keras
from keras.layers import Rescaling, Resizing                                          # cлои библиотеки Keras
from keras.models import Model, Sequential                                            # конструкторы построения моделей библиотеки Keras

from keras.optimizers import Adam                                                     # оптимизатор Adam
from keras.preprocessing.image import  load_img                                       # загрузка изображений
from keras.utils import to_categorical                                                # преобразует вектор класса (целые числа) в двоичную матрицу класса

from keras.callbacks import EarlyStopping

import random                                                                         # генератор случайных чисел

import numpy as np                                                                    # библиотека линейной алгебры
import pandas as pd                                                                   # библиотека обработки табличных данных
import os                                                                             # библиотека работы с функциями операционной системы, в том числе с файлами
import albumentations as A                                                            # библиотека аугментации изображений (https://albumentations.ai/)

import matplotlib.pyplot as plt                                                       # библиотека для рисования графиков
%matplotlib inline
!pip install opendatasets
import opendatasets as op

op.download("https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database/")

dataset_path = './covid19-radiography-database/COVID-19_Radiography_Dataset/Normal'

image_dir = 'images'
label_dir = 'masks'

original_image = os.path.join(dataset_path, image_dir, 'Normal-1.png')       # путь до ориганального изображения
label_image_semantic = os.path.join(dataset_path, label_dir, 'Normal-1.png') # путь до маски

fig, axs = plt.subplots(1, 2, figsize=(16, 8))                          # задаем область для построения (канвас)

img = np.array(load_img(original_image, target_size=(256, 256), color_mode='rgb'))   # загружаем оригинальное изображение как RGB с 3 каналами
mask = np.array(load_img(label_image_semantic, target_size=(256, 256), color_mode='grayscale'))  # загружаем маску как "отеннки серого", т.е. в один канал

axs[0].imshow(img)  # отрисовываем оригинальное изображение
axs[0].grid(False)

axs[1].imshow(mask) # отрисовываем маску (одноканальное изображение, каждый класс отображается как отдельный цвет)
axs[1].grid(False)

input_img_path = sorted(
    [
        os.path.join(dataset_path, image_dir, fname)
        for fname in os.listdir(os.path.join(dataset_path, image_dir))
        if fname.endswith(".png")
    ]
)

target_img_path = sorted(
    [
        os.path.join(dataset_path, label_dir, fname)
        for fname in os.listdir(os.path.join(dataset_path, label_dir))
        if fname.endswith(".png")
    ]
)

batch_size = 16
img_size = (256, 256)
NUM_CLASSES = 2

# Генератор для перебора данных (в виде массивов Numpy)
class datasetGenerator(keras.utils.Sequence):

    def __init__(self, batch_size, img_size, input_img_path, target_img_path = None, num_classes = NUM_CLASSES, validation = False):
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_path = input_img_path
        self.target_img_path = target_img_path
        self.num_classes = num_classes
        self.validation = validation


    def __len__(self):
        """Возвращает число мини-батчей обучающей выборки"""
        return len(self.target_img_path) // self.batch_size


    def __getitem__(self, idx):
        """Возвращает кортеж (input, target) соответствующий индексу пакета idx"""

        # Формируем пакеты из ссылок путем среза длинной в batch_size и возвращаем пакет по индексу
        batch_input_img_path = self.input_img_path[idx*self.batch_size:(idx+1)*self.batch_size]
        batch_target_img_path = self.target_img_path[idx*self.batch_size:(idx+1)*self.batch_size]

        # Создадим массив numpy, заполненный нулями, для входных данных формы (BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, 3) и типа данных float32
        x = np.zeros((self.batch_size, *self.img_size, 3), dtype="float32")

        # Создадим массив numpy, заполненный нулями, для выходных данных формы (BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, 1) и типа данных uint8
        y = np.zeros((self.batch_size, *self.img_size, self.num_classes), dtype="uint8")

        # В цикле заполняем массивы с изображениями x и y
        # Перебираем пакеты из путей batch_input_img_path и batch_target_img_path к изображениям
        # zip возвращает для нескольких последовательностей список кортежей из элементов последовательностей с одинаковыми индексами
        for _, paths in enumerate(zip(batch_input_img_path, batch_target_img_path)):

            # Загружаем изображение и маску используя путь файловой системы
            img = np.array(load_img(paths[0], target_size=self.img_size, color_mode='rgb'))         # 3 канала для изображения
            mask = np.array(load_img(paths[1], target_size=self.img_size, color_mode='grayscale'))  # 1 канал для маски

            x[_] = img / 255 # нормализуем изображение
            y[_] = to_categorical(mask / 255, num_classes=self.num_classes) # преобразует маску из целых чисел в двоичную матрицу класса

        return x, y


import random

seed = 1523
random.Random(seed).shuffle(input_img_path)
random.Random(seed).shuffle(target_img_path)


percent = 20 # процент расщепления validation

val_samples = len(input_img_path) * percent // 100

# Расщепим наш датасет  на обучающую и проверочные выборки
train_input_img_path = input_img_path[:-val_samples]
train_target_img_path = target_img_path[:-val_samples]
val_input_img_path = input_img_path[-val_samples:]
val_target_img_path = target_img_path[-val_samples:]

print(len(train_input_img_path))
print(len(val_input_img_path))

train_gen = datasetGenerator(batch_size, img_size, train_input_img_path, train_target_img_path, NUM_CLASSES)
val_gen = datasetGenerator(batch_size, img_size, val_input_img_path, val_target_img_path, NUM_CLASSES, validation = True)

# Архитектура нейросети
def unet_plus_plus(input_size, num_classes, base_filter_num=64):
    inputs = Input(input_size)
    conv0_0 = Conv2D(base_filter_num, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv0_0 = Conv2D(base_filter_num, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv0_0)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv0_0)

    conv1_0 = Conv2D(base_filter_num * 2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv1_0 = Conv2D(base_filter_num * 2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1_0)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv1_0)

    up1_0 = Conv2DTranspose(base_filter_num, (2, 2), strides=(2, 2), padding='same')(conv1_0)
    merge00_10 = Concatenate()([conv0_0,up1_0])
    conv0_1 = Conv2D(base_filter_num, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge00_10)
    conv0_1 = Conv2D(base_filter_num, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv0_1)

    conv2_0 = Conv2D(base_filter_num * 4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv2_0 = Conv2D(base_filter_num * 4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2_0)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv2_0)

    up2_0 = Conv2DTranspose(base_filter_num*  2, (2, 2), strides=(2, 2), padding='same')(conv2_0)
    merge10_20 = Concatenate()([conv1_0,up2_0])
    conv1_1 = Conv2D(base_filter_num * 2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge10_20)
    conv1_1 = Conv2D(base_filter_num * 2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1_1)

    up1_1 = Conv2DTranspose(base_filter_num, (2, 2), strides=(2, 2), padding='same')(conv1_1)
    merge01_11 = Concatenate()([conv0_0,conv0_1,up1_1])
    conv0_2 = Conv2D(base_filter_num, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge01_11)
    conv0_2 = Conv2D(base_filter_num, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv0_2)

    conv3_0 = Conv2D(base_filter_num * 8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv3_0 = Conv2D(base_filter_num * 8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3_0)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv3_0)

    up3_0 = Conv2DTranspose(base_filter_num * 4, (2, 2), strides=(2, 2), padding='same')(conv3_0)
    merge20_30 = Concatenate()([conv2_0,up3_0])
    conv2_1 = Conv2D(base_filter_num * 4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge20_30)
    conv2_1 = Conv2D(base_filter_num * 4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2_1)

    up2_1 = Conv2DTranspose(base_filter_num * 2, (2, 2), strides=(2, 2), padding='same')(conv2_1)
    merge11_21 = Concatenate()([conv1_0,conv1_1,up2_1])
    conv1_2 = Conv2D(base_filter_num * 2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge11_21)
    conv1_2 = Conv2D(base_filter_num * 2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1_2)

    up1_2 = Conv2DTranspose(base_filter_num, (2, 2), strides=(2, 2), padding='same')(conv1_2)
    merge02_12 = Concatenate()([conv0_0,conv0_1,conv0_2,up1_2])
    conv0_3 = Conv2D(base_filter_num, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge02_12)
    conv0_3 = Conv2D(base_filter_num, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv0_3)

    conv4_0 = Conv2D(base_filter_num * 16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv4_0 = Conv2D(base_filter_num * 16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4_0)

    up4_0 = Conv2DTranspose(base_filter_num * 8, (2, 2), strides=(2, 2), padding='same')(conv4_0)
    merge30_40 = Concatenate()([conv3_0,up4_0])
    conv3_1 = Conv2D(base_filter_num * 8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge30_40)
    conv3_1 = Conv2D(base_filter_num * 8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3_1)

    up3_1 = Conv2DTranspose(base_filter_num * 4, (2, 2), strides=(2, 2), padding='same')(conv3_1)
    merge21_31 = Concatenate()([conv2_0,conv2_1,up3_1])
    conv2_2 = Conv2D(base_filter_num * 4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge21_31)
    conv2_2 = Conv2D(base_filter_num * 4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2_2)

    up2_2 = Conv2DTranspose(base_filter_num * 2, (2, 2), strides=(2, 2), padding='same')(conv2_2)
    merge12_22 = Concatenate()([conv1_0,conv1_1,conv1_2,up2_2])
    conv1_3 = Conv2D(base_filter_num * 2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge12_22)
    conv1_3 = Conv2D(base_filter_num * 2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1_3)

    up1_3 = Conv2DTranspose(base_filter_num, (2, 2), strides=(2, 2), padding='same')(conv1_3)
    merge03_13 = Concatenate()([conv0_0,conv0_1,conv0_2,conv0_3,up1_3])
    conv0_4 = Conv2D(base_filter_num, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge03_13)
    conv0_4 = Conv2D(base_filter_num, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv0_4)

    outputs = Conv2D(num_classes, kernel_size = (1, 1), activation = 'softmax')(conv0_4)

    model = Model(inputs, outputs)

    return model


num_classes = NUM_CLASSES # 2 класса объектов
input_shape = (img_size[0], img_size[1], 3) # размер к которому преобразуем изображение, 3 канала - RGB

# Инициализация модели
model = unet_plus_plus(input_shape, num_classes)

model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=['accuracy'])

callbacks = [
    EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)
]


# Обучение модели
epochs = 2
history = model.fit(train_gen,
                    validation_data=val_gen,
                    epochs=epochs,
                    batch_size=batch_size,
                    callbacks=callbacks
                   )



