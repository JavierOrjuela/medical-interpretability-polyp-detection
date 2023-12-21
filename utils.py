# Librerías para manejo de archivos, arrays, procesos, etc.
import os
import cv2
import json
import random
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm

# Librerías relacionadas con modelos de aprendizaje automático, métricas, optimizadores, etc.
from functools import partial
import tensorflow as tf
from tensorflow import keras
from sklearn.utils import shuffle
os.environ["SM_FRAMEWORK"] = "tf.keras"
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.applications import *
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import CustomObjectScope
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.callbacks import *
from tensorflow.keras.optimizers import Adam, Nadam, Adamax, SGD
from tensorflow.keras.metrics import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.image import ResizeMethod
import tensorflow_probability as tfp
import matplotlib.ticker as mticker

# Librerías para la visualización de gráficos y manipulación de imágenes
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import seaborn as sns
from PIL import Image
from skimage.transform import resize

# Configuración para ignorar advertencias
import warnings
import segmentation_models as sm


BATCH_SIZE=8

# BinaryIoU: Calcula el coeficiente de intersección sobre unión (IoU) para clases binarias.
iou_metric = tf.keras.metrics.BinaryIoU()
# Precision: Calcula la precisión del modelo.
prec_metric = tf.keras.metrics.Precision()
# Recall: Calcula el recall (sensibilidad) del modelo.
rec_metric = tf.keras.metrics.Recall()
# AUC: Calcula el área bajo la curva (AUC).
roc_metric = tf.keras.metrics.AUC(curve='ROC', name='roc_auc')
# Accuracy: Calcula la precisión del modelo.
acc_metric = tf.keras.metrics.BinaryAccuracy()
FN_metric= tf.keras.metrics.FalseNegatives()
FP_metric= tf.keras.metrics.FalsePositives()

def augmented(train,image_path,mask_path):
    # Definición de dimensiones y coordenadas para el recorte
    crop_width=310
    crop_height=265
    a=9
    b=54

    # Lectura de las imágenes y máscaras
    images = tf.io.read_file(image_path)
    mask = tf.io.read_file(mask_path)

    # Decodificación de las imágenes y máscaras
    images = tf.io.decode_png(images,channels=3)
    mask = tf.io.decode_png(mask,channels=1)

    # Codificación de las imágenes y máscaras a formato jpeg
    images = tf.io.encode_jpeg(images)
    mask = tf.io.encode_jpeg(mask)

    # Decodificación y recorte de las imágenes / máscaras
    images = tf.io.decode_and_crop_jpeg(images,[a,b,crop_height-a,crop_width-b],channels=3)
    mask = tf.io.decode_and_crop_jpeg(mask,[a,b,crop_height-a,crop_width-b],channels=1)

    # Para train
    if train:
        # Ajuste aleatorio del brillo de la imagen
        images = tf.image.random_brightness(images,0.25,seed=1234)

        # Ajuste aleatorio del contraste de la imagen
        images = tf.image.random_contrast(images,0.6,2,seed=1234)

        # Ajuste aleatorio de la saturación de la imagen
        images = tf.image.random_saturation(images,0.4,2,seed=1234)

        # Flip izquierda/derecha de la imagen y la máscara
        if tf.random.uniform([],seed=1234) > 0.5:
          images = tf.image.flip_left_right(images)
          mask = tf.image.flip_left_right(mask)

        # Flip arriba/abajo de la imagen y la máscara
        if tf.random.uniform([],seed=1234) > 0.5:
          images = tf.image.flip_up_down(images)
          mask = tf.image.flip_up_down(mask)

    # Normalización de las imágenes y máscaras
    images = tf.cast(images, tf.float32)/ 255.0
    mask = tf.cast(mask, tf.float32)

    # Binarización de la máscara
    mask = tf.where(mask > 125,1.,0.)

    return images, mask


def preprocessing(df_path,BATCH=BATCH_SIZE,train=False):
    dataset = tf.data.Dataset.from_tensor_slices((df_path['png_image_path'],df_path['png_mask_path']))
    if train:
        dataset= dataset.shuffle(len(dataset))
    dataset = dataset.map(partial(augmented,train)).batch(BATCH)
    return dataset


def modelos_segmentacion(MODEL,loss,BACKBONE,epochs,train_dataset,valid_dataset,test_dataset,plot=False,summary = False):
    # Se selecciona el modelo de segmentación a utilizar
    print('Modelo seleccionado:', MODEL)
    print('Backbone seleccionado:', BACKBONE)
    n_classes=1

    # Se configura el modelo con la estructura (BACKBONE).
    if MODEL == 'Unet':
        model = sm.Unet(BACKBONE, classes=n_classes, activation='sigmoid')
    elif MODEL == 'FPN':
        model = sm.FPN(BACKBONE, classes=n_classes, activation='sigmoid')
    elif MODEL == 'Linknet':
        model = sm.Linknet(BACKBONE, classes=n_classes, activation='sigmoid')
    elif MODEL == 'SegFormer':
        model = SegFormer_B0(input_shape = (256, 256, 3), num_classes = 1)
    elif MODEL == 'FPNFLIP' or MODEL == 'FPNREP':
        model = smflip.FPN(BACKBONE, classes=n_classes, activation='sigmoid')
    elif MODEL == 'UNETREP':
        model = smflip.Unet(BACKBONE, classes=n_classes, activation='sigmoid')
    elif MODEL == 'LINKREP':
        model = smflip.Linknet(BACKBONE, classes=n_classes, activation='sigmoid')

    neg_log= lambda x, prob:-prob.log_prob(x)

    LOSSES = {
    'bce':tf.keras.losses.BinaryCrossentropy(),
    'JaccardLoss': sm.losses.JaccardLoss(),  # Jaccard Loss
    'DiceLoss':sm.losses.DiceLoss(),  # Dice Loss
    'BinaryFocalLoss':sm.losses.BinaryFocalLoss(),  # Binary Focal Loss
    'total':sm.losses.DiceLoss() + (0.5 * sm.losses.BinaryFocalLoss()), # Combinación de Dice Loss y Categorical Focal Loss
    'bnn': neg_log
      }
    LOSS= LOSSES[loss]
    # Se compila el modelo definiendo la función de pérdida, el optimizador y las métricas a utilizar
    print('Función de Pérdida seleccionada:', LOSS)
    model.compile(
        loss=LOSS,
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
        metrics=[acc_metric,iou_metric,prec_metric,rec_metric,roc_metric,FN_metric,FP_metric]
    )

    folder_model='/content/drive/MyDrive/Articulo/Modelos_BNN/bnn_model_{}-{}-{}'.format(MODEL,BACKBONE,LOSS)
    if not os.path.exists(folder_model):
        os.makedirs(folder_model)

    # Se entrena el modelo usando el conjunto de datos de entrenamiento y validación, y se especifican las funciones de callback
    callbacks =  [
    EarlyStopping(monitor='val_loss',mode='min', patience=4,verbose=1),
    ModelCheckpoint(filepath=folder_model + '/checkpoint',
                    save_best_only=True, monitor='val_loss', mode='min',
                    save_weights_only=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',factor=0.8,patience=2,verbose=1,mode='min',min_lr=1e-7)
    ]
    if summary == True:
        model.summary()

    history= model.fit(
        train_dataset,
        epochs=epochs,
        validation_data=valid_dataset,
        callbacks=callbacks,
        verbose=1
    )
    print('Empezar evaluacion:')
    hist1=model.evaluate(test_dataset)
    np.save(folder_model + '/' + 'eval_bnn.npy',hist1)
    if plot == True:
      print('Visualización de las imágenes:')
      imagen_show(model,2)
    df1=pd.DataFrame(history.history)
    df1.to_csv(folder_model + '/' + 'bnn_dataframe.csv',index=False)
    return model


def carga_evaluacion(ruta,MODEL='Unet',BACKBONE='efficientnetb7'):
    n_classes=1
    if MODEL == 'Unet':
        model = sm.Unet(BACKBONE, classes=n_classes, activation='sigmoid')
        model.load_weights(ruta + 'checkpoint.h5')
    elif MODEL == 'FPN':
        model = sm.FPN(BACKBONE, classes=n_classes, activation='sigmoid')
        model.load_weights(ruta + 'checkpoint.h5')
    elif MODEL == 'Linknet':
        model = sm.Linknet(BACKBONE, classes=n_classes, activation='sigmoid')
        model.load_weights(ruta + 'checkpoint.h5')
    # Cargar los resultados
    df=pd.read_csv(ruta + '/dataframe.csv')
    data_eval = np.load (ruta + '/eval.npy')
    data_eval=pd.DataFrame(data_eval,index=columnas)
    df['epoch'] = df.index + 1

    #Graficas vs Epoca
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter(useOffset=False))
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter(useOffset=False))
    ax.plot(df['epoch'], df['binary_accuracy'], label='Accuracy Train', color='black',linestyle='dashdot')
    ax.plot(df['epoch'], df['loss'], label='Perdida Train', color='black',linestyle='--')
    ax.plot(df['epoch'], df['val_binary_accuracy'], label='Accuracy Val', color='black')
    ax.plot(df['epoch'], df['val_loss'], label='Perdida Val', color='black',linestyle='dotted')
    ax.set_title('Accuracy y loss vs epoch')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Valor')
    plt.ylim(0.0,1.05)
    ax.legend()
    plt.tight_layout()
    plt.show()
    return df,data_eval,model,fig


def imagenes(j,model_2,test_dataset,importance=False,n=8,limit=8,unica=0):

  #Definimos la funcion para graficar las predicciones y la desviacion estandar por modelo.

  fig, ax = plt.subplots(n, 3, figsize=(13, 13))

  for i, (img, mask) in enumerate(test_dataset):
      if i<limit:
        # Imagen original
        if unica == 0:
          ax[i, 0].imshow(img[j])
        if unica == 1:
          ax[i, 0].imshow(img)
        ax[i, 0].set_title('Input Image')
        ax[i, 0].axis('off')

        # Máscara original
        if unica == 0:
           ax[i, 1].imshow(mask[j], cmap='gray')
           ax[i, 1].set_title('Ground Truth'.format(i,j))
        if unica == 1:
           ax[i, 1].imshow(mask, cmap='gray')
           ax[i, 1].set_title('Ground Truth'.format(i,j))
        ax[i, 1].axis('off')

        # Máscara determinista
        if unica == 0:
           predictions_det = model_2.predict(img,verbose=0)
           predictions_det = (predictions_det > 0.5).astype('uint8')
           ax[i, 2].imshow(predictions_det[j], cmap='gray')
        if unica == 1:
           img = np.expand_dims(img, axis=0)
           predictions_det = model_2.predict(img,verbose=0)
           predictions_det = (predictions_det > 0.5).astype('uint8')
           ax[i, 2].imshow(predictions_det[0], cmap='gray')
        ax[i, 2].set_title('Prediction')
        ax[i, 2].axis('off')
  plt.tight_layout()
  plt.show()


def imagenes_seleccionadas(seleccion,data):
  lista = []
  for i, (img, mask) in enumerate(data):
      for n in range(8):
        tuple_location=(i,n)
        #print(tuple_location)
        if tuple_location in seleccion:
            lista.append((img[n], mask[n]))

  def generator():
    for element in lista:
        yield element

  dataset = tf.data.Dataset.from_generator(generator, output_signature=(
    tf.TensorSpec(shape=(256, 256, 3), dtype=tf.float32),
    tf.TensorSpec(shape=(256, 256, 1), dtype=tf.float32)
  ))

  return dataset
