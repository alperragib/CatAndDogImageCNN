from unicodedata import category
import numpy as np
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
import os
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Activation, BatchNormalization, Flatten

image_width = 128
image_height = 128
image_size = (image_width,image_height)
image_channels = 3

filenames = os.listdir("C:/Users/alper/PythonProjects/CatAndDogImageProject/train")
categories = []
for filename in filenames:
    category = filename.split(".")[0]
    if category == "dog":
        categories.append(1)
    else:
        categories.append(0)
df = pd.DataFrame({"filename": filenames, "category": categories})

sample = random.choice(filenames)
image = load_img("C:/Users/alper/PythonProjects/CatAndDogImageProject/train/"+sample)
plt.imshow(image)

model = Sequential()
model.add(Conv2D(32,(3,3), activation="relu", input_shape=(image_width,image_height,image_channels)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Conv2D(64,(3,3), activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Conv2D(128,(3,3), activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(512, activation="relu"))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(2, activation="softmax"))
model.summary()

model.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=["accuracy"])

df["category"] = df["category"].replace({0:"cat", 1: "dog"})
train_df, validate_df = train_test_split(df,test_size=0.2)
train_df = train_df.reset_index(drop=True)
validate_df = validate_df.reset_index(drop=True)

total_train = train_df.shape[0]
total_validate = validate_df.shape[0]
batch_size = 15

train_datagen = ImageDataGenerator(
    rotation_range = 15,
    rescale= 1./255,
    shear_range= 0.2,
    zoom_range= 0.2,
    horizontal_flip= True,
    width_shift_range= 0.1,
    height_shift_range= 0.1)
train_generator = train_datagen.flow_from_dataframe(
    train_df,
    "C:/Users/alper/PythonProjects/CatAndDogImageProject/train/",
    x_col= "filename",
    y_col= "category",
    target_size= image_size,
    class_mode= "categorical",
    batch_size= batch_size
)

validation_datagen = ImageDataGenerator(rescale= 1./255)
validation_generator = validation_datagen.flow_from_dataframe(
    validate_df,
    "C:/Users/alper/PythonProjects/CatAndDogImageProject/train/",
    x_col= "filename",
    y_col= "category",
    target_size= image_size,
    class_mode= "categorical",
    batch_size= batch_size
)

epoch = 1
history = model.fit_generator(
    train_generator,
    epochs=epoch,
    validation_data=validation_generator,
    validation_steps=total_validate,
    steps_per_epoch=total_train
)

test_filenames = os.listdir("C:/Users/alper/PythonProjects/CatAndDogImageProject/test/")
test_df = pd.DataFrame(
    {'filename': test_filenames}
)
nb_samples = test_df.shape[0]

test_gen = ImageDataGenerator(rescale=1./255)
test_generator = test_gen.flow_from_dataframe(
    test_df,
    "C:/Users/alper/PythonProjects/CatAndDogImageProject/test/",
    x_col= "filename",
    y_col= None,
    target_size= image_size,
    class_mode= None,
    batch_size= batch_size,
    shuffle=False
)

predict = model.predict_generator(
    test_generator, steps=np.ceil(nb_samples/batch_size)
)

test_df["category"] = np.argmax(predict, axis=-1)
label_map = dict((v,k) for k,v in train_generator.class_indices.items())
test_df["category"] = test_df["category"].replace(label_map)

sample_test = test_df.head(18)
sample_test.head()

for index, row in sample_test.iterrows():
    filename = row["filename"]
    category = row["category"]
    img = load_img("C:/Users/alper/PythonProjects/CatAndDogImageProject/test/"+filename,target_size=image_size)
    plt.subplot(6,3,index+1)
    plt.imshow(img)
    plt.xlabel(category)
plt.tight_layout()
plt.show()


