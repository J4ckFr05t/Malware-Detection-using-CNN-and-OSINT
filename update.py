import logging
import os
import platform
import tensorflow as tf
import numpy as np
from tensorflow import keras
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from PIL import Image

log_format = "%(levelname)s %(asctime)s - %(message)s"
logging.basicConfig(filename="update_logs.log", level=logging.DEBUG,
                    format=log_format, filemode='w')
logger = logging.getLogger()

cwd = os.getcwd()

Image.MAX_IMAGE_PIXELS = 1000000000  # To avoid Decompression Bomb Warning
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # disable tenserflow debug info

# Two Classes : Malware and Benign
num_classes = 2
#To Balance Dataset
class_weights = {0:1.04746729,1:0.95664828}
OS_NAME = platform.system()

formatted_OS_NAME = OS_NAME.casefold()
if(formatted_OS_NAME == "windows"):
    path_root = cwd+"\Dataset\Dataset"
    tmp_root = os.path.dirname(__file__)+"\\temp\samples\\"

elif(formatted_OS_NAME == "linux" or formatted_OS_NAME == "darwin"):
    path_root = cwd+"/Dataset/Dataset"
    tmp_root = cwd+"/temp/samples/"

class MalUpdate:
    def __init__(self):
        self.Malware_model = Sequential()
        self.preprocess()
        self.cnn_model()
        self.train_model()

    def preprocess(self):
        logger.debug("# preprocessing dataset #")
        print("Preprocessing Dataset..............................................")
        batches = ImageDataGenerator().flow_from_directory(
            directory=path_root, target_size=(64, 64), batch_size=19000)
        self.imgs, self.labels = next(batches)

        x_train, x_test, y_train, y_test = train_test_split(
            self.imgs / 255, self.labels, test_size=0.3)

        self.x_train = np.array(x_train)
        self.y_train = np.array(y_train)

        self.x_test = np.array(x_test)
        self.y_test = np.array(y_test)
        print("Dataset Preprocessing Complete......................................")

    def cnn_model(self):
        logger.debug("# Creating CNN model #")
        print("Creating Model...........................................................")
        self.Malware_model.add(Conv2D(30, kernel_size=(
            3, 3), activation='relu', input_shape=(64, 64, 3)))
        self.Malware_model.add(MaxPooling2D(pool_size=(2, 2)))
        self.Malware_model.add(Conv2D(15, (3, 3), activation='relu'))
        self.Malware_model.add(MaxPooling2D(pool_size=(2, 2)))
        self.Malware_model.add(Dropout(0.25))
        self.Malware_model.add(Flatten())
        self.Malware_model.add(Dense(150, activation='relu'))
        self.Malware_model.add(Dropout(0.4))
        self.Malware_model.add(Dense(70, activation='relu'))
        self.Malware_model.add(Dense(num_classes, activation='sigmoid'))
        self.Malware_model.compile(
            loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    def train_model(self):

        logger.debug("# Training Model #")
        # Train and test the model
        print("Started Training Model ....................................")
        self.Malware_model.fit(self.x_train, self.y_train, validation_data=(
            self.x_test, self.y_test), epochs=20, batch_size=64, class_weight=class_weights, verbose=1)
        self.Malware_model.save("CNN_Model")
        print("model saved\n","Training Completed")
        print('',flush=True)
        print('True')

if __name__ == "__main__":
    update = MalUpdate()
