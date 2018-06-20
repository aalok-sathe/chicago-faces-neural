from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam

import matplotlib.pyplot as plt
from progressbar import progressbar

import json
import os
import time
import datetime
import numpy as np
import get_face

class ACGAN():
    def __init__(self):
        # Input shape
        self.img_rows = 32
        self.img_cols = 32
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        self.num_genders = 2
        self.num_races = 4
        self.num_emotions = 5

        self.num_classes = 2
        self.latent_dim = 100

        self.timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d_%H%M')
        self.face_db = get_face.face_provider()
        self.face_db.load_from_pickle()

        optimizer = Adam(0.0002, 0.5)
        losses = ['binary_crossentropy',
                    'sparse_categorical_crossentropy',
                    'sparse_categorical_crossentropy',
                    'sparse_categorical_crossentropy']

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss=losses,
            optimizer=optimizer,
            metrics=['accuracy'])

    def build_discriminator(self):

        img = Input(shape=self.img_shape)

        model = Sequential()

        model.add(Conv2D(16, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(32, kernel_size=4, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0,1),(0,1))))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(64, kernel_size=4, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(128, kernel_size=4, strides=1, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Flatten())

        # Extract feature representation
        features = model(img)

        # Determine validity and label of the image
        validity = Dense(1, activation="sigmoid")(features)
        gender = Dense(self.num_genders+1, activation="softmax")(features)
        race = Dense(self.num_races+1, activation="softmax")(features)
        emotion = Dense(self.num_emotions+1, activation="softmax")(features)

        model.summary()

        return Model(img, [validity, race, gender, emotion])

    def train(self, epochs, batch_size=128, sample_interval=50):

        # Load the dataset
        (X_train, y_train), (X_test, y_test) = self.face_db.load_data(grayscale=(self.channels==1), resize=(32,32))

        # Configure inputs
        # X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        X_train = np.expand_dims(X_train, axis=3)
        X_test = np.expand_dims(X_test, axis=3)

        y_train = y_train.reshape(-1, 3, 1)
        y_test = y_test.reshape(-1, 3, 1)

        X_train = X_train.reshape(y_train.shape[0],self.img_rows,self.img_cols,self.channels)
        X_test = X_test.reshape(y_test.shape[0],self.img_rows,self.img_cols,self.channels)

        print(X_train.shape, y_train.shape, '\n', X_test.shape, y_test.shape)

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in progressbar(range(epochs), redirect_stdout=True):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random batch of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]

            # Image labels. 0-1 if image is valid or 2 if it is generated (fake)
            img_labels = y_train[idx]
            fake_labels = 0 * np.ones(img_labels.shape)

            # print(np.array([np.array([a,*b]) for (a,b) in [*zip(valid, img_labels)]]))
            # print(np.array([img_labels]))
            # print(np.array([valid,
            #   np.array([img_labels])[:,:,0,][0],
            #   np.array([img_labels])[:,:,1,][0],
            #   np.array([img_labels])[:,:,2,][0]
            # ]))

            # Train the discriminator
            d_loss = self.discriminator.train_on_batch(imgs,
                [
                  valid,
                  np.array([img_labels])[:,:,0,:][0],
                  np.array([img_labels])[:,:,1,:][0],
                  np.array([img_labels])[:,:,2,:][0],
                ]
            )
            # d_loss_fake = self.discriminator.train_on_batch(fake,
            #     [fake,
            #       np.array([fake_labels])[:,:,0,][:],
            #       np.array([fake_labels])[:,:,1,][:],
            #       np.array([fake_labels])[:,:,2,][:]
            #     ]
            # )
            # d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%, op_acc: %.2f%%]" % (epoch, d_loss[0], 100*d_loss[3], 100*d_loss[4]))

            # If at save interval => save generated image samples
            runtime_params = dict()
            if not epoch%5:
                with open("runtime.kerasconfig", 'r') as file:
                    runtime_params = json.load(file)
                    sample_interval = runtime_params.get("sample_every",
                                                          sample_interval)

            # if epoch % sample_interval == 0:
                # self.save_model()
                # self.sample_images(epoch=epoch,
                #                     cmap=runtime_params.get("cmap", "gray"))
            if epoch >= runtime_params.get("num_epochs", epochs):
                break

        print(self.discriminator.evaluate(
        X_test[:],
                [
                  valid,
                  np.array([y_test[:]])[:,:,0,:][0],
                  np.array([y_test[:]])[:,:,1,:][0],
                  np.array([y_test[:]])[:,:,2,:][0],
                ]
            )
        )

    def save_model(self):

        def save(model, model_name):
            model_path = "saved_model/%s.json" % model_name
            weights_path = "saved_model/%s_weights.h5" % model_name
            options = {"file_arch": model_path,
                        "file_weight": weights_path}
            json_string = model.to_json()
            open(options['file_arch'], 'w').write(json_string)
            model.save_weights(options['file_weight'])

        # save(self.generator, "generator%s"%self.timestamp)
        save(self.discriminator, "labeler%s"%self.timestamp)


if __name__ == '__main__':
    acgan = ACGAN()
    acgan.train(epochs=2048, batch_size=32, sample_interval=10)
