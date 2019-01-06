import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import h5py

f_h5 = 'latent_gender_and_emotion_training.h5'
with h5py.File(f_h5, 'r') as h5:
    X = h5['Z'][...]
    y = h5['woman'][...]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42)

'''

from keras import backend as K
from keras import callbacks
from keras import layers
from keras import models
from keras.wrappers.scikit_learn import KerasClassifier

# Use Tenserflow backend
sess = tf.Session()
K.set_session(sess)

def model():
    act = None
    model = models.Sequential([
        layers.Dense(256, input_dim=X_train.shape[1], activation=act),
        layers.Dense(128, activation=act),
        layers.Dense(64, activation=act),
        layers.Dense(1, activation='sigmoid')
    ])
    loss = 'mean_squared_error'
    model.compile(loss=loss, optimizer='adam', metrics=['accuracy'])
    return model

clf = KerasClassifier(
    build_fn=model, nb_epoch=1000, batch_size=6,
    validation_split=0.2,
)

clf.fit(X_train, y_train)
yp = clf.predict(X_test).ravel()
'''

import pylab as plt
import seaborn as sns
sns.distplot(y_test)
sns.distplot(y_train)

plt.show()

import pylab as plt
_x = np.linspace(0,1)
plt.plot(_x, _x, 'r--', lw=2, alpha=0.25)
plt.scatter(yp, y_test)
plt.xlabel("Predicted")
plt.ylabel("GAN truth")
plt.title(f"Predicting gender from latent vector")
plt.show()
