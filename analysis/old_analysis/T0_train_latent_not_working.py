#import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
#from  sklearn.ensemble import RandomForestRegressor as model
from sklearn.ensemble import RandomForestClassifier as model
import h5py

f_h5 = 'latent_gender_and_emotion_training.h5'
with h5py.File(f_h5, 'r') as h5:
    X = h5['Z'][...]
    y = h5['woman'][...]

#idx = (y < 0.4) | (y>0.95)
#X = X[idx]
#y = y[idx]
y = y > 0.8

#import random
#random.shuffle(y)

print(len(y), y.mean())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42)

clf = model(n_estimators=400, n_jobs=-1)
clf.fit(X_train, y_train)
yp = clf.predict(X_test)

score = clf.score(X_test, y_test)
print(score)
exit()

import pylab as plt
_x = np.linspace(0,1)
plt.plot(_x, _x, 'r--', lw=2, alpha=0.25)
plt.scatter(yp, y_test)
plt.xlabel("Predicted")
plt.ylabel("GAN truth")
plt.title(f"Predicting gender from latent vector {score:0.3f}")

plt.show()


print(Z)
print(y)
