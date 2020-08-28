"""
Aktarmalı öğrenme için kullanılabilir.
"""

from tensorflow.keras.models import load_model,save_model
import numpy as np
x_train = np.load('x_train.npy')
x_valid = np.load('x_valid.npy')
y_train = np.load('y_train.npy')
y_valid = np.load('y_valid.npy')
#%%
model = load_model('model_v3')
model.fit(x_train,y_train,
          batch_size=64,
          epochs=90,
          verbose=1,
          validation_data=(x_valid,y_valid),
          shuffle=True)
#%%
model.save('model_v3_transfer')