import numpy as np
from sklearn.model_selection import train_test_split

x = np.load('numpyDatas/x_data.npy')
y = np.load('numpyDatas/y_data.npy')

# Veri %70 eğitim, %12 test ve %18 doğrulama olarak bölümlenir.
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
X_test, X_valid, y_test, y_valid = train_test_split(X_test, y_test, test_size=0.6, random_state=41)


# Sonradan kullanılmak üzere veriler kaydedilir.
np.save('numpyDatas/x_train', X_train)
np.save('numpyDatas/y_train', y_train)

np.save('numpyDatas/x_test', X_test)
np.save('numpyDatas/y_test', y_test)

np.save('numpyDatas/x_valid', X_valid)
np.save('numpyDatas/y_valid', y_valid)

