import pandas as pd
import numpy as np
import os
from tensorflow.keras.utils import to_categorical

# Verilerin kaydedileceği klasör oluştur.
if not os.path.exists('numpyDatas'):
    os.makedirs('numpyDatas')

# CSV verisini oku
data = pd.read_csv('fer2013.csv',',',skiprows=0)

# Verilerin pixel ve etiket bilgilerini ayrıştır
x_data = data['pixels']
y_data = data['emotion']

pixels = []
# fer2013 içerisindeki her bir pixel verisi için 48x48x1 boyutunda bir diziye aktar ve bu diziyi pixeller dizisinde biriktir.
for row in x_data:
    arr = row.split(' ')
    arr = np.reshape(arr,(48,48,1))#FOR 2D
    pixels.append(arr)

# Düzenlenmiş pixel datalarını yeni x_data olarak güncelle
x_data = pixels
# Pixeller dizisini 8 bit işaretsiz tamsayı veri tipine sahip numpy dizisine çevir.
x_data = np.array(x_data,dtype='uint8')
# Etiket verisini kategorik veriye çevir.
y_data = to_categorical(y_data)
# Etiket adedini yazdır
print("Number of labels : "+str(len(y_data[0])))

# Diziler numpyDatas içerisine kaydedilir.
np.save('numpyDatas/x_data', x_data)
np.save('numpyDatas/y_data', y_data)