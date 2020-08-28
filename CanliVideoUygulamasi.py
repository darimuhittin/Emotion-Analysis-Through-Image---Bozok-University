import numpy as np
import cv2
import tensorflow as tf

# Kaydedilen model alınır.

model = tf.keras.models.load_model('model_v3')

# openCL kullanımı ve gereksiz mesajlar engellenir.
cv2.ocl.setUseOpenCL(False)

# Yüz sınıflandırıcı nesnesi oluşturulur.
faceClassifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# VideoCapture nesnesi oluşturulur. ( 0 varsayılan kamera flag )
videoCapture = cv2.VideoCapture(0)



# Verilen görüntüyü, verilen model ile tahmin işlemine tabi tutacak fonksiyon oluşturulur
def PredictImageWithModel(image, model):
    image = np.array(image)
    image = np.reshape(image, (48, 48, 1))
    image = np.expand_dims(image, axis=0)
    res = model.predict(image)
    # print('Tahmin işlemi tamamlandı.')
    return res



while True:
    # Sıradaki çerçeve okunur.
    ret, frame = videoCapture.read()

    # Çerçeve alınmış ise
    if (ret):

        # Yüz tespit et
        faces = tuple(faceClassifier.detectMultiScale(frame))

        # Yüz bulundu ise
        if (len(faces) != 0):

            # Her yüz için
            for (x, y, w, h) in faces:

                # Yüzün bulunduğu kısmı kes.
                framePredict = frame[y:y + h, x:x + w]

                # 48x48 ve gri skala resim haline getir.
                framePredict = cv2.resize(framePredict, (48, 48))
                framePredict = cv2.cvtColor(framePredict, cv2.COLOR_BGR2GRAY)

                # Tahmin edilen çerçeveyi görüntüle.
                cv2.imshow('Tahmin edilen çerçeve',framePredict)

                # Tahmin gerçekleştir.
                sonuc = PredictImageWithModel(framePredict, model)
                skor = np.max(sonuc)

                if skor > 0.8:

                    sonucInd = np.argmax(sonuc)

                    if sonucInd == 0:
                        sonucString = 'Sinirli'
                    elif sonucInd == 1:
                        sonucString = 'Tiksinmis'
                    elif sonucInd == 2:
                        sonucString = 'Korkmus'
                    elif sonucInd == 3:
                        sonucString = 'Mutlu'
                    elif sonucInd == 4:
                        sonucString = 'Uzgun'
                    elif sonucInd == 5:
                        sonucString = 'Sasirmis'
                    elif sonucInd == 6:
                        sonucString = 'Normal'

                    # Orijinal frame'de yüz üzerine dörtgen çiz ve gerekli sonuç yazısını ekle.
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    cv2.putText(frame, sonucString, (x, y - 5), cv2.FONT_HERSHEY_TRIPLEX, 2, (0, 255, 0))
            # Her yüz için SON

        # Yüz bulundu ise SON

        # Sonuç çerçeveyi goster
        cv2.imshow('Cam', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        print("Video capture olusturmada hata - (Kamera bulunamamış olabilir.)")
        break

# İşlem bittiğinde kaynakları serbest bırak.
videoCapture.release()
cv2.destroyAllWindows()
