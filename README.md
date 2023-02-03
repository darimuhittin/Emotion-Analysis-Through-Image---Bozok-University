# Emotion Analysis

*The file *fer2013.csv must be located in the main folder.

---

## DATA CREATION
	
1) Convert data to numpy array with RawDataCompose.py
2) Split the saved numpy data from the previous step with TrainTestDataCreate.py.
    *NOTE: The generated data will be saved in the folder named numpyDatas.*

---

## MODEL TRAINING - (A READY-RECORDED MODEL IS AVAILABLE, WHICH CAN BE RE-RUN AND TRAINED IF DESIRED)
	
1) Run ModelEgit.py from the command line with the number of epochs as a parameter.
	*NOTE : python ModelEgit.py 50*

---

## LIVE APPLICATION -

1) Run CanliVideoUygulamasi.py.

---

## EXTERNAL CODES -
1) You can extract the data from fer2013.csv file with extract data from csv.py.
	*It extracts PrivateTest PublicTest and Train to the folder where the project is located.
2) There is a small code in TransferLearning.py to train on the model.


******************************************************************************************************

# Duygu Analizi

*fer2013.csv dosyasının ana klasörde bulunması gerekir.*

---

## VERİ OLUŞTURMA
	
1) HamVeriOlustur.py ile verileri numpy dizisine çevir
2) TrainTestVeriOlustur.py ile bir önceki adımdaki kaydedilen numpy verisini böl.
    *NOT: Oluşturulan veriler numpyDatas adlı klasöre kaydedilecektir.*

---

## MODEL EĞİTME - (HAZIR KAYITLI MODEL MEVCUTTUR, EĞER DİLENİRSE TEKRAR ÇALIŞTIRILIP EĞİTİLEBİLİR)
	
1) ModelEgit.py 'ı komut satırından epoch sayısını parametre ile vererek çalıştır.
	*ÖRN : python ModelEgit.py 50*

---

## CANLI UYGULAMA -

1) CanliVideoUygulamasi.py 'ı çalıştır.

---

## HARİCİ KODLAR -
1) VerileriCsvdencikar.py ile verileri fer2013.csv dosyasından ayıklayabilirsiniz.
	*Projenin bulunduğu klasöre PrivateTest PublicTest ve Train olmak üzere çıkarır.*
2) TransferLearning.py içerisinde modelin üzerine eğitmek için ufak bir kod bulunmakta.

