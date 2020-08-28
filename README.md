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

