import pandas as pd
import requests
from bs4 import BeautifulSoup
from ultralytics import YOLO
import os
import re
import cv2

# CSV dosyasından linkleri oku
def csvden_linkleri_oku(dosya_adi):
    return pd.read_csv(dosya_adi)

# Sayfadaki resim URL'sini bul
def resim_url_bul(sayfa):
    img_etiketi = sayfa.find('img', src=re.compile(r'.*\.jpg'))
    return img_etiketi['src'] if img_etiketi else None

# Resmi indir ve kaydet
def resmi_indir_ve_kaydet(img_url, dosya_adi):
    img_verisi = requests.get(img_url).content
    with open(dosya_adi, 'wb') as dosya:
        dosya.write(img_verisi)

# YOLO modelini yükle
def yolo_modeli_yukle(model_dosya):
    return YOLO(model_dosya)

# Tespit sonuçlarını işle
def tespit_sonuclari_isle(sonuclar, img_dosya_adi, model):
    tespitler = []
    for sonuc in sonuclar:
        kutular = sonuc.boxes
        if kutular is None or len(kutular) == 0:
            print(f"{img_dosya_adi} içinde nesne tespit edilmedi")
            continue
        for kutu in kutular:
            sinif = int(kutu.cls.cpu().numpy())
            sinif_adi = model.names[sinif]
            tespitler.append(sinif_adi)
            print(f"{img_dosya_adi} içinde {sinif_adi} tespit edildi")
    return tespitler

# Sonuçları kaydet ve görselleştir
def sonuclari_kaydet_ve_gorsellestir(sonuclar, dosya_adi):
    sonuc_yolu = f"sonuc_{dosya_adi}.jpg"
    sonuclar[0].save(sonuc_yolu)
    print(f"{sonuc_yolu} olarak kaydedildi")

    # OpenCV ile kaydedilen sonucu göster
    resim = cv2.imread(sonuc_yolu)
    if resim is not None:
        cv2.imshow(f"Tespit Sonucu {dosya_adi}", resim)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print(f"{sonuc_yolu} yüklenemedi")

# Ana iş akışı
def ana_islem():
    veriler = csvden_linkleri_oku("linkler.csv")
    model = yolo_modeli_yukle("yolo11n.pt")
    tespit_sonuclari = []

    for indeks, satir in veriler.iterrows():
        url = satir['linkler']
        print(f"{url} işleniyor...")

        try:
            cevap = requests.get(url)
            sayfa = BeautifulSoup(cevap.content, 'html.parser')
            img_url = resim_url_bul(sayfa)

            if img_url:
                img_dosya_adi = f"resim_{indeks}.jpg"
                resmi_indir_ve_kaydet(img_url, img_dosya_adi)
                print(f"{img_dosya_adi} olarak kaydedildi")

                sonuclar = model([img_dosya_adi])
                tespitler = tespit_sonuclari_isle(sonuclar, img_dosya_adi, model)

                if tespitler:
                    tespit_sonuclari.extend([{'url': url, 'sinif': tespit} for tespit in tespitler])

                sonuclari_kaydet_ve_gorsellestir(sonuclar, indeks)
            else:
                print(f"{url} üzerinde resim bulunamadı")

        except Exception as e:
            print(f"{url} işlenirken hata oluştu: {str(e)}")

    print("Tespit işlemi tamamlandı.")

# İşlemi başlat
if __name__ == "__main__":
    ana_islem()
