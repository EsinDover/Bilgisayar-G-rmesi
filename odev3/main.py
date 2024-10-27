import cv2
import numpy as np
import matplotlib.pyplot as plt

# Kare fotoğrafı yükleme
img = cv2.imread('kare.png')
#img = cv2.imread('elipse.png')

# Görüntünün yüklendiğini kontrol et
if img is None:
    print("Görüntü yüklenemedi. Lütfen dosya yolunu kontrol edin.")
    exit()

# Fotoğrafı gri tonlamaya çevirme
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Yatay türev filtresi [-1, 1]
horizontal_filter = np.array([[-1, 1]])

# Yatay türevi hesaplama
horizontal_derivative = cv2.filter2D(gray_img, cv2.CV_64F, horizontal_filter)

# Dikey türev filtresi [-1, 1]T
vertical_filter = np.array([[-1], [1]])

# Dikey türevi hesaplama
vertical_derivative = cv2.filter2D(gray_img, cv2.CV_64F, vertical_filter)

# Türev büyüklüğünü hesaplama
magnitude = np.sqrt(np.square(horizontal_derivative) + np.square(vertical_derivative))

# Normalizasyon
horizontal_derivative = cv2.normalize(horizontal_derivative, None, 0, 255, cv2.NORM_MINMAX)
horizontal_derivative = horizontal_derivative.astype(np.uint8)

vertical_derivative = cv2.normalize(vertical_derivative, None, 0, 255, cv2.NORM_MINMAX)
vertical_derivative = vertical_derivative.astype(np.uint8)

magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
magnitude = magnitude.astype(np.uint8)

# Görüntüleri yan yana gösterme
plt.figure(figsize=(20, 5))

# Orijinal fotoğraf
plt.subplot(1, 4, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Orijinal Fotoğraf')
plt.axis('off')

# Yatay türev
plt.subplot(1, 4, 2)
plt.imshow(horizontal_derivative, cmap='gray')
plt.title('Yatay Türev')
plt.axis('off')

# Dikey türev
plt.subplot(1, 4, 3)
plt.imshow(vertical_derivative, cmap='gray')
plt.title('Dikey Türev')
plt.axis('off')

# Kenar büyüklüğü (magnitude)
plt.subplot(1, 4, 4)
plt.imshow(magnitude, cmap='plasma')
plt.title('Yatay ve Dikey Türevlerin Birleşimi')
plt.axis('off')

plt.tight_layout()
plt.show()
