# 🔢 MNIST MLP: From Scratch with NumPy

Bu proje, derin öğrenme kütüphaneleri (PyTorch, TensorFlow vb.) kullanılmadan, bir sinir ağının temel mekanizmalarını anlamak amacıyla **sıfırdan (from scratch)** inşa edilmiştir. Sadece **NumPy** kullanılarak matris operasyonları, ileri yayılım ve geri yayılım süreçleri manuel olarak kodlanmıştır.

## 🧠 Projenin Amacı ve Arka Plan
Yüksek seviyeli kütüphaneler karmaşık modelleri saniyeler içinde kurmamıza olanak tanısa da, bu proje "kaputun altındaki" matematiğe odaklanır. Amaç; gradyan inişi (gradient descent), aktivasyon fonksiyonlarının türevleri ve ağırlık güncellemelerinin ardındaki lineer cebir ve kalkülüs süreçlerini bizzat deneyimlemektir.

### İşin Mutfağı: Matematiksel Mantık

Sinir ağının öğrenme süreci üç ana aşamada manuel olarak kurgulanmıştır:

#### 1. İleri Yayılım (Forward Propagation)
Veri giriş katmanından başlar ve her katmanda bir ağırlık matrisi ($W$) ile çarpılıp sapma ($b$) değeriyle toplanır: $z = XW + b$.
Sistemi lineerlikten kurtarmak için şu aktivasyon fonksiyonları kullanılmıştır:
* **ReLU (Rectified Linear Unit):** Gizli katmanlarda kullanıldı. Pozitif değerleri olduğu gibi geçirirken, negatifleri sıfırlayarak ağın karmaşık örüntüleri öğrenmesini sağlar.
* **Softmax:** Çıktı katmanında, modelin tahminlerini 10 sınıf (0-9 rakamları) için bir olasılık dağılımına dönüştürür.

#### 2. Kayıp Fonksiyonu (Cross-Entropy Loss)
Modelin tahmininin gerçek değerden ne kadar uzak olduğunu ölçmek için **Cross-Entropy** kullanılmıştır. İstatistiksel olarak bu, modelin tahmin ettiği olasılık dağılımı ile gerçek dağılım arasındaki farkı (divergence) hesaplar.

#### 3. Geri Yayılım (Backpropagation) - Öğrenmenin Gerçekleştiği Yer
Zincir kuralı (Chain Rule) kullanılarak, kaybın her bir ağırlık ve sapma üzerindeki etkisi (gradyanlar) hesaplanmıştır. 
* **Türev Hesaplama:** ReLU ve Softmax fonksiyonlarının türevleri manuel olarak kodlanarak hata sinyalinin geriye doğru akması sağlanmıştır.
* **Gradyan İnişi:** Ağırlıklar, öğrenme oranı (learning rate) doğrultusunda bu gradyanların tersi yönünde güncellenerek hata minimize edilmiştir.

## 🛠️ Teknik Mimari
Model, 3 katmanlı bir **Multilayer Perceptron (MLP)** yapısına sahiptir:

* **Giriş Katmanı:** 784 nöron (28x28 piksellik görüntüler).
* **Gizli Katman 1:** 128 nöron + ReLU.
* **Gizli Katman 2:** 64 nöron + ReLU.
* **Çıktı Katmanı:** 10 nöron (0-9 rakamları) + Softmax.

## 📈 Performans ve Analiz
Model, MNIST veri seti üzerinde yapılan testlerde **%97.4** doğruluk (accuracy) oranına ulaşmıştır. 

### İstatistiksel Hata Analizi
Notebook içerisinde, modelin hangi rakamları neden karıştırdığına dair bir analiz bölümü bulunmaktadır. Örneğin, modelin 5 rakamını neden bazen 3 olarak algıladığı, piksellerin benzerliği üzerinden görselleştirilerek incelenmiştir. Bu, modelin sınırlarını ve veri üzerindeki etkisini anlamak adına kritik bir adımdır.

## 📁 Proje İçeriği
* `mnist_mlp.ipynb`: Verinin yüklenmesinden gradyan hesaplamalarına ve hata analizine kadar tüm süreci içeren ana çalışma dosyası.
* **Veri Önişleme:** İkili (binary) formattaki MNIST verilerinin NumPy dizilerine dönüştürülmesi ve 0-1 arasına normalize edilmesi.

## 🚀 Kurulum ve Çalıştırma
1.  Bu kodu çalıştırmak için MNIST veri setini [buradan](https://www.kaggle.com/datasets/hojjatk/mnist-dataset?resource=download) indirip ana dizine atın
2.  Gerekli kütüphaneleri yükleyin(büyük bir olasılıkla yüklüdür.):
    ```bash
    pip install numpy matplotlib
    ```
3.  `mnist_mlp.ipynb` dosyasını Jupyter Notebook ortamında açın ve hücreleri sırasıyla çalıştırın.

---
*Bu çalışma, yapay zekanın arkasındaki matematiksel temelleri anlama ve bu teorik bilgiyi pratik koda dökme yolculuğumun bir parçasıdır.*
