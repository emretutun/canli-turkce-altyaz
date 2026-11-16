# Canlı Türkçe Altyazı

Televizyon yayınları için Türkçe konuşmayı gerçek zamanlı olarak altyazıya çeviren ve
**Vizrt Viz Engine**'e basan bir prototip. Tamamen yerel çalışır: ses dışarı gönderilmez,
bulut servisi ve lisans ücreti gerekmez.

```
Stüdyo sesi ──> Ses girişi ──> Whisper (GPU) ──> Kesinleşen kelimeler ──> Viz Engine (satir1 / satir2)
                                   │                                  └──> Tarayıcı önizleme
                                   └──> Konuşmacı ayırma (A: / B:)
```

## Özellikler

- **Canlı altyazı:** ~2 sn gecikme. Yazı titremez; kelime yalnızca iki tanıma turunda aynı çıkınca ekrana gider.
- **Viz Engine çıkışı:** 6100 komut portu üzerinden sahnedeki iki Text container'ı günceller. Roll-up (alttan yukarı kayan) iki satır.
- **Konuşmacı ayırma:** Tek kanallı seste ses parmak izi ile kişileri `A:`, `B:` diye ayırır. Kimseyi önceden tanıtmak gerekmez.
- **Dosyadan altyazı:** MP4/MXF/WAV'dan SRT üretir. RTX 3050'de gerçek zamanın ~17 katı hızlı.
- **Test modu:** Bir kaydı canlı yayın hızında besleyip videoyu aynı anda oynatır; stüdyo olmadan demo yapılabilir.
- Türkçe eklerin bitişik yazımı (`Abraham'ı`, `8,5`), Whisper'ın bilinen Türkçe uydurmalarının ("Altyazı M.K.") filtrelenmesi.

## Gereksinimler

- Windows 10/11, NVIDIA ekran kartı (RTX 3050 8 GB ile test edildi)
- Python 3.12
- ffmpeg (`winget install Gyan.FFmpeg`) — `--oynat` için `ffplay` gerekir
- Viz Artist / Viz Engine 3.x (Viz çıkışı için; 3.14.4 ile test edildi)

## Kurulum

```powershell
pip install -r requirements.txt

# Konuşmacı ayırma modeli (~26 MB)
mkdir modeller
Invoke-WebRequest "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_eres2net_sv_en_voxceleb_16k.onnx" -OutFile modeller\konusmaci_eres2net.onnx
```

Whisper modeli (`large-v3-turbo`, ~1,6 GB) ilk çalıştırmada otomatik indirilir.

## Viz sahnesi

Viz Artist'te yeni bir sahne oluşturun:

1. **Server → FONTS** klasöründen Türkçe karakter destekleyen bir fontu Tree'ye sürükleyin, container adını `satir1` yapın.
2. Aynı fontu tekrar sürükleyin, adını `satir2` yapın.
3. İkisini de ortala (Text editöründe üst sıradaki 2. düğme), Scaling ~0.6, `satir1` üstte `satir2` altta olacak şekilde safe area içine yerleştirin.
4. Kaydedip **On Air** moduna geçin.

## Kullanım

```powershell
# Kaydı canlı gibi oynat, altyazıyı Viz'e bas, konuşmacıları ayır
python canli_altyazi.py --dosya kayit.mp4 --oynat --viz localhost --konusmaci

# Mikrofon / ses kartı girişinden canlı
python canli_altyazi.py --liste          # girişleri listele
python canli_altyazi.py --cihaz 6 --viz localhost --konusmaci

# Dosyadan SRT üret
python mp4_altyazi.py kayit.mp4
python mp4_altyazi.py kayit.mp4 --isimler isimler.txt   # özel isim ipuçları
```

Altyazı önizlemesi: http://localhost:8765 — her çalıştırmada `canli_kayit_*.txt` dökümü oluşur.

### Önemli parametreler

| Parametre | Açıklama |
|---|---|
| `--viz <adres>` | Viz Engine adresi (`localhost`, `10.0.0.5` …) |
| `--viz-katman` | `MAIN_LAYER` (varsayılan), `FRONT_LAYER`, `BACK_LAYER` |
| `--viz-kodlama cp1254` | Türkçe harfler Viz'de bozuk çıkarsa |
| `--konusmaci-esik` | 0.3 varsayılan. Aynı kişi iki harf alıyorsa düşürün, farklı kişiler aynı harfi alıyorsa yükseltin |
| `--kanal` | Çok kanallı ses kartında kullanılacak kanal |
| `--beam 5` | Daha doğru ama daha yavaş tanıma (özel isimlerde belirgin fark eder) |
| `--isimler isimler.txt` | Özel isim listesi. Whisper'a ipucu verir ve yanlış yazılan isimleri düzeltir |

### Özel isimler (oyuncu, yabancı isimler)

`isimler.txt` dosyasına her satıra bir isim yazılır. Whisper bir ismi hep aynı şekilde
yanlış yazıyorsa, yanlış hâller `=` sonrasına virgülle eklenir:

```
Torreira = Torayra, Toreyra
Oh Hyeon-gyu = Oyan Yuh, Oiyanyu
```

İsimler Whisper'a `hotwords` olarak verilir. Ayrıca tanıma sonrası bilinen yanlış hâller ve
isme çok benzeyen büyük harfli kelimeler düzeltilir, ekler korunur (`Torayra'nın` → `Torreira'nın`).
Yanlış hâller `canli_kayit_*.txt` dökümlerine bakılarak eklenebilir.

### Satır düzeni

Konuşmacı ayırma açıkken varsayılan düzen **her satır bir konuşmacı**dır: `satir1` ilk konuşanın,
`satir2` ikinci konuşanın satırıdır, iki kişinin sözü aynı satırda karışmaz. Satır dolunca o
konuşmacının satırı baştan yazılır; sıra bir kişiye geçince sözü temiz satırda başlar; 8 sn susan
kişinin satırı silinir (`--satir-bekleme`). Üçüncü bir konuşmacı en uzun süredir susan kişinin
satırını geçici olarak kullanır, sahibi konuşunca satır ona döner.

Eski alttan yukarı kayan düzen için `--satir-duzeni kayan` (başlatma ekranında "Her satır bir
konuşmacı" kutusunu kaldırmak). Viz'de yazı küçültülürse `--satir-uzunlugu` (ekranda "Satır
uzunluğu") artırılarak bir satıra daha çok kelime sığdırılabilir.

### Konuşmacılara isim verme

Harfler konuşma sırasına göre verilir (ilk konuşan A). Başlatma ekranındaki **Konuşmacı isimleri**
kutularına `A = CEM`, `B = KONUK` gibi isim yazılınca, yayın sürerken bile Viz'deki ve önizlemedeki
satırlar yarım saniye içinde `CEM: ...` olarak güncellenir. Komut satırında aynı şey
`--konusmaci-adlari konusmaci_isimleri.json` ile yapılır; program dosyayı değiştikçe yeniden okur.

Komut yazmadan çalıştırmak için `Canli Altyazi.bat` dosyasına çift tıklanır: video ve isim
listesi seçilip **Başlat**'a basılır.

## Nasıl çalışır

- **Akışlı tanıma:** Whisper akışlı çalışmaz. Ses tamponu her saniye yeniden tanınır; art arda iki turda aynı çıkan kelimeler kesinleşir (LocalAgreement). Konuşma durunca bekleyen kelimeler de kesinleşir.
- **Konuşmacı ayırma:** Konuşma 1 sn arayla 2 sn'lik pencerelere bölünür, her pencereden [ERes2Net](https://github.com/modelscope/3D-Speaker) ile ses parmak izi çıkarılır ve bilinen seslerle kosinüs benzerliğiyle karşılaştırılır. Yeni kişi, art arda iki pencerede duyulmadan açılmaz.

## Bilinen sınırlamalar

- **Üst üste konuşma:** Tek kanallı seste baskın konuşan yazılır, diğeri kaçabilir. Kesin çözüm her mikrofonu ayrı kanaldan almaktır.
- **Kısa araya girişler** ("Evet.", "Doğru.") yanlış kişiye yazılabilir.
- **Konuşmacı harfleri** her çalıştırmada sıfırdan verilir; isimle eşleşmez.
- **Özel isimler** (oyuncu, yabancı isimler) hatalı yazılabilir; `--isimler` ile iyileşir.
- Operatör onay/düzeltme ekranı ve teletext (Newfor) çıkışı henüz yok.

## Yol haritası

- [ ] Çok kanallı mikrofon girişi ve kanal → isim eşleştirme
- [ ] Sabit sunucular için ses tanıtma (enrollment)
- [ ] Web tabanlı operatör düzeltme paneli
- [ ] Newfor / teletext kapalı altyazı çıkışı
- [ ] Rundown'dan otomatik isim sözlüğü
