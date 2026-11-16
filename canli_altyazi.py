"""
Canlı Türkçe altyazı.

Mikrofondan / ses kartı girişinden (stüdyoda: ses masasının aux çıkışı) gelen sesi
anlık yazıya döker ve tarayıcıda yayın altyazısı gibi gösterir.

Kullanım:
    python canli_altyazi.py --liste               # ses girişlerini listele
    python canli_altyazi.py                       # varsayılan mikrofon
    python canli_altyazi.py --cihaz 6             # belirli bir giriş
    python canli_altyazi.py --cihaz 6 --kanal 2   # çok kanallı kartta 2. kanal
    python canli_altyazi.py --dosya kayit.mp4     # kaydı canlı yayın gibi gerçek zamanlı besle (test/demo)
    python canli_altyazi.py --dosya kayit.mp4 --oynat --viz localhost
                                                  # videoyu oynat + altyazıyı Viz Engine'e gönder
    python canli_altyazi.py --dosya kayit.mp4 --oynat --viz localhost --konusmaci
                                                  # + kimin konuştuğunu "A: / B:" diye ayır

Altyazı ekranı: http://localhost:8765
Viz sahnesinde "satir1" ve "satir2" adında iki Text container olmalı.
"""
import argparse
import json
import os
import queue
import re
import socket
import subprocess
import sys
import threading
import time
import webbrowser
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np


def nvidia_dll_yollarini_ekle():
    """pip ile gelen CUDA kütüphanelerini (cuBLAS, cuDNN) Windows'a tanıt."""
    if os.name != "nt":
        return
    import site
    for sp in site.getsitepackages() + [site.getusersitepackages()]:
        nvidia = Path(sp) / "nvidia"
        if not nvidia.is_dir():
            continue
        for bin_dir in nvidia.glob("*/bin"):
            os.add_dll_directory(str(bin_dir))
            os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ["PATH"]


nvidia_dll_yollarini_ekle()
from faster_whisper import WhisperModel, decode_audio  # noqa: E402

ORNEK_HIZI = 16000       # Whisper 16 kHz mono ister
SATIR_UZUNLUGU = 42      # altyazı satırı en fazla karakter
EN_KISA_PARCA = 1.0      # en az bu kadar yeni ses birikince tanıma yap (sn)
TAMPON_UST_SINIR = 15.0  # tampon bundan uzunsa kesinleşmiş kısmı at (sn)

# Whisper'ın Türkçede sessizlik/müzikte uydurduğu bilinen cümleler
UYDURMALAR = [
    "altyazı m.k", "altyazı m. k", "izlediğiniz için teşekkür",
    "abone olmayı unutmayın", "bir sonraki videoda görüşmek",
]


def sade(k: str) -> str:
    """Kelimeleri karşılaştırmak için noktalama ve büyük harfleri at."""
    return re.sub(r"[^\w]", "", k.lower())


def yapisik(k: str) -> bool:
    """Whisper "Abraham'ı" ve "8,5"i parçalara böler ('ı / ,5). Önünde boşluk olmayan
    ek ve noktalama parçaları önceki kelimeye bitişik yazılmalı."""
    return not k.startswith(" ") and bool(re.match(r"['’,.;:!?%)\-]", k))


def birlestir(kelimeler) -> str:
    metin = ""
    for k in kelimeler:
        if metin and not yapisik(k):
            metin += " "
        metin += k.strip()
    return metin


class IsimDuzeltici:
    """
    Özel isimleri (oyuncu, siyasetçi, yabancı isimler) doğru yazdırır.

    isimler.txt biçimi (her satır bir isim, # ile başlayan satır açıklama):
        Torreira
        Oh Hyeon-gyu = Oyan Yuh, Oiyanyu
    "=" sonrası, Whisper'ın o ismi yanlış yazdığı görülmüş hâlleridir (virgülle).

    İki katmanda çalışır:
      1. İsimler Whisper'a "hotwords" olarak verilir, model doğru yazmaya yönelir.
      2. Tanıma sonrası: bilinen yanlış hâller birebir, isme çok benzeyen büyük
         harfli kelimeler de benzerlikle düzeltilir. Ek korunur (Torayra'nın -> Torreira'nın).
    """

    BENZERLIK = 0.8   # 0-1; düşürürsen daha çok kelimeyi isme çevirir (yanlış düzeltme riski)
    EN_KISA = 5       # bundan kısa kelimeler benzerlikle düzeltilmez

    def __init__(self, yol):
        self.isimler = []          # doğru yazılışlar
        self.yanlislar = {}        # sade(yanlış hâl, boşluksuz) -> doğru isim
        self.parcalar = {}         # benzerlik için: sade(isim parçası) -> doğru parça
        self.butunler = {}         # çok parçalı isimler: anahtar ("ohhyeongyu") -> doğru isim
        self.en_uzun = 1           # en uzun yanlış hâl kaç kelime
        for satir in Path(yol).read_text(encoding="utf-8-sig").splitlines():
            satir = satir.split("#", 1)[0].strip()
            if not satir:
                continue
            dogru, _, yanlis = satir.partition("=")
            dogru = dogru.strip()
            if not dogru:
                continue
            self.isimler.append(dogru)
            parca_sayisi = len(re.split(r"[\s\-]+", dogru))
            if parca_sayisi > 1:
                self.butunler[self._anahtar(dogru)] = dogru
                self.en_uzun = max(self.en_uzun, parca_sayisi + 1)
            for y in yanlis.split(","):
                if y.strip():
                    self.yanlislar[self._anahtar(y)] = dogru
                    # Whisper kelimeyi hece hece de bölebilir; boşluk sayısına 2 pay bırak
                    self.en_uzun = max(self.en_uzun, len(y.split()) + 2)
            for parca in [dogru] + dogru.split():
                if len(sade(parca)) >= self.EN_KISA:
                    self.parcalar[sade(parca)] = parca

    @staticmethod
    def _anahtar(metin):
        return sade(metin.replace(" ", ""))

    def ipucu(self):
        """Whisper'a verilecek hotwords metni."""
        return ", ".join(self.isimler) or None

    def _benzer(self, kelime):
        import difflib
        temiz = kelime.strip()
        if len(sade(temiz)) < self.EN_KISA or not temiz[:1].isupper():
            return None
        if sade(temiz) in self.parcalar:
            return None  # zaten doğru
        en_iyi = difflib.get_close_matches(sade(temiz), self.parcalar, n=1, cutoff=self.BENZERLIK)
        return self.parcalar[en_iyi[0]] if en_iyi else None

    def _grup_eslesmesi(self, grup, benzerlikle):
        """Bir ya da birkaç kelimelik grup bir isme karşılık geliyor mu?"""
        anahtar = self._anahtar("".join(k[2] for k in grup))
        if not benzerlikle:
            return self.yanlislar.get(anahtar) or self.butunler.get(anahtar)
        if len(grup) > 1 and len(anahtar) >= 6 and grup[0][2].strip()[:1].isupper():
            # "O Hyeon -gyu" gibi parçalara bölünmüş çok kelimeli isim.
            # Uzunluk yakın olmalı; yoksa isimden sonraki kelimeyi de yutar.
            import difflib
            adaylar = [b for b in self.butunler if abs(len(b) - len(anahtar)) <= 2]
            en_iyi = difflib.get_close_matches(anahtar, adaylar, n=1, cutoff=self.BENZERLIK)
            if en_iyi:
                return self.butunler[en_iyi[0]]
        return None

    def duzelt(self, kelimeler):
        """[(bas, son, kelime)] listesini düzeltilmiş hâliyle döndür."""
        sonuc, i = [], 0
        while i < len(kelimeler):
            # 1) Bilinen yanlış hâl / parçalanmış isim (ör. "Oyan" "Yuh", "O" "Hyeon" "-gyu").
            #    Önce birebir eşleşme, bulunmazsa benzerlik denenir.
            eslesme = None
            for benzerlikle in (False, True):
                for n in range(min(self.en_uzun, len(kelimeler) - i), 0, -1):
                    grup = kelimeler[i:i + n]
                    # tire ("-gyu") ismin parçası olabilir; ek/noktalama ("'ta", ",") olamaz
                    if n > 1 and any(yapisik(k[2]) and not k[2].startswith("-") for k in grup[1:]):
                        continue
                    dogru = self._grup_eslesmesi(grup, benzerlikle)
                    if dogru:
                        eslesme = (grup, dogru)
                        break
                if eslesme:
                    break
            if eslesme:
                grup, dogru = eslesme
                bosluk = " " if grup[0][2].startswith(" ") else ""
                noktalama = re.search(r"[.,;:!?…]+$", grup[-1][2].strip())
                sonuc.append((grup[0][0], grup[-1][1],
                              bosluk + dogru + (noktalama.group() if noktalama else "")))
                i += len(grup)
            else:
                # 2) İsme çok benzeyen tek kelime
                bas, son, k = kelimeler[i]
                dogru = self._benzer(k)
                if dogru:
                    k = (" " if k.startswith(" ") else "") + dogru
                sonuc.append((bas, son, k))
                i += 1
        return sonuc


class CanliYazici:
    """
    Sürekli gelen sesi tamponda tutar, her turda tamponu yeniden tanır ve
    art arda iki turda aynı çıkan kelimeleri 'kesin' kabul eder (LocalAgreement).
    Kesinleşen kelime bir daha değişmez; ekrandaki yazı titremez.
    """

    def __init__(self, model, beam, sessizlik_esigi, duzeltici=None):
        self.model = model
        self.beam = beam
        self.esik = sessizlik_esigi
        self.duzeltici = duzeltici
        self.ses = np.zeros(0, dtype=np.float32)
        self.ses_baslangic = 0.0   # tampondaki ilk örneğin akış zamanı (sn)
        self.kesin = []            # [(bas, son, kelime)]
        self.onceki = []           # önceki turun kesinleşmemiş tahmini
        self.son_kesin_zaman = 0.0

    @property
    def alinan(self):
        """Başlangıçtan beri alınan toplam ses (sn)."""
        return self.ses_baslangic + len(self.ses) / ORNEK_HIZI

    def ekle(self, parca):
        self.ses = np.concatenate([self.ses, parca])

    def _sessiz_mi(self, sn):
        son = self.ses[-int(sn * ORNEK_HIZI):]
        return len(son) > 0 and float(np.sqrt(np.mean(son ** 2))) < self.esik

    def _yeni_kelimeler(self, ham):
        """Tamponun yeni tanımasından, ekrana zaten yazılmış kelimelerden SONRAKİLERİ seç.

        Zamana bakarak ayırmak kelime kaybettiriyordu: Whisper duraklamadan önceki son
        kelimenin bitişini bazen saniyelerce uzatır, o aralıkta gerçekten söylenen sözler
        "zaten yazıldı" sanılıp atılıyordu. Bunun yerine yazılmış son kelimeleri yeni
        tanımada metin olarak bulup sonrasını alıyoruz."""
        if not self.kesin:
            return ham
        son_bas = self.kesin[-1][0]
        for n in range(min(6, len(self.kesin), len(ham)), 0, -1):
            kuyruk = [sade(k[2]) for k in self.kesin[-n:]]
            for p in range(len(ham) - n + 1):
                if ([sade(h[2]) for h in ham[p:p + n]] == kuyruk
                        and abs(ham[p + n - 1][0] - son_bas) <= 1.0):
                    return ham[p + n:]
        # Yazılmış kelimeler yeni tanımada yok (tampon kesildi ya da Whisper farklı duydu).
        # Önceki turda bekleyen kelimeler yeni tanımada varsa oradan devam et: Whisper
        # tamponun başındaki kelimelere bazen yanlış zaman verir, zamana bakmak onları atar.
        if self.onceki:
            ilk = [sade(k[2]) for k in self.onceki[:2]]
            for p in range(len(ham) - len(ilk) + 1):
                if [sade(h[2]) for h in ham[p:p + len(ilk)]] == ilk:
                    return ham[p:]
        # Son çare: son yazılan kelimeden sonra başlayanlar
        return [h for h in ham if h[0] > son_bas + 0.15]

    def _tamponu_kirp(self):
        sure = len(self.ses) / ORNEK_HIZI
        if sure > TAMPON_UST_SINIR:
            # Son yazılan kelimenin (düzeltilmiş) bitişinden kes, ama bekleyen bir sonraki
            # kelimenin içine girme: kesilen yerdeki söz kaybolurdu
            kes_zaman = self.kesin[-1][1] if self.kesin else 0.0
            if self.onceki:
                kes_zaman = min(kes_zaman, self.onceki[0][0] - 0.05)
            kes = kes_zaman - self.ses_baslangic
            if kes <= 0 or sure - kes > TAMPON_UST_SINIR:
                # Uzun süredir kesinleşen yok (müzik, gürültü): son 5 sn'yi tut
                kes = sure - 5.0
                self.onceki = []
            kes_ornek = int(kes * ORNEK_HIZI)
            self.ses = self.ses[kes_ornek:]
            self.ses_baslangic += kes_ornek / ORNEK_HIZI

    def isle(self):
        """Tamponu tanı. Dönüş: (yeni kesinleşen kelimeler, bekleyen kelimeler)"""
        baglam = birlestir(k[2] for k in self.kesin[-30:]) or None
        segmentler, _ = self.model.transcribe(
            self.ses,
            language="tr",
            beam_size=self.beam,
            word_timestamps=True,
            vad_filter=True,
            initial_prompt=baglam,
            hotwords=self.duzeltici.ipucu() if self.duzeltici else None,
            condition_on_previous_text=False,
        )
        ham = []
        for seg in segmentler:
            if any(u in seg.text.lower() for u in UYDURMALAR):
                continue
            for w in seg.words or []:
                bas = self.ses_baslangic + w.start
                # Whisper'ın uzattığı bitişi makul bir süreye indir (konuşmacı ayırma ve
                # duraklama tespiti gerçek konuşma süresini görsün)
                son = min(self.ses_baslangic + w.end, bas + 0.3 + 0.1 * len(w.word.strip()))
                # boşluğu atmıyoruz: ekin kelimeye bitişik olup olmadığını o söylüyor
                ham.append((bas, son, w.word))
        if self.duzeltici:
            ham = self.duzeltici.duzelt(ham)
        yeni = self._yeni_kelimeler(ham)

        kesinlesen = []
        while yeni and self.onceki and sade(yeni[0][2]) == sade(self.onceki[0][2]):
            kesinlesen.append(yeni.pop(0))
            self.onceki.pop(0)
        self.onceki = yeni

        # Konuşmacı durduysa beklemeye gerek yok, kalanları da kesinleştir
        if yeni and self._sessiz_mi(1.2):
            kesinlesen += yeni
            yeni, self.onceki = [], []

        if kesinlesen:
            self.kesin += kesinlesen
            self.son_kesin_zaman = kesinlesen[-1][1]
        self._tamponu_kirp()
        return kesinlesen, yeni


# ---------------------------------------------------------------- konuşmacı ayırma

class KonusmaciAyirici:
    """
    Tek kanallı seste kimin konuştuğunu bulur. Konuşmayı 1 sn arayla 2 sn'lik
    pencerelere böler, her pencereden bir "ses parmak izi" (embedding) çıkarır ve
    bilinen seslerle karşılaştırır. Hiçbirine benzemeyen ses art arda iki pencerede
    duyulursa yeni konuşmacı açılır (tek pencerelik gürültü sahte kişi yaratmasın).
    """
    PENCERE = 2.0
    ADIM = 0.5     # yarım saniyede bir pencere: konuşmacı değişimi daha erken yakalanır
    GECMIS = 30.0  # bu kadar saniyelik ses ve etiket tutulur

    def __init__(self, model_yolu, esik, en_fazla, sessizlik_esigi):
        import sherpa_onnx
        ayar = sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(model_yolu), num_threads=2)
        self.cikarici = sherpa_onnx.SpeakerEmbeddingExtractor(ayar)
        self.esik = esik
        self.en_fazla = en_fazla
        self.sessizlik = sessizlik_esigi
        self.ses = np.zeros(0, dtype=np.float32)
        self.ses_baslangic = 0.0
        self.sonraki = 0.0      # işlenecek bir sonraki pencerenin başı (sn)
        self.merkezler = []     # her konuşmacının parmak izlerinin toplamı
        self.aday = None        # henüz onaylanmamış yeni ses
        self.etiketler = []     # [(pencere_ortası, konuşmacı_no)]

    def _parmak_izi(self, a, b):
        i, j = int((a - self.ses_baslangic) * ORNEK_HIZI), int((b - self.ses_baslangic) * ORNEK_HIZI)
        parca = self.ses[i:j]
        if len(parca) == 0 or float(np.sqrt(np.mean(parca ** 2))) < self.sessizlik:
            return None
        akis = self.cikarici.create_stream()
        akis.accept_waveform(ORNEK_HIZI, parca)
        akis.input_finished()
        e = np.array(self.cikarici.compute(akis))
        return e / np.linalg.norm(e)

    def _sinifla(self, e):
        benz = [float(m @ e) / np.linalg.norm(m) for m in self.merkezler]
        if benz and max(benz) >= self.esik:
            i = int(np.argmax(benz))
        elif len(self.merkezler) >= self.en_fazla:
            i = int(np.argmax(benz))
        elif self.aday is not None and float(self.aday[0] @ e) >= self.esik:
            # yeni ses iki kez duyuldu: yeni konuşmacı. İlk duyulduğu pencere de onun
            self.merkezler.append(self.aday[0] + e)
            self.etiketler.append((self.aday[1], len(self.merkezler)))
            self.aday = None
            return len(self.merkezler)
        else:
            self.aday = (e, self._pencere_ortasi)
            return None  # emin değiliz, bu pencere etiketsiz kalsın
        self.merkezler[i] = self.merkezler[i] + e
        self.aday = None
        return i + 1

    def ekle(self, parca):
        self.ses = np.concatenate([self.ses, parca])
        son = self.ses_baslangic + len(self.ses) / ORNEK_HIZI
        while self.sonraki + self.PENCERE <= son:
            self._pencere_ortasi = self.sonraki + self.PENCERE / 2
            e = self._parmak_izi(self.sonraki, self.sonraki + self.PENCERE)
            if e is not None:
                if not self.merkezler:
                    self.merkezler.append(e.copy())
                    no = 1
                else:
                    no = self._sinifla(e)
                if no:
                    self.etiketler.append((self.sonraki + self.PENCERE / 2, no))
            self.sonraki += self.ADIM
        # eski sesi ve etiketleri at
        fazla = len(self.ses) / ORNEK_HIZI - self.GECMIS
        if fazla > 0:
            kes = int(fazla * ORNEK_HIZI)
            self.ses = self.ses[kes:]
            self.ses_baslangic += kes / ORNEK_HIZI
            self.etiketler = [x for x in self.etiketler if x[0] >= self.ses_baslangic]

    @property
    def islenen(self):
        """Konuşmacısı belirlenmiş sesin sonu (sn): son işlenen pencerenin bitişi."""
        return self.sonraki - self.ADIM + self.PENCERE

    def hazir_mi(self, zaman):
        """Bu ana kadar söylenen kelimenin konuşmacısına karar verecek pencereler hazır mı?
        Kelimenin yarım pencere sonrasına kadar ses işlenmiş olmalı; yoksa konuşmacı
        değiştiği anda yalnızca eski konuşmacının pencereleri oy verir."""
        return self.islenen >= zaman + self.PENCERE / 2 + 0.25

    def kim(self, a, b):
        """[a, b] aralığında en çok kim konuşmuş? Kelimeye yakın pencerenin oyu daha ağır."""
        orta = (a + b) / 2
        yaricap = max(0.75, (b - a) / 2 + 0.25)
        oylar = {}
        for t, no in self.etiketler:
            uzaklik = abs(t - orta)
            if uzaklik <= yaricap:
                oylar[no] = oylar.get(no, 0.0) + 1.0 - uzaklik / (yaricap + 0.01)
        if oylar:
            return max(oylar, key=oylar.get)
        if self.etiketler:
            return min(self.etiketler, key=lambda x: abs(x[0] - orta))[1]
        return None


def konusmaci_gruplari(kelimeler, ayirici):
    """Her kelimenin konuşmacısını ayrı belirle, aynı kişinin art arda kelimelerini grupla.
    Dönüş: [(konuşmacı_no, [kelimeler])]. Konuşmacı cümlenin ortasında değişse bile
    değiştiği kelimeden itibaren yeni kişiye yazılır."""
    if not kelimeler:
        return []
    if ayirici is None:
        return [(None, g) for g in gruplara_bol(kelimeler)]
    kimler = [ayirici.kim(b, e) for b, e, _ in kelimeler]
    # ek ve noktalama parçası ('ı, ,) önceki kelimeyle aynı kişinindir
    for i in range(1, len(kelimeler)):
        if yapisik(kelimeler[i][2]):
            kimler[i] = kimler[i - 1]
    # tek kelimelik sapmayı düzelt: A A B A -> A A A A (pencere karışması)
    for i in range(1, len(kimler) - 1):
        if kimler[i - 1] == kimler[i + 1] != kimler[i]:
            kimler[i] = kimler[i - 1]
    sonuc, bas = [], 0
    for grup in gruplara_bol(kelimeler):
        parca, onceki_kim = [], kimler[bas]
        for j, k in enumerate(grup):
            kim = kimler[bas + j]
            if parca and kim != onceki_kim:
                sonuc.append((onceki_kim, parca))
                parca = []
            parca.append(k)
            onceki_kim = kim
        sonuc.append((onceki_kim, parca))
        bas += len(grup)
    return sonuc


def gruplara_bol(kelimeler):
    """Kesinleşen kelimeleri duraklama ve cümle sonlarından gruplara ayır;
    her grubun konuşmacısı ayrı belirlenir."""
    gruplar, mevcut = [], []
    for k in kelimeler:
        if mevcut and (k[0] - mevcut[-1][1] > 0.3 or mevcut[-1][2].endswith((".", "?", "!"))):
            gruplar.append(mevcut)
            mevcut = []
        mevcut.append(k)
    if mevcut:
        gruplar.append(mevcut)
    return gruplar


def harf(no):
    """Konuşmacı numarasını ekranda gösterilecek etikete çevir: 1 -> A, 2 -> B ...
    Operatör o harfe isim verdiyse (konusmaci_isimleri.json) isim gösterilir."""
    h = chr(ord("A") + no - 1)
    return konusmaci_adlari.get(h) or h


# Operatörün yayın sırasında harflere verdiği isimler: {"A": "CEM", "B": "AHMET"}
konusmaci_adlari = {}
_adlar_dosyasi = None
_adlar_zamani = None


def konusmaci_adlarini_yukle():
    """Dosya değiştiyse isimleri yeniden oku. Değiştiyse True döner."""
    global _adlar_zamani
    if not _adlar_dosyasi:
        return False
    try:
        zaman = _adlar_dosyasi.stat().st_mtime
    except OSError:
        zaman = None
    if zaman == _adlar_zamani:
        return False
    _adlar_zamani = zaman
    try:
        veri = json.loads(_adlar_dosyasi.read_text(encoding="utf-8")) if zaman else {}
    except (OSError, ValueError):
        return False  # dosya yazılırken okunduysa bir sonraki turda tekrar dener
    yeni = {str(k).upper(): str(v).strip() for k, v in veri.items() if str(v).strip()}
    if yeni == konusmaci_adlari:
        return False
    konusmaci_adlari.clear()
    konusmaci_adlari.update(yeni)
    return True


# ---------------------------------------------------------------- ekran

durum = {"satirlar": ["", ""], "bekleyen": "", "gecikme": 0.0, "durum": "Başlıyor..."}
durum_kilidi = threading.Lock()
satirlar = [[None, ""]]  # her satır: [konuşmacı_no, metin]
son_konusmaci = None


def satir_metni(satir):
    """Ekrana giden hali: konuşmacı biliniyorsa her satır "A: ..." diye başlar."""
    no, metin = satir
    return f"{harf(no)}: {metin}" if no is not None and metin else metin


def ekran_satirlari():
    if KONUSMACI_DUZENI:
        return [satir_metni((y["kim"], y["metin"])) for y in yuvalar]
    return ([""] * 2 + [satir_metni(s) for s in satirlar])[-2:]


# ---- "Her satır bir konuşmacı" düzeni
# satir1 ilk konuşanın, satir2 ikinci konuşanın satırıdır; iki kişi karışmaz.
# Satır dolunca o konuşmacının satırı temizlenip yeni cümleyle baştan yazılır (sayfa çevirme).
# Satırın sahibi sabittir. Üçüncü biri konuşursa en uzun süredir susan kişinin satırını
# geçici olarak ödünç alır; sahibi konuşunca satırı geri alır.
KONUSMACI_DUZENI = False
SATIR_BEKLEME = 8.0  # bir konuşmacı bu kadar sn susarsa satırı silinir (0 = silme)
yuvalar = [{"sahip": None, "kim": None, "metin": "", "son": 0.0} for _ in range(2)]


def _yuva_bul(kim):
    if kim is None:
        # kim konuştuğu belirsizse en son konuşanın satırına yaz
        return max(range(len(yuvalar)), key=lambda j: yuvalar[j]["son"])
    for anahtar in ("sahip", "kim"):  # önce kendi satırı, sonra ödünç aldığı satır
        for j, y in enumerate(yuvalar):
            if y[anahtar] == kim:
                return j
    for j, y in enumerate(yuvalar):  # sahipsiz satır: artık onun
        if y["sahip"] is None:
            y["sahip"] = kim
            return j
    return min(range(len(yuvalar)), key=lambda j: yuvalar[j]["son"])  # ödünç


_son_yazan = [None]


def yuva_kelime_ekle(k, kim, zaman):
    i = _yuva_bul(kim)
    y = yuvalar[i]
    if kim is not None and y["kim"] != kim:
        y["kim"], y["metin"] = kim, ""  # satıra başka biri geçti: baştan yaz
    elif kim is not None and _son_yazan[0] != kim:
        y["metin"] = ""  # sıra ona geçti: yeni sözü temiz satırda başlasın
    if kim is not None:
        _son_yazan[0] = kim
    # ödünç alan konuştuğu sürece, başka bir satır onu "kim" olarak tutmasın
    for d in yuvalar:
        if d is not y and d["kim"] == kim and d["sahip"] != kim:
            d["kim"], d["metin"] = d["sahip"], ""
    bitisik = yapisik(k)
    k = k.strip()
    if bitisik and y["metin"]:
        y["metin"] += k
    elif y["metin"] and len(satir_metni((y["kim"], y["metin"]))) + 1 + len(k) > SATIR_UZUNLUGU:
        y["metin"] = k
    else:
        y["metin"] = (y["metin"] + " " + k).strip()
    y["son"] = zaman


def eski_satirlari_sil(simdi):
    """Uzun süredir susan konuşmacının satırını boşalt. Değişiklik olduysa True."""
    if not KONUSMACI_DUZENI or SATIR_BEKLEME <= 0:
        return False
    degisti = False
    for y in yuvalar:
        if y["metin"] and simdi - y["son"] > SATIR_BEKLEME:
            y["metin"] = ""  # satır o kişide kalır, tekrar konuşunca aynı satıra yazar
            degisti = True
    return degisti


def kelime_ekle(k, konusmaci=None):
    """Roll-up altyazı: satır dolunca alta yeni satır açılır, üstteki kayar.
    Konuşmacı değişince her zaman yeni satıra geçilir."""
    global son_konusmaci
    bitisik = yapisik(k)
    k = k.strip()
    son = satirlar[-1]
    if konusmaci is not None and konusmaci != son_konusmaci:
        son_konusmaci = konusmaci
        if son[1]:
            satirlar.append([konusmaci, k])
        else:
            son[0], son[1] = konusmaci, k
    elif bitisik and son[1]:
        son[1] += k  # 'ı, ,5 gibi ekler satır sonunda bile kelimeden ayrılmasın
    elif son[1] and len(satir_metni(son)) + 1 + len(k) > SATIR_UZUNLUGU:
        satirlar.append([son_konusmaci, k])
    else:
        son[1] = (son[1] + " " + k).strip()
    del satirlar[:-2]


SAYFA = """<!doctype html>
<html lang="tr"><head><meta charset="utf-8"><title>Canlı Altyazı</title>
<style>
  body { margin:0; height:100vh; background:#1b1f24; font-family:Arial, sans-serif;
         display:flex; flex-direction:column; justify-content:flex-end; }
  #altyazi { margin:0 auto 6vh; background:rgba(0,0,0,.78); color:#fff; padding:.4em .8em;
             font-size:clamp(20px,3.4vw,46px); line-height:1.35; min-width:60vw; max-width:90vw; }
  #altyazi div { min-height:1.35em; white-space:pre; }
  #bilgi { position:fixed; top:12px; left:16px; right:16px; color:#8a939d; font-size:14px; }
  #bekleyen { color:#5d6670; font-style:italic; }
</style></head><body>
<div id="bilgi"><span id="durum"></span> · gecikme: <b id="gecikme">-</b> sn
  <div>bekleyen: <span id="bekleyen"></span></div></div>
<div id="altyazi"><div id="s1"></div><div id="s2"></div></div>
<script>
async function yenile() {
  try {
    const d = await (await fetch('/durum')).json();
    s1.textContent = d.satirlar[0] || ''; s2.textContent = d.satirlar[1] || '';
    bekleyen.textContent = d.bekleyen; durum.textContent = d.durum;
    gecikme.textContent = d.gecikme.toFixed(1);
  } catch (e) { durum.textContent = 'Programa ulaşılamıyor'; }
  setTimeout(yenile, 150);
}
yenile();
</script></body></html>"""


class Sunucu(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/durum":
            with durum_kilidi:
                govde = json.dumps(durum, ensure_ascii=False).encode("utf-8")
            tip = "application/json; charset=utf-8"
        else:
            govde, tip = SAYFA.encode("utf-8"), "text/html; charset=utf-8"
        self.send_response(200)
        self.send_header("Content-Type", tip)
        self.send_header("Content-Length", str(len(govde)))
        self.end_headers()
        self.wfile.write(govde)

    def log_message(self, *a):
        pass


# ---------------------------------------------------------------- Viz Engine çıkışı

class VizCikis:
    """
    Viz Engine'in komut portuna (6100) bağlanıp sahnedeki satir1/satir2 yazılarını günceller.
    Bağlantı koparsa altyazı durmaz; arka planda tekrar bağlanmaya çalışır.
    """

    def __init__(self, host, port, katman, kodlama, sahne=None):
        self.adres = (host, port)
        self.kodlama = kodlama
        self.sahne = sahne
        # Ana katmandaki sahneye RENDERER*TREE ile, ön/arka katmana RENDERER*<KATMAN>*TREE ile ulaşılır
        self.agac = "RENDERER*TREE" if katman == "MAIN_LAYER" else f"RENDERER*{katman}*TREE"
        self.katman = katman
        self.sayac = 0
        self.soket = None
        self.son_gonderilen = None
        self.son_deneme = 0.0
        self.kilit = threading.Lock()

    def _baglan(self):
        # Viz kapalıysa her kelimede bekleyip altyazıyı geciktirmeyelim: 5 sn'de bir dene
        if time.time() - self.son_deneme < 5:
            return False
        self.son_deneme = time.time()
        try:
            self.soket = socket.create_connection(self.adres, timeout=2)
            self.soket.settimeout(None)
            threading.Thread(target=self._cevaplari_oku, args=(self.soket,), daemon=True).start()
            print(f"\n[Viz] Bağlandı: {self.adres[0]}:{self.adres[1]}")
            if self.sahne:
                self._komut(f"RENDERER*{self.katman} SET_OBJECT SCENE*{self.sahne}")
            self.son_gonderilen = None
            return True
        except OSError as e:
            print(f"\n[Viz] Bağlanamadı ({e}). Altyazı tarayıcıda devam ediyor.")
            self.soket = None
            return False

    def _cevaplari_oku(self, s):
        """Viz her komuta cevap verir; okunmazsa tampon dolar. Hataları ekrana yaz."""
        try:
            while True:
                veri = s.recv(4096)
                if not veri:
                    break
                for cevap in veri.decode(self.kodlama, "replace").split("\0"):
                    if "ERROR" in cevap.upper():
                        print(f"\n[Viz] {cevap.strip()}")
        except OSError:
            pass

    def _komut(self, komut):
        self.sayac += 1
        self.soket.sendall(f"{self.sayac} {komut}".encode(self.kodlama, "replace") + b"\0")

    def yaz(self, satir1, satir2):
        with self.kilit:
            if (satir1, satir2) == self.son_gonderilen:
                return
            if self.soket is None and not self._baglan():
                return
            try:
                self._komut(f"{self.agac}*$satir1*GEOM*TEXT SET {satir1}")
                self._komut(f"{self.agac}*$satir2*GEOM*TEXT SET {satir2}")
                self.son_gonderilen = (satir1, satir2)
            except OSError:
                print("\n[Viz] Bağlantı koptu, tekrar denenecek.")
                self.soket = None


# ---------------------------------------------------------------- ses girişi

ses_kuyrugu: "queue.Queue[np.ndarray]" = queue.Queue()


def yeniden_ornekle(x, kaynak_hizi):
    if kaynak_hizi == ORNEK_HIZI:
        return x.astype(np.float32)
    n = int(round(len(x) * ORNEK_HIZI / kaynak_hizi))
    return np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.float32)


def mikrofonu_baslat(cihaz, kanal):
    import sounddevice as sd
    bilgi = sd.query_devices(cihaz, "input")
    hiz = int(bilgi["default_samplerate"])
    if kanal > bilgi["max_input_channels"]:
        sys.exit(f"Bu girişte {bilgi['max_input_channels']} kanal var, {kanal}. kanal yok.")

    def geldi(indata, frames, t, status):
        ses_kuyrugu.put(yeniden_ornekle(indata[:, kanal - 1], hiz))

    akis = sd.InputStream(device=cihaz, channels=kanal, samplerate=hiz,
                          blocksize=int(hiz * 0.1), dtype="float32", callback=geldi)
    akis.start()
    return akis, f"Dinleniyor: {bilgi['name']} (kanal {kanal})"


def dosyayi_canli_besle(yol, oynat=False):
    """Kaydı sanki stüdyodan geliyormuş gibi gerçek zamanlı hızda besle.
    oynat=True ise video da aynı anda ffplay ile oynatılır (demo için)."""
    ses = decode_audio(str(yol), sampling_rate=ORNEK_HIZI)
    parca = int(ORNEK_HIZI * 0.1)

    def besle():
        if oynat:
            subprocess.Popen(["ffplay", "-autoexit", "-loglevel", "quiet",
                              "-window_title", "Test yayini", str(yol)])
        baslangic = time.time()
        for i in range(0, len(ses), parca):
            hedef = baslangic + i / ORNEK_HIZI
            time.sleep(max(0.0, hedef - time.time()))
            ses_kuyrugu.put(ses[i:i + parca])
        ses_kuyrugu.put(None)  # bitti

    threading.Thread(target=besle, daemon=True).start()
    return f"Kayıt canlı gibi çalıyor: {Path(yol).name}"


# ---------------------------------------------------------------- ana döngü

def main():
    p = argparse.ArgumentParser(description="Canlı Türkçe altyazı")
    p.add_argument("--liste", action="store_true", help="Ses girişlerini listele ve çık")
    p.add_argument("--cihaz", type=int, help="Ses girişi numarası (--liste ile bak)")
    p.add_argument("--kanal", type=int, default=1, help="Kullanılacak kanal (varsayılan 1)")
    p.add_argument("--dosya", help="Mikrofon yerine bu kaydı gerçek zamanlı besle")
    p.add_argument("--model", default="large-v3-turbo")
    p.add_argument("--beam", type=int, default=1, help="1 hızlı, 5 daha doğru ama yavaş")
    p.add_argument("--esik", type=float, default=0.01, help="Sessizlik eşiği (RMS)")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--oynat", action="store_true", help="--dosya ile birlikte videoyu da oynat")
    p.add_argument("--viz", help="Viz Engine adresi (ör. localhost veya 10.0.0.5)")
    p.add_argument("--viz-port", type=int, default=6100)
    p.add_argument("--viz-katman", default="MAIN_LAYER",
                   choices=["MAIN_LAYER", "FRONT_LAYER", "BACK_LAYER"])
    p.add_argument("--viz-sahne", help="Açılışta yüklenecek sahne (ör. Proje/Altyazi). Verilmezse yüklü sahne kullanılır")
    p.add_argument("--viz-kodlama", default="utf-8",
                   help="Türkçe harfler Viz'de bozuk çıkarsa cp1254 deneyin")
    p.add_argument("--konusmaci", action="store_true",
                   help="Kimin konuştuğunu ayır, satırları 'A: ... / B: ...' diye göster")
    p.add_argument("--konusmaci-esik", type=float, default=0.3,
                   help="Düşürürsen daha az kişi bulur, yükseltirsen daha çok (0.2-0.4)")
    p.add_argument("--konusmaci-sayisi", type=int, default=4, help="En fazla kaç kişi")
    p.add_argument("--tarayici-acma", action="store_true", help="Altyazı sayfasını otomatik açma")
    p.add_argument("--isimler", help="Özel isim listesi (.txt). Biçim için isimler.txt'ye bak")
    p.add_argument("--konusmaci-adlari",
                   help='Harflere isim veren .json ({"A": "CEM", "B": "AHMET"}). '
                        "Yayın sırasında değiştirilebilir, program kendisi okur")
    p.add_argument("--satir-duzeni", choices=["konusmaci", "kayan"], default="konusmaci",
                   help="konusmaci: satir1 = 1. konuşmacı, satir2 = 2. konuşmacı (varsayılan, --konusmaci ile). "
                        "kayan: iki satır alttan yukarı kayar")
    p.add_argument("--satir-bekleme", type=float, default=8.0,
                   help="'konusmaci' düzeninde, bu kadar sn susan kişinin satırı silinir (0 = silme)")
    p.add_argument("--satir-uzunlugu", type=int, default=42,
                   help="Bir satıra en fazla kaç karakter sığar (Viz'de yazıyı küçültürsen artır)")
    a = p.parse_args()

    global KONUSMACI_DUZENI, SATIR_BEKLEME, SATIR_UZUNLUGU, _adlar_dosyasi
    SATIR_UZUNLUGU = max(20, a.satir_uzunlugu)
    KONUSMACI_DUZENI = a.konusmaci and a.satir_duzeni == "konusmaci"
    SATIR_BEKLEME = a.satir_bekleme

    if a.liste:
        import sounddevice as sd
        for i, d in enumerate(sd.query_devices()):
            if d["max_input_channels"] > 0:
                print(f"{i:3}  {d['name']}  ({d['max_input_channels']} kanal)")
        return

    sunucu = ThreadingHTTPServer(("0.0.0.0", a.port), Sunucu)
    threading.Thread(target=sunucu.serve_forever, daemon=True).start()
    adres = f"http://localhost:{a.port}"
    print(f"Altyazı ekranı: {adres}")
    if not a.tarayici_acma:
        webbrowser.open(adres)

    print(f"Model yükleniyor: {a.model} ...")
    model = WhisperModel(a.model, device="cuda", compute_type="float16")
    model.transcribe(np.zeros(ORNEK_HIZI, dtype=np.float32), language="tr")  # ısınma turu
    duzeltici = None
    if a.isimler:
        duzeltici = IsimDuzeltici(a.isimler)
        print(f"İsim listesi: {len(duzeltici.isimler)} isim ({Path(a.isimler).name})")
    yazici = CanliYazici(model, a.beam, a.esik, duzeltici)

    ayirici = None
    if a.konusmaci:
        model_yolu = Path(__file__).parent / "modeller" / "konusmaci_eres2net.onnx"
        if not model_yolu.exists():
            sys.exit(f"Konuşmacı modeli bulunamadı: {model_yolu}")
        ayirici = KonusmaciAyirici(model_yolu, a.konusmaci_esik, a.konusmaci_sayisi, a.esik)
        print("Konuşmacı ayırma açık (A, B, C, D)")
        if KONUSMACI_DUZENI:
            print("Satır düzeni: satir1 = 1. konuşmacı, satir2 = 2. konuşmacı")
        if a.konusmaci_adlari:
            _adlar_dosyasi = Path(a.konusmaci_adlari)
            konusmaci_adlarini_yukle()
            if konusmaci_adlari:
                print("Konuşmacı isimleri: " + ", ".join(f"{h}={ad}" for h, ad in sorted(konusmaci_adlari.items())))

    viz = None
    if a.viz:
        viz = VizCikis(a.viz, a.viz_port, a.viz_katman, a.viz_kodlama, a.viz_sahne)
        viz.yaz("", "")  # bağlan ve eski yazıyı temizle

    if a.dosya:
        mesaj = dosyayi_canli_besle(a.dosya, a.oynat)
    else:
        _akis, mesaj = mikrofonu_baslat(a.cihaz, a.kanal)
    print(mesaj)
    with durum_kilidi:
        durum["durum"] = mesaj

    kayit_yolu = Path(f"canli_kayit_{datetime.now():%Y%m%d_%H%M%S}.txt")
    kayit = kayit_yolu.open("w", encoding="utf-8")
    son_islenen = 0.0
    bitti = False
    konusma_kuyrugu = []  # kesinleşti, konuşmacısının belli olmasını bekliyor
    print("Durdurmak için Ctrl+C\n")

    try:
        while not bitti:
            parca = ses_kuyrugu.get()
            parcalar = [parca]
            while not ses_kuyrugu.empty():
                parcalar.append(ses_kuyrugu.get_nowait())
            if any(x is None for x in parcalar):
                bitti = True
            parcalar = [x for x in parcalar if x is not None]
            if parcalar:
                yeni_ses = np.concatenate(parcalar)
                yazici.ekle(yeni_ses)
                if ayirici:
                    ayirici.ekle(yeni_ses)
            if konusmaci_adlarini_yukle():
                # Operatör isim verdi/değiştirdi: ekrandaki satırları hemen yeni isimle bas
                print("\n[İsimler] " + (", ".join(f"{h}={ad}" for h, ad in sorted(konusmaci_adlari.items())) or "harflere dönüldü"))
                if viz:
                    viz.yaz(*ekran_satirlari())
                with durum_kilidi:
                    durum["satirlar"] = ekran_satirlari()
            if not bitti and yazici.alinan - son_islenen < EN_KISA_PARCA:
                continue

            t0 = time.time()
            alinan = son_islenen = yazici.alinan
            kesinlesen, bekleyen = yazici.isle()
            if bitti and bekleyen:  # kayıt bitti, kalanları da yaz
                kesinlesen += bekleyen
                bekleyen = []

            # Konuşmacı ayırma açıksa kelimeyi, kimin söylediği belli olana kadar (sesin
            # yarım pencere sonrası da işlenene kadar, ~1 sn) beklet. Yoksa yeni konuşmacının
            # ilk kelimeleri eski konuşmacıya yazılıyordu.
            konusma_kuyrugu.extend(kesinlesen)
            n = len(konusma_kuyrugu)
            if ayirici and not bitti:
                n = 0
                while n < len(konusma_kuyrugu) and ayirici.hazir_mi(konusma_kuyrugu[n][1]):
                    n += 1
            kesinlesen, konusma_kuyrugu[:] = konusma_kuyrugu[:n], konusma_kuyrugu[n:]

            if kesinlesen:
                for kim, grup in konusmaci_gruplari(kesinlesen, ayirici):
                    if kim is not None and kim != son_konusmaci:
                        print(f"\n{harf(kim)}:", end=" ")
                    for _, son, k in grup:
                        kelime_ekle(k, kim)  # kayan düzen + terminal/döküm için konuşmacı takibi
                        if KONUSMACI_DUZENI:
                            yuva_kelime_ekle(k, kim, son)
                    metin = birlestir(k for _, _, k in grup)
                    print(metin, end=" ", flush=True)
                    etiket = f" {harf(kim)}:" if kim is not None else ""
                    kayit.write(f"[{grup[0][0]:8.1f}]{etiket} {metin}\n")
                kayit.flush()
            silinen = eski_satirlari_sil(alinan)
            if viz and (kesinlesen or silinen):
                viz.yaz(*ekran_satirlari())
            with durum_kilidi:
                durum["satirlar"] = ekran_satirlari()
                durum["bekleyen"] = birlestir(k for _, _, k in konusma_kuyrugu + bekleyen)
                if kesinlesen:
                    durum["gecikme"] = alinan - kesinlesen[-1][1] + (time.time() - t0)
    except KeyboardInterrupt:
        pass
    finally:
        kayit.close()
        print(f"\n\nDöküm kaydedildi: {kayit_yolu}")

    if bitti:
        print("Kayıt bitti. Ekran açık kalıyor, kapatmak için Ctrl+C.")
        try:
            while True:
                time.sleep(1)
                if konusmaci_adlarini_yukle():
                    if viz:
                        viz.yaz(*ekran_satirlari())
                    with durum_kilidi:
                        durum["satirlar"] = ekran_satirlari()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
