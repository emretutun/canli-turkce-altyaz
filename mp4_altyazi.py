"""
MP4 (veya herhangi bir ses/video dosyası) -> Türkçe SRT altyazı.

Kullanım:
    python mp4_altyazi.py kayit.mp4
    python mp4_altyazi.py kayit.mp4 --isimler isimler.txt
    python mp4_altyazi.py kayit.mp4 --model medium

Çıktı: kayit.srt (videonun yanına). VLC ile videoyu açınca altyazı otomatik gelir.
"""
import argparse
import os
import sys
import textwrap
import time
from pathlib import Path


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
from faster_whisper import WhisperModel  # noqa: E402

SATIR_UZUNLUGU = 42     # bir altyazı satırındaki en fazla karakter
SATIR_SAYISI = 2        # ekranda aynı anda en fazla satır
EN_UZUN_SURE = 6.0      # bir altyazının ekranda kalacağı en uzun süre (sn)
SESSIZLIK_BOSLUGU = 1.0 # bu kadar sessizlik olursa yeni altyazıya geç (sn)


def zaman(sn: float) -> str:
    ms = int(round(sn * 1000))
    s, ms = divmod(ms, 1000)
    d, s = divmod(s, 60)
    sa, d = divmod(d, 60)
    return f"{sa:02}:{d:02}:{s:02},{ms:03}"


def kelimeleri_altyaziya_bol(kelimeler):
    """Kelime zamanlarını kullanarak okunabilir altyazı blokları oluştur."""
    max_karakter = SATIR_UZUNLUGU * SATIR_SAYISI
    bloklar, mevcut = [], []

    def kapat():
        if mevcut:
            metin = "".join(k.word for k in mevcut).strip()
            bloklar.append((mevcut[0].start, mevcut[-1].end, metin))
            mevcut.clear()

    for k in kelimeler:
        if mevcut:
            aday = "".join(x.word for x in mevcut) + k.word
            if (
                len(aday.strip()) > max_karakter
                or k.end - mevcut[0].start > EN_UZUN_SURE
                or k.start - mevcut[-1].end > SESSIZLIK_BOSLUGU
            ):
                kapat()
        mevcut.append(k)
        # Cümle bittiyse ve blok yeterince doluysa orada kes
        if k.word.strip().endswith((".", "?", "!")) and len("".join(x.word for x in mevcut)) > SATIR_UZUNLUGU:
            kapat()
    kapat()
    return bloklar


def main():
    p = argparse.ArgumentParser(description="Dosyadan Türkçe SRT altyazı üretir.")
    p.add_argument("dosya", help="MP4 / MXF / WAV vb.")
    p.add_argument("--model", default="large-v3-turbo",
                   help="large-v3-turbo (varsayılan), large-v3, medium, small")
    p.add_argument("--isimler", help="Her satırda bir özel isim olan .txt (ör. bakan, şehir adları)")
    p.add_argument("--cpu", action="store_true", help="Ekran kartı yerine işlemciyle çalış")
    a = p.parse_args()

    girdi = Path(a.dosya)
    if not girdi.exists():
        sys.exit(f"Dosya bulunamadı: {girdi}")
    cikti = girdi.with_suffix(".srt")

    ipucu = None
    if a.isimler:
        isimler = [s.strip() for s in Path(a.isimler).read_text(encoding="utf-8").splitlines() if s.strip()]
        ipucu = "Haber bülteni. Geçen isimler: " + ", ".join(isimler) + "."

    cihaz, tip = ("cpu", "int8") if a.cpu else ("cuda", "float16")
    print(f"Model yükleniyor: {a.model} ({cihaz}) - ilk seferde indirilir, biraz sürebilir...")
    model = WhisperModel(a.model, device=cihaz, compute_type=tip)

    print(f"Yazıya dökülüyor: {girdi.name}")
    t0 = time.time()
    segmentler, bilgi = model.transcribe(
        str(girdi),
        language="tr",
        vad_filter=True,                       # sessizlik/müzikte metin uydurmasın
        vad_parameters={"min_silence_duration_ms": 500},
        word_timestamps=True,
        initial_prompt=ipucu,
        condition_on_previous_text=False,      # bir hata sonraki cümlelere yayılmasın
    )

    kelimeler = []
    for seg in segmentler:
        kelimeler.extend(seg.words or [])
        print(f"  [{zaman(seg.start)[:8]}] {seg.text.strip()}")
    gecen = time.time() - t0

    bloklar = kelimeleri_altyaziya_bol(kelimeler)
    with cikti.open("w", encoding="utf-8") as f:
        for i, (bas, son, metin) in enumerate(bloklar, 1):
            satirlar = textwrap.wrap(metin, SATIR_UZUNLUGU)
            f.write(f"{i}\n{zaman(bas)} --> {zaman(son)}\n" + "\n".join(satirlar) + "\n\n")

    sure = bilgi.duration
    print()
    print(f"Bitti: {cikti}")
    print(f"Kayıt süresi : {sure/60:.1f} dk")
    print(f"İşlem süresi : {gecen/60:.1f} dk  (gerçek zamanın {sure/gecen:.1f} katı hızlı)")
    if sure / gecen < 1.5:
        print("Uyarı: canlı yayın için bu hız sınırda. Daha küçük bir model denenebilir (--model medium).")


if __name__ == "__main__":
    main()
