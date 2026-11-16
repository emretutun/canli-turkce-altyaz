"""
Canlı Altyazı başlatma ekranı.

Komut yazmadan: videoyu seç, Başlat'a bas. Arka planda canli_altyazi.py'yi
seçilen ayarlarla çalıştırır ve çıktısını pencerede gösterir.
"""

import codecs
import json
import os
import subprocess
import sys
import threading
import tkinter as tk
import webbrowser
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

KLASOR = Path(__file__).resolve().parent
PROGRAM = KLASOR / "canli_altyazi.py"
AYAR_DOSYASI = KLASOR / "baslat_ayar.json"
ADLAR_DOSYASI = KLASOR / "konusmaci_isimleri.json"  # program yayın sırasında okur
HARFLER = "ABCD"
VARSAYILAN = {
    "video": "",
    "viz": "localhost",
    "sahne": "/000_WORK/canli_altyazi",
    "konusmaci": True,
    "dogru": False,
    "oynat": True,
    "satir_konusmaci": True,
    "uzunluk": 42,
    "isimler": str(KLASOR / "isimler.txt"),
}


def python_exe():
    # .pyw, pythonw.exe ile açılır; alt süreç için aynı kurulumdaki python.exe'yi kullan
    exe = Path(sys.executable)
    aday = exe.with_name("python.exe")
    return str(aday if aday.exists() else exe)


def ayar_oku():
    try:
        return {**VARSAYILAN, **json.loads(AYAR_DOSYASI.read_text(encoding="utf-8"))}
    except (OSError, ValueError):
        return dict(VARSAYILAN)


class Pencere:
    def __init__(self, kok):
        self.kok = kok
        self.surec = None
        ayar = ayar_oku()

        kok.title("Canlı Altyazı")
        kok.geometry("800x680")
        kok.minsize(600, 420)

        cerceve = ttk.Frame(kok, padding=12)
        cerceve.pack(fill="both", expand=True)
        cerceve.columnconfigure(1, weight=1)

        # Video
        ttk.Label(cerceve, text="Video:").grid(row=0, column=0, sticky="w", pady=4)
        self.video = tk.StringVar(value=ayar["video"])
        ttk.Entry(cerceve, textvariable=self.video).grid(row=0, column=1, sticky="ew", padx=6)
        ttk.Button(cerceve, text="Video Seç...", command=self.video_sec).grid(row=0, column=2)

        # Viz
        ttk.Label(cerceve, text="Viz adresi:").grid(row=1, column=0, sticky="w", pady=4)
        self.viz = tk.StringVar(value=ayar["viz"])
        ttk.Entry(cerceve, textvariable=self.viz, width=20).grid(row=1, column=1, sticky="w", padx=6)

        ttk.Label(cerceve, text="Viz sahnesi:").grid(row=2, column=0, sticky="w", pady=4)
        self.sahne = tk.StringVar(value=ayar["sahne"])
        ttk.Entry(cerceve, textvariable=self.sahne).grid(row=2, column=1, sticky="ew", padx=6)
        ttk.Label(cerceve, text="(boş = Viz'de açık sahne)", foreground="gray").grid(row=2, column=2, sticky="w")

        # İsim listesi
        ttk.Label(cerceve, text="İsim listesi:").grid(row=3, column=0, sticky="w", pady=4)
        self.isimler = tk.StringVar(value=ayar["isimler"])
        ttk.Entry(cerceve, textvariable=self.isimler).grid(row=3, column=1, sticky="ew", padx=6)
        isim_dugme = ttk.Frame(cerceve)
        isim_dugme.grid(row=3, column=2, sticky="w")
        ttk.Button(isim_dugme, text="Seç...", command=self.isim_sec).pack(side="left")
        ttk.Button(isim_dugme, text="Düzenle", command=self.isim_duzenle).pack(side="left", padx=(4, 0))

        # Seçenekler
        secenek = ttk.Frame(cerceve)
        secenek.grid(row=4, column=0, columnspan=3, sticky="w", pady=(8, 4))
        self.konusmaci = tk.BooleanVar(value=ayar["konusmaci"])
        self.dogru = tk.BooleanVar(value=ayar["dogru"])
        self.oynat = tk.BooleanVar(value=ayar["oynat"])
        self.satir_konusmaci = tk.BooleanVar(value=ayar["satir_konusmaci"])
        ttk.Checkbutton(secenek, text="Konuşmacıları ayır (A: / B:)", variable=self.konusmaci).pack(side="left", padx=(0, 16))
        ttk.Checkbutton(secenek, text="Her satır bir konuşmacı", variable=self.satir_konusmaci).pack(side="left", padx=(0, 16))
        ttk.Checkbutton(secenek, text="Videoyu da oynat", variable=self.oynat).pack(side="left", padx=(0, 16))
        ttk.Checkbutton(secenek, text="Daha doğru (yavaş)", variable=self.dogru).pack(side="left", padx=(0, 16))
        ttk.Label(secenek, text="Satır uzunluğu:").pack(side="left")
        self.uzunluk = tk.IntVar(value=ayar["uzunluk"])
        ttk.Spinbox(secenek, from_=20, to=90, increment=2, width=4, textvariable=self.uzunluk).pack(side="left", padx=(4, 0))

        # Konuşmacı isimleri: yayın sırasında da yazılabilir, ekrana anında yansır
        adlar = ttk.LabelFrame(cerceve, text="Konuşmacı isimleri (yayın sırasında da değiştirebilirsin)", padding=8)
        adlar.grid(row=5, column=0, columnspan=3, sticky="ew", pady=(4, 0))
        try:
            kayitli = json.loads(ADLAR_DOSYASI.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            kayitli = {}
        self.adlar = {}
        self._adlar_zamanlayici = None
        for i, h in enumerate(HARFLER):
            ttk.Label(adlar, text=f"{h} =").grid(row=0, column=i * 2, sticky="e", padx=(0 if i == 0 else 12, 4))
            d = tk.StringVar(value=kayitli.get(h, ""))
            d.trace_add("write", lambda *_: self.adlari_sonra_yaz())
            ttk.Entry(adlar, textvariable=d, width=14).grid(row=0, column=i * 2 + 1)
            self.adlar[h] = d
        ttk.Button(adlar, text="Temizle", command=self.adlari_temizle).grid(row=0, column=len(HARFLER) * 2, padx=(12, 0))
        ttk.Label(adlar, foreground="gray",
                  text="Harfler konuşma sırasına göre verilir: ilk konuşan A, ikinci B... "
                       "Kimin hangi harf olduğunu görünce adını yaz.").grid(
            row=1, column=0, columnspan=len(HARFLER) * 2 + 1, sticky="w", pady=(6, 0))

        # Düğmeler
        dugmeler = ttk.Frame(cerceve)
        dugmeler.grid(row=6, column=0, columnspan=3, sticky="w", pady=8)
        self.baslat_dugme = ttk.Button(dugmeler, text="▶  Başlat", command=self.baslat)
        self.baslat_dugme.pack(side="left")
        self.durdur_dugme = ttk.Button(dugmeler, text="■  Durdur", command=self.durdur, state="disabled")
        self.durdur_dugme.pack(side="left", padx=8)
        ttk.Button(dugmeler, text="Önizlemeyi aç", command=lambda: webbrowser.open("http://localhost:8765")).pack(side="left")
        ttk.Button(dugmeler, text="Klasörü aç", command=lambda: os.startfile(KLASOR)).pack(side="left", padx=8)

        self.durum = tk.StringVar(value="Hazır. Video seçip Başlat'a bas.")
        ttk.Label(cerceve, textvariable=self.durum, foreground="#1a5fb4").grid(row=7, column=0, columnspan=3, sticky="w")

        # Çıktı
        self.cikti = tk.Text(cerceve, wrap="word", font=("Consolas", 11), height=12)
        self.cikti.grid(row=8, column=0, columnspan=3, sticky="nsew", pady=(6, 0))
        kaydirma = ttk.Scrollbar(cerceve, command=self.cikti.yview)
        kaydirma.grid(row=8, column=3, sticky="ns", pady=(6, 0))
        self.cikti["yscrollcommand"] = kaydirma.set
        cerceve.rowconfigure(8, weight=1)

        kok.protocol("WM_DELETE_WINDOW", self.kapat)

    # ------------------------------------------------------------ eylemler

    def video_sec(self):
        mevcut = Path(self.video.get())
        baslangic = mevcut.parent if mevcut.parent.exists() and str(mevcut) else Path.home() / "Desktop"
        yol = filedialog.askopenfilename(
            title="Test videosunu seç",
            initialdir=str(baslangic),
            filetypes=[("Video / ses", "*.mp4 *.mxf *.mov *.mkv *.avi *.wav *.mp3"), ("Tüm dosyalar", "*.*")],
        )
        if yol:
            self.video.set(os.path.normpath(yol))

    def _uzunluk(self):
        try:
            return min(90, max(20, int(self.uzunluk.get())))
        except (tk.TclError, ValueError):
            return 42

    def adlari_sonra_yaz(self):
        # Her tuşta değil, yazmayı bırakınca (0,5 sn) kaydet; yarım isim ekrana gitmesin
        if self._adlar_zamanlayici:
            self.kok.after_cancel(self._adlar_zamanlayici)
        self._adlar_zamanlayici = self.kok.after(500, self.adlari_yaz)

    def adlari_yaz(self):
        self._adlar_zamanlayici = None
        veri = {h: d.get().strip() for h, d in self.adlar.items() if d.get().strip()}
        gecici = ADLAR_DOSYASI.with_suffix(".tmp")
        try:
            gecici.write_text(json.dumps(veri, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(gecici, ADLAR_DOSYASI)  # program yarım dosya okumasın
        except OSError:
            return
        if self.surec:
            self.durum.set("İsimler güncellendi: " + (", ".join(f"{h}={a}" for h, a in veri.items()) or "harfler"))

    def adlari_temizle(self):
        for d in self.adlar.values():
            d.set("")

    def isim_sec(self):
        yol = filedialog.askopenfilename(
            title="İsim listesini seç", initialdir=str(KLASOR),
            filetypes=[("Metin", "*.txt"), ("Tüm dosyalar", "*.*")],
        )
        if yol:
            self.isimler.set(os.path.normpath(yol))

    def isim_duzenle(self):
        yol = Path(self.isimler.get().strip() or KLASOR / "isimler.txt")
        if not yol.exists():
            yol.write_text("# Her satıra bir isim yaz\n", encoding="utf-8")
            self.isimler.set(str(yol))
        subprocess.Popen(["notepad.exe", str(yol)])

    def ayar_kaydet(self):
        ayar = {
            "video": self.video.get(), "viz": self.viz.get(), "sahne": self.sahne.get(),
            "konusmaci": self.konusmaci.get(), "dogru": self.dogru.get(), "oynat": self.oynat.get(),
            "satir_konusmaci": self.satir_konusmaci.get(),
            "uzunluk": self._uzunluk(),
            "isimler": self.isimler.get(),
        }
        try:
            AYAR_DOSYASI.write_text(json.dumps(ayar, ensure_ascii=False, indent=2), encoding="utf-8")
        except OSError:
            pass

    def baslat(self):
        video = self.video.get().strip()
        if not video or not Path(video).exists():
            messagebox.showwarning("Canlı Altyazı", "Önce geçerli bir video seç.")
            return
        komut = [python_exe(), "-u", str(PROGRAM), "--dosya", video]
        if self.oynat.get():
            komut.append("--oynat")
        if self.viz.get().strip():
            komut += ["--viz", self.viz.get().strip()]
            if self.sahne.get().strip():
                komut += ["--viz-sahne", self.sahne.get().strip()]
        if self.konusmaci.get():
            self.adlari_yaz()
            komut += ["--konusmaci", "--konusmaci-adlari", str(ADLAR_DOSYASI),
                      "--satir-duzeni", "konusmaci" if self.satir_konusmaci.get() else "kayan"]
        if self.dogru.get():
            komut += ["--beam", "5"]
        komut += ["--satir-uzunlugu", str(self._uzunluk())]
        isimler = self.isimler.get().strip()
        if isimler:
            if not Path(isimler).exists():
                messagebox.showwarning("Canlı Altyazı", f"İsim listesi bulunamadı:\n{isimler}")
                return
            komut += ["--isimler", isimler]
        self.ayar_kaydet()

        ortam = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
        self.cikti.delete("1.0", "end")
        self.yaz(f"Başlatılıyor: {Path(video).name}\n\n")
        try:
            self.surec = subprocess.Popen(
                komut, cwd=KLASOR, env=ortam,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
        except OSError as e:
            messagebox.showerror("Canlı Altyazı", f"Program başlatılamadı:\n{e}")
            return
        self.baslat_dugme["state"] = "disabled"
        self.durdur_dugme["state"] = "normal"
        self.durum.set("Çalışıyor... (model ilk açılışta 20-30 sn yüklenir)")
        threading.Thread(target=self.cikti_oku, args=(self.surec,), daemon=True).start()

    def cikti_oku(self, surec):
        # Program kelimeleri satır sonu beklemeden basıyor; parça parça oku
        cozucu = codecs.getincrementaldecoder("utf-8")("replace")
        while True:
            veri = surec.stdout.read1(4096)
            if not veri:
                break
            metin = cozucu.decode(veri)
            if metin:
                self.kok.after(0, self.yaz, metin)
        surec.wait()
        self.kok.after(0, self.bitti, surec)

    def yaz(self, metin):
        self.cikti.insert("end", metin)
        self.cikti.see("end")
        if "not allowed in this mode" in metin:
            self.durum.set("UYARI: Viz On Air modunda değil! Viz'de sağ üstten On Air'e geç.")
        elif "Bağlandı" in metin:
            self.durum.set("Viz'e bağlandı. Altyazı gidiyor.")
        elif "Bağlanamadı" in metin:
            self.durum.set("Viz'e bağlanamadı! Viz açık ve On Air mı? Altyazı tarayıcıda devam ediyor.")
        elif "Kayıt bitti" in metin:
            self.durum.set("Video bitti. Yeniden başlatmak için önce Durdur'a bas.")

    def durdur(self):
        if self.surec and self.surec.poll() is None:
            # ffplay penceresiyle birlikte kapat
            subprocess.run(["taskkill", "/PID", str(self.surec.pid), "/T", "/F"],
                           capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW)

    def bitti(self, surec):
        if surec is not self.surec:
            return
        self.surec = None
        self.baslat_dugme["state"] = "normal"
        self.durdur_dugme["state"] = "disabled"
        self.durum.set("Durdu. Yazılanlar klasördeki canli_kayit_*.txt dosyasında.")
        self.yaz("\n\n--- Durdu ---\n")

    def kapat(self):
        self.durdur()
        self.kok.destroy()


if __name__ == "__main__":
    kok = tk.Tk()
    Pencere(kok)
    kok.mainloop()
