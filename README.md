# Deepfake-classifier

## 2 Modelul 2 – CNN + FFT (Rețea Convoluțională cu Fourier Features)

### 2.1 Context & idee
Această metodă combină spațiul RGB cu informații din spațiul frecvențial (transformata Fourier). Ideea este ca imaginile generate de modele deepfake conțin artefacte subtile în frecvențele înalte sau medii, invizibile pentru ochiul uman, dar detectabile statistic.

### 2.2 Preprocesare
- Imaginile sunt redimensionate la 100×100 pixeli.
- Pentru fiecare canal (R, G, B), se calculează FFT 2D.
- Se aplică un band-pass filter (reține frecvențele între 5 și 30 pixeli în jurul centrului).
- Se extrage magnitudinea logaritmică, apoi se normalizează între [0, 1].
- Cele 3 canale FFT sunt concatenate cu canalele RGB → tensor de dimensiune (6, 100, 100).

### 2.3 Model CNN

#### Arhitectura rețelei CNN + FFT
Rețeaua convoluțională utilizată este construită de la zero, fără a folosi modele pre-antrenate, și este antrenată pe un input extins de dimensiune (6, 100, 100) — format din 3 canale RGB + 3 canale FFT (magnitudine spectru filtrat band-pass).

1. **Blocuri convoluționale:**
   - Arhitectura conține 4 blocuri identice ca structură, fiecare având:
     - Două straturi convoluționale: `Conv2d(in_ch, out_ch, kernel=3, padding=1)`
     - Normalizare în lot: `BatchNorm2d`
     - Funcție de activare: `ReLU(inplace=True)`
     - `MaxPooling2d(2)` — reduce dimensiunile spațiale la jumătate
     - Dropout cu probabilitate progresivă (`p = 0.2 → 0.5`)
   - Dimensiunile spațiale ale tensorului se reduc astfel:
     `(6, 100, 100) → (32, 50, 50) → (64, 25, 25) → (128, 12, 12) → (256, 6, 6)`

2. **Straturi dense:**
   - Tabloul convoluțional final este aplatizat: `Flatten()`
   - Se aplică: `Linear(256 × 6 × 6, 512) → ReLU → Dropout(0.5) → Linear(512, 5)`
   - Ultimul strat produce un vector de dimensiune 5 — corespunzător celor 5 clase de generare.

3. **Funcția de pierdere:**
   - S-a utilizat `CrossEntropyLoss` ponderată în funcție de distribuția etichetelor din setul de antrenare.
   - Ponderile sunt calculate cu `sklearn.utils.class_weight.compute_class_weight`.

4. **Optimizator și scheduler:**
   - Optimizatorul folosit este `Adam` cu learning rate `lr = 0.0005`.
   - S-a aplicat un scheduler de tip `CosineAnnealingLR` cu `Tmax = 100`, `ηmin = 1e−5` pentru ajustarea progresivă a ratei de învățare pe durata antrenării.
   - Se antrenează până la 100 epoci, păstrând cel mai bun model pe validare.
