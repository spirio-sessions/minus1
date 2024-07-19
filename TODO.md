# TODOs

### Data
- Multiply data by increasing its size by 12x via transcribing it half a tone
  - Yup!

### Realtime
- pitch extractor can only gather one note at a time?
  - egal, 
- test with immediate feedback via midi
- use confidence score? (already working with tolerance?)

### LSTM-Model
- Loss function via reinforcement learning or simple formular, hidden layer as performance increase
  - (cross entrophy)
### Auto-regressiv
- Which part is getting dumped or saved? 
  - (both are predicted in training, in inference melody is dumped)


### Dataset Problem
- Has problems recognising the harmony when only one tone is played in the melody (EGAL)

Daten mit 12 Tasten + Oktave

Buffer größer machen, damit er mehr Zeit hat.
Loss function anpassen

Timewindow (png in privaten Repo als Timewindow.png)
-> Im LSTM schummeln und window einbauen für Vergangenheit
Überlgen ob Convolutional layer, Convolution 1D, dann haben wir ein zeitliches Fenster.
1. Windown
2. cnn vllt

Resultat:
1. Demo aus Testdaten 
2. Eigengespieltes,
3. Echtzeit

Im Training sagen wir linke und rechte Hand voraus. So lernt das Model (nicht nur harmony)
In der Testinferenz sagen wir auch beides voraus, werfen die Melody weg und ersetzten sie durch die Ground Truth.
Im Training autoregressiv arbeiten, aber in den tests dann die Melody wegschmeißen/ersetzen.
Wenn er beide predicted gibts viele Fehler und es zerfällt, Stützte durch Interpolation mit Ground Truth und dann in der Inferenz mit der gespielten Melody ersetzen
Kreuzentrophy und dann über Differenzmaß oder Abstand eine Kostenfunktion selbst bauen.
Mal 12 alles nehmen, damit wir für alle Tonarten trainiert haben.

Alte Sachen nicht wegschmeißen.
1. 88 Keys
2. 12 + oct
3. 12 + oct wegschmeißen
4. Loss function hier und da
5. Andere Loss function
6. Daten Multiplizieren *12

Immer: Stats an denen man es gut ausrechnen kann + Hördemo
I-Tüpfelchen dann die Live Demo


LSTM History states erstmal anfahren lassen. Probieren, dass der Klavierspieler erstmal die linke Hand mitspielt, und dann aufhört.

-------



- [ ] Im Training  ganz normale binary cross entropy für beide Teile.

- [ ] In der Inferenz dann der SoftMax auf der Hand, die ich haben will. Oder per Threshholding.

- [ ] die andere Hand wird wieder reinkopiert, die braucht keinen Softmax. Pro Dimension eine Sigmoid anstatt Softmax

- [ ] Los function. Wie ist die Ableitung von where class. Müssen eine Ableitungsfunktion schaffen. .A Aber eigentlich sehr schnell. Nochmal in die Dokumentation

- [ ] Wenn die loss function harmonies und melodies nimmt, kann es nicht mehr autoregressiv sein.

- [ ] input_size und output muss schon 24 sein. Er soll beide voraussagen
- [ ] Inverenz muss dann die seq_lenght manuell sein (sollte im Code passen)

- [ ] Normal      T, T+1, T+2, T+3, T+4
- [ ] Dataloader  T+1, T+2, T+3, T+4 (oder auch mehr als 1 Frame)
- [ ] auto torch dataloader,
- [ ] Beide hände, autoregressive, einmal beide hände rein, andere raus, einmal eine hand rein und beide Hände raus.
Wenn der Dataloader einmal stimmt.

- [ ] Nur aus rechter Hand die linke generieren ist ein gutes Experiement, wird aber nicht gut funktionieren.

- [ ] Tensorboard Torch Loss schauen, ob es gut trainiert. Erste Diagnostic.

https://pytorch.org/tutorials/recipes/recipes/tensorboard_with_pytorch.html

- [ ] Sequence Length 512 snapshots anstatt alles.
- [ ] Aber noch weniger ist besser 32, 64 und mit overlapping
- [ ] Im Tensor Board dann nachchecken.
