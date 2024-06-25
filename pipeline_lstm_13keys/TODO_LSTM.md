# TODO:

## Right now!
- MIDI restart without plugin in and out
- Best num_layers and hidden layers
- Puffer-function to delay and predict future for auto-regressiv
  - >Timewindow (png in privaten Repo als Timewindow.png)
    -> Im LSTM schummeln und window einbauen für Vergangenheit
    Überlegen ob Convolutional layer, Convolution 1D, dann haben wir ein zeitliches Fenster.
    >1. Windown
    >2. cnn vllt
- Test with transpose and without
- training to full gpu
- loss function music-theoreticly okay? Different kinds of loss functions
- give him more than 12 keys to play in

## Later!
- velocity of notes same to notes played live
- create nice readme
- more statistics for evaluation in inference.
- train on life left hand and then take over?
- as someone smart said:
  - >Im Training sagen wir linke und rechte Hand voraus. So lernt das Model (nicht nur harmony)
    In der Testinferenz sagen wir auch beides voraus, werfen die Melody weg und ersetzten sie durch die Ground Truth.
    Im Training autoregressiv arbeiten, aber in den tests dann die Melody wegschmeißen/ersetzen.
    Wenn er beide predicted gibts viele Fehler und es zerfällt, Stützte durch Interpolation mit Ground Truth und dann in der Inferenz mit der gespielten Melody ersetzen
    Kreuzentrophy und dann über Differenzmaß oder Abstand eine Kostenfunktion selbst bauen.
    Mal 12 alles nehmen, damit wir für alle Tonarten trainiert haben.

## Finish!
- Multiple tests with:
  - demo
  - self-played
  - real-time
- swap hands to calculate the other side
- preparation for live demo