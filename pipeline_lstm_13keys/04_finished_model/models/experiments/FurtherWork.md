Create a heatmap for each model to a good song.
Show, that they tend to play better stuff when the loss of lower, or, that loss doesnt manipulate the whole output,
like when its shit, but the loss is good...

More stats?
how many notes? how long do they hold them?

Prof: Show, what data the model had the most/least trouble with.
Ask Felix on how to do that properly


num_epochs:
5 shows a lot
10-25 optimal

hidden_size:
kleiner < 256=512

num_layers:
2 = 3 = 4 > 10

learning_rate:
0,001 <= 0,0001

batch_size:
64=128=256, dafür aber schneller

seq_length:
kleiner = besserer loss
kleiner != bessere Musik?????
seq256 = 24-25%
seq32 = 21-22%
seq8 = 16-17%
seq4 = 13%
seq2 = 8-9%
8 opt, 4?, 2?, 1?

stride:
Keinen Unterschied...

databank:
nicht direkt besser, eher schlechter.



Pipeline:
seq16 probieren mit Notenqualität





INFERNECE:

Artificially delete all notes that are only 1-3 snapshots long?