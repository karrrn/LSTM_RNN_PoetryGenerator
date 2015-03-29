When writing song text I figured they do all are kind of alike. So I was wondering if I could make a machine that could do my job or even better: Creating new text from looking at artists that I like or whom's combination I imagine to be fun.

So here is the approach: We take a LSTM recurrent neural network and train it character-wise on wikipedia [1]. The weights that we trained on Wikipedia will serve as a prior for our actual model i.e. we continue training with our own text e.g. Göthe or Bushido.

When training is done we can generate an entire song text by giving the model a random or chosen word/phrase. I wonder whether it can learn rhythmics when I provide mostly lyrical text. Will it figure the number syllables to be important?

What to do with the Text we got out? Obvious! In contrast to the by now overused approach to let a computer read human written text, humans will perform computer written text. Opposing to the notion that machines are enemies, or slaves. They are our friend, part of our lives they are us. Or like Mufasa used to say we are all one. 