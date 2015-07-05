In the context of a machine supported orchestra/band, when writing song texts, I figured they are all are quiet alike. So I was wondering if I could make a machine that could do my job or even better: Creating new text from looking at artists that I like or whom's combination I imagine to be fun.

So here is the approach: We take an LSTM recurrent neural network (non-linear directed cycles with internal memory) and train (=adjust internal weights) it syllable-wise on Wikipedia [1]. The model will compute the probability of the next word given the previous words. The weights that we trained on Wikipedia will serve as a prior for our actual model i.e. we continue training with our own text e.g. Goethe or KIZ. 

When training is done we can generate an entire song text  by giving the model a random or chosen word/phrase. We can also interrupt at any point, change phrases and the model will react on that. Its interesting to see if it can learn rhythms and rhymes from syllables.

In a larger context, the resulting texts are supposed to be performed by humans in contrast to the by now overused approach to let a computer read human written text.
