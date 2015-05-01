#--------------------------------------
# Libraries
#--------------------------------------

import theano, theano.tensor as T
import theano_lstm
from theano_lstm import Embedding, LSTM, RNN, StackedCells, Layer, create_optimization_updates, masked_loss

import numpy as np
import random
import re
from time import time
import cPickle as pickle

from load_wiki import Wiki_articles, Vocab
floatX = theano.config.floatX


def pad_into_matrix(rows, padding = 0):
    if len(rows) == 0:
        return np.array([0, 0], dtype=np.int32)
    lengths = [i for i in map(len, rows)]
    width = max(lengths)
    #width = 2500
    height = len(rows)
    mat = np.empty([height, width], dtype=rows[0].dtype)
    mat.fill(padding)
    for i, row in enumerate(rows):
        mat[i, 0:len(row)] = row
    return mat, list(lengths)
    #return width

#--------------------------------------
# Model
#--------------------------------------
def softmax(x):
    """
    Wrapper for softmax, helps with
    pickling, and removing one extra
    dimension that Theano adds during
    its exponential normalization.
    """
    return T.nnet.softmax(x.T)

def has_hidden(layer):
    """
    Whether a layer has a trainable
    initial hidden state.
    """
    return hasattr(layer, 'initial_hidden_state')

def matrixify(vector, n):
    return T.repeat(T.shape_padleft(vector), n, axis=0)

def initial_state(layer, dimensions = None):
    """
    Initalizes the recurrence relation with an initial hidden state
    if needed, else replaces with a "None" to tell Theano that
    the network **will** return something, but it does not need
    to send it to the next step of the recurrence
    """
    if dimensions is None:
        return layer.initial_hidden_state if has_hidden(layer) else None
    else:
        return matrixify(layer.initial_hidden_state, dimensions) if has_hidden(layer) else None
    
def initial_state_with_taps(layer, dimensions = None):
    """Optionally wrap tensor variable into a dict with taps=[-1]"""
    state = initial_state(layer, dimensions)
    if state is not None:
        return dict(initial=state, taps=[-1])
    else:
        return None

class Model:
    """
    Simple predictive model for forecasting words from
    sequence using LSTMs. Choose how many LSTMs to stack
    what size their memory should be, and how many
    words can be predicted.
    """
    def __init__(self, hidden_size, input_size, vocab_size, stack_size=1, celltype=LSTM, load_model=None):
        # declare model
        self.model = StackedCells(input_size, celltype=celltype, layers =[hidden_size] * stack_size)
        # add an embedding
        self.model.layers.insert(0, Embedding(vocab_size, input_size))
        # add a classifier:
        self.model.layers.append(Layer(hidden_size, vocab_size, activation = softmax))
        # inputs are matrices of indices,
        # each row is a sentence, each column a timestep
        self._stop_word   = theano.shared(np.int32(999999999), name="stop word")
        self.for_how_long = T.ivector()
        self.input_mat = T.imatrix()
        self.priming_word = T.iscalar()
        self.srng = T.shared_randomstreams.RandomStreams(np.random.randint(0, 1024))
        # create symbolic variables for prediction:
        self.predictions = self.create_prediction()
        # create symbolic variable for greedy search:
        self.greedy_predictions = self.create_prediction(greedy=True)
        # create gradient training functions:
        self.create_cost_fun()
        self.create_training_function()
        self.create_predict_function()
        if load_model is not None:
            self.model.params = pickle.load(open( load_model, "rb" ))

        #self.model.clip_gradients = True
        
    def stop_on(self, idx):
        self._stop_word.set_value(idx)
        
    @property
    def params(self):
        return self.model.params
                                 
    def create_prediction(self, greedy=False):
        def step(idx, *states):
            # new hiddens are the states we need to pass to LSTMs
            # from past. Because the StackedCells also include
            # the embeddings, and those have no state, we pass
            # a "None" instead:
            new_hiddens = [None] + list(states)
            
            new_states = self.model.forward(idx, prev_hiddens = new_hiddens)
            if greedy:
                new_idxes = new_states[-1]
                new_idx   = new_idxes.argmax()
                # provide a stopping condition for greedy search:
                return ([new_idx.astype(self.priming_word.dtype)] + new_states[1:-1]), theano.scan_module.until(T.eq(new_idx,self._stop_word))
            else:
                return new_states[1:]
        # in sequence forecasting scenario we take everything
        # up to the before last step, and predict subsequent
        # steps ergo, 0 ... n - 1, hence:
        inputs = self.input_mat[:, 0:-1]
        num_examples = inputs.shape[0]
        # pass this to Theano's recurrence relation function:
        
        # choose what gets outputted at each timestep:
        if greedy:
            outputs_info = [dict(initial=self.priming_word, taps=[-1])] + [initial_state_with_taps(layer) for layer in self.model.layers[1:-1]]
            result, _ = theano.scan(fn=step,
                                n_steps=200,
                                outputs_info=outputs_info)
        else:
            outputs_info = [initial_state_with_taps(layer, num_examples) for layer in self.model.layers[1:]]
            result, _ = theano.scan(fn=step,
                                sequences=[inputs.T],
                                outputs_info=outputs_info)
                                 
        if greedy:
            return result[0]
        # softmaxes are the last layer of our network,
        # and are at the end of our results list:
        return result[-1].transpose((2,0,1))
        # we reorder the predictions to be:
        # 1. what row / example
        # 2. what timestep
        # 3. softmax dimension
                                 
    def create_cost_fun (self):
        # create a cost function that
        # takes each prediction at every timestep
        # and guesses next timestep's value:
        what_to_predict = self.input_mat[:, 1:]
        # because some sentences are shorter, we
        # place masks where the sentences end:
        # (for how long is zero indexed, e.g. an example going from `[2,3)`)
        # has this value set 0 (here we substract by 1):
        for_how_long = self.for_how_long - 1
        # all sentences start at T=0:
        starting_when = T.zeros_like(self.for_how_long)
                                 
        self.cost = T.mean(masked_loss(self.predictions,
                                what_to_predict,
                                for_how_long,
                                starting_when))
        
    def create_predict_function(self):
        self.pred_fun = theano.function(
            inputs=[self.input_mat],
            outputs =self.predictions,
            allow_input_downcast=False
        )
        
        self.greedy_fun = theano.function(
            inputs=[self.priming_word],
            outputs=T.concatenate([T.shape_padleft(self.priming_word), self.greedy_predictions]),
            allow_input_downcast=False
        )
                                 
    def create_training_function(self):
        updates, _, _, _, _ = create_optimization_updates(self.cost, self.params, method="adadelta")
        self.update_fun = theano.function(
            inputs=[self.input_mat, self.for_how_long],
            outputs=self.cost,
            updates=updates,
            allow_input_downcast=False)
        
    def __call__(self, x):
        return self.pred_fun(x)

def main():
    # generate dataset                
    articles = Wiki_articles()
    vocabulary = Vocab(syllable2index = '../data/vocabulary_3K')

    print 'Build model ...'
        # construct model & theano functions:
    model = Model(
        input_size=1200,
        hidden_size=1200,
        vocab_size=len(vocabulary),
        stack_size=1, # make this bigger, but makes compilation slow
        celltype=LSTM, # use RNN or LSTM
        load_model = "model_backup.p",
    )

    #model.stop_on(vocabulary.syllable2index[u"."])
    
    try:
        #train
        print 'Training'
        time_0 = time()
        update_counter = 0
        article_counter = 0
        for article in articles():
            start = time()
            article_counter += 1
            sentences = article.split(u'.')
            numerical_lines = []
            for sentence in sentences:
                numerical_lines.append(vocabulary(sentence+u'.'))
            numerical_lines, numerical_lengths = pad_into_matrix(numerical_lines)
            error = model.update_fun(numerical_lines, numerical_lengths) 
            if article_counter % 100 == 0:
                print(vocabulary(model.greedy_fun(vocabulary.syllable2index[u"die"][0])))
            print 'Iteration: ',
            print article_counter,
            print  'Update duration: ',
            print  time()-start,
            print 'Total time: ',
            print (time()-time_0)/3600,
            print 'error: ',
            print error

    except KeyboardInterrupt:
        pickle.dump(model.model.params,open( "model_backup.p", "wb" ))
        try:
            pickle.dump(model.model.params,open( "model_backup.p", "wb" ))
            print "Model saved. Exiting ..."
        except: 
            print "Exiting without saving a model."
            


if __name__ == "__main__":

    main()   