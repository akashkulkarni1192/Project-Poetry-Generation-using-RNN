import theano
import theano.tensor as T
import numpy as np
import matplotlib.pyplot as plt
import util as myutil
from sklearn.utils import shuffle


class RNN_class(object):
    def __init__(self, embedding_size, hidden_layer_size, vocabulary_size):
        self.D = embedding_size
        self.M = hidden_layer_size
        self.V = vocabulary_size

    def fit(self, X, learning_rate=10e-1, mu=0.99, reg=1.0, activation=T.tanh, epochs=50):
        N = len(X)
        D = self.D
        M = self.M
        V = self.V

        # initialize all required Weights
        We = myutil.init_weights(V, D)  # We indicates Word encodings
        Wx = myutil.init_weights(D, M)
        h0 = np.zeros(M)
        Wh = myutil.init_weights(M, M)
        bh = np.zeros(M)
        Wo = myutil.init_weights(M, V)
        bo = np.zeros(V)

        self.set(We, Wh, Wo, Wx, bh, bo, h0, activation, learning_rate, mu)

        costs = []
        n_total = sum((len(sentence) + 1) for sentence in X)
        for i in range(epochs):
            X = shuffle(X)
            n_correct = 0
            cost = 0
            for j in range(N):
                input_sentence = [0] + X[j]
                output_sentence = X[j] + [1]

                c, pY = self.train_op(input_sentence, output_sentence)
                cost += c
                for pj, yj in zip(pY, output_sentence):
                    if pj == yj:
                        n_correct += 1
            print("i: {0} cost: {1} correct_rate: {2}".format(i, cost, (float(n_correct) / n_total)))
            costs.append(cost)
        plt.plot(costs)

    def set(self, We, Wh, Wo, Wx, bh, bo, h0, activation, learning_rate=10e-1, mu=0.99):
        self.f = activation
        self.lr = learning_rate
        self.We = theano.shared(We)
        self.Wx = theano.shared(Wx)
        self.Wh = theano.shared(Wh)
        self.bh = theano.shared(bh)
        self.h0 = theano.shared(h0)
        self.Wo = theano.shared(Wo)
        self.bo = theano.shared(bo)
        self.params = [self.We, self.Wx, self.Wh, self.bh, self.h0, self.Wo, self.bo]
        thX = T.ivector('X')
        EncX = self.We[thX]  # To get word encoding of the words in the sentence, Ei will be TxD
        thY = T.ivector('Y')

        def recurrence(x, prev_h):
            print(x)
            h = self.f(x.dot(self.Wx) + prev_h.dot(self.Wh) + self.bh)
            y = T.nnet.softmax(h.dot(self.Wo) + self.bo)
            return h, y

        [h, y], _ = theano.scan(
            fn=recurrence,
            outputs_info=[self.h0, None],
            sequences=EncX,
            n_steps=EncX.shape[0],
        )
        py_x = y[:, 0, :]
        prediction = T.argmax(py_x, axis=1)
        cost = -T.mean(T.log(py_x[T.arange(thY.shape[0]), thY]))
        grads = T.grad(cost, self.params)
        dparams = [theano.shared(p.get_value() * 0) for p in self.params]
        updates = [
                      (p, p + mu * dp - learning_rate * g) for p, dp, g in zip(self.params, dparams, grads)
                  ] + [
                      (dp, mu * dp - learning_rate * g) for dp, g in zip(dparams, grads)
                  ]
        self.predict_op = theano.function(inputs=[thX], outputs=prediction, allow_input_downcast=True)
        self.train_op = theano.function(
            inputs=[thX, thY],
            outputs=[cost, prediction],
            updates=updates
        )

    def save(self, filename):
        np.savez(filename, *[p.get_value() for p in self.params] + [self.f, self.lr])

    @staticmethod
    def load(filename, activation, learning_rate):
        npz = np.load(filename)
        We = npz['arr_0']
        Wx = npz['arr_1']
        Wh = npz['arr_2']
        bh = npz['arr_3']
        h0 = npz['arr_4']
        Wo = npz['arr_5']
        bo = npz['arr_6']
        V, D = We.shape
        _, M = Wx.shape
        rnn = RNN_class(D, M, V)
        rnn.set(We, Wh, Wo, Wx, bh, bo, h0, activation=activation)
        return rnn

    def generate(self, pi, word2idx_map, lines_generate = 4):
        # We need to convert word2indexmap to index2wordmap
        # Because process is we will be picking random index from vocabulary and precict next index until we get end i.e ind=1
        # For each index, we will be printing its word value, hence index2wordmap is needed
        idx2word_map = {value: key for key, value in word2idx_map.items()}
        V = len(pi)

        n_lines = 0

        # Randomly pick start word based on its state distribution other wise same start word will lead mostly to same sentence
        X = [np.random.choice(V, p=pi)]
        # print starting word of the first line
        print(idx2word_map[X[0]] + " ", end="")
        prev_word = -99
        while n_lines < lines_generate:
            P = self.predict_op(X)[-1]
            if(prev_word == 1 and P == 1):
                continue    # To avoid generating blank lines
            else:
                prev_word = P
            X += [P]
            if P > 1:
                # it's a real word, not start/end token
                word = idx2word_map[P]
                print(word + " ", end="")
            elif P == 1:
                # end token
                n_lines += 1
                print('')
                if n_lines < lines_generate:
                    X = [np.random.choice(V, p=pi)]  # reset to start of line
                    print(idx2word_map[X[0]] + " " , end="")


def train_poetry_model():
    sentences, word2indmap = myutil.get_robert_frost_word_indexes()
    rnn_model = RNN_class(30, 30, len(word2indmap))
    rnn_model.fit(sentences, learning_rate=10e-5, activation=T.nnet.relu, epochs=1000)
    rnn_model.save('RNN_trained_model.npz')


def generate_poetry():
    sentences, word2idx_map = myutil.get_robert_frost_word_indexes()
    rnn = RNN_class.load('RNN_trained_model.npz', learning_rate=10e-5, activation=T.nnet.relu)

    V = len(word2idx_map)
    # maintain initial state/probability distribution table of each starting word of a sentence
    # this will help in picking the starting word though randomly but considering each starting word's probability for generating lines
    pi = np.zeros(V)
    for sentence in sentences:
        pi[sentence[0]] += 1
    pi /= pi.sum()
    rnn.generate(pi, word2idx_map, lines_generate=10)


if __name__ == '__main__':
    train_poetry_model()
    generate_poetry()
