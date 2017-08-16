Project-Poetry-Generation-using-Rated-Recurrent-Unit
====================================================
Using RNN, created a language model with word embedding and later improved it using a Rated Recurrent Unit
Goal
----
Aim of the project is to build a language model using RNN (Recurrent Neural Network) and word embedding. This model will be trained on poems written by Robert Frost. Later, this model will be used to generate lines of poem in the style similar to Robert Frost's.

Dataset and data processing
---------------------------
There are 1581 lines of Robert Frost's poetry in robert_frost_poem.txt. Using this I built a vocabulary corpus and mapped each word to an index in a dictionary. Then I processed each sentence separately and generated a list of indexes of each corresponding word present in the sentence.
