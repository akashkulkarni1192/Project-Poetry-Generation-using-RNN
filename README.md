Project-Poetry-Generation-using-Rated-Recurrent-Unit
====================================================
Using RNN, created a language model with word embedding and later improved it using a Rated Recurrent Unit

Goal
----
Aim of the project is to build a language model using RNN (Recurrent Neural Network) and word embedding. This model will be trained on poems written by Robert Frost. Later, this model will be used to generate lines of poem in the style similar to Robert Frost's.

Dataset and data processing
---------------------------
There are 1581 lines of Robert Frost's poetry in robert_frost_poem.txt. Using this I built a vocabulary corpus and mapped each word to an index in a dictionary. Then I processed each sentence separately and generated a list of indexes of each corresponding word present in the sentence. This is then converted into a word vector where the length of the vector is of the vocabulary size. Each value in the word vector corresponds to the presence of the given word in the sentence i.e 1 if present, 0 otherwise. For example, "I can jump" would generate a vector V = [0,5,4,2,1] if the vocabulary dictionary is defined as D = {"jump":2, "can":4, "I":5, "sugar":6, "but":7}. Note that, each sentence is prepended with index 0 to indicate START and appended with 1 to indicate END.
Since the number of words in the vocabulary is huge, I converted the word vectors into word embedding using a word embedding matrix. These word embeddings have lesser dimensions than the corresponding word vector. These word embeddings is what is fed as an input the neural network. 

Implemention
------------
I implemented this in two ways
* **Simple Recurrent Unit**: 
* **Rated Recurrent Unit**
