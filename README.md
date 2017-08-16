Project-Poetry-Generation-using-Rated-Recurrent-Unit
====================================================
Using RNN, created a language model with word embedding and later improved it using a Rated Recurrent Unit

Goal
----
Aim of the project is to build a language model using RNN (Recurrent Neural Network) and word embedding. This model will be trained on poems written by Robert Frost. Later, this model will be used to generate lines of poem in the style similar to Robert Frost's.

Dataset and data processing
---------------------------
There are 1581 lines of Robert Frost's poetry in robert_frost_poem.txt. Using this I built a vocabulary corpus and mapped each word to an index in a dictionary. Then I processed each sentence separately and generated a list of indexes of each corresponding word present in the sentence. This is then converted into a list of the tokens i.e. word vector where the length of the vector is of the vocabulary size. Each token corresponds to the presence of the given word in the sentence i.e 1 if present, 0 otherwise. For example, "I can jump" would generate a vector V = [0,5,4,2,1] if the vocabulary dictionary is defined as D = {"jump":2, "can":4, "I":5, "sugar":6, "but":7}. Note that, each sentence is prepended with index 0 to indicate START token and appended with 1 to indicate END token.

Since the number of words in the vocabulary is huge, I converted the word vectors into word embedding using a word embedding matrix. These word embeddings have lesser dimensions than the corresponding word vector. These word embeddings is what is fed as an input the neural network. 

Implemention
------------
I implemented this in two ways
* **Simple Recurrent Unit :**  In this model, I used 1 Word embedding layer of size 30 followed by 1 hidden RNN layer of size 30. Once the model is trained, I created an initial word distribution table (which stores the probability of the all the initial word starting the sentence). The model then picks a random initial word using this table to generate a line of poetry. There were **two limitations** of this model
  * Sentences that were generated are very short, because there are many words whose immediate next word is END i.e the END token was over-represented.
  * Same sentences were generated most of the times, since the probability of the initial words(of any sentence) in the distribution is much higher than other non-initial words
* **Rated Recurrent Unit:** In this model, I used a rated recurrent unit with a rate matrix insted of simple recurrent unit. The **two limitations were solved** in the following ways:
  * While generating the lines, the END token was added only 10% of the times. This decreased the probability of occurence of END token increasing the length of the sentences.
  * Instead of using initial word distribution table, I used the output probability of the softmax function at the output layer as the distribution. This increased the variety of the sentences being generated.
