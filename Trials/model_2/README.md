# Model 2: BPE to Char encoding model

The files and structure of this code are kept as close to model\_1 as possible.

## Source of the code:

https://github.com/spro/practical-pytorch/blob/master/seq2seq-translation/seq2seq-translation-batched.ipynb

# Documentation of the code:

## 1) Data Preparation:

- The code assumes that the data is stored in a path = "../eng-fre.txt". (We can modify this based on our purpose.) In the text file, every sentence contains sentence in both the languages (tab separated). 
- The "Lang" class stores the text in a dict data structure, word2index. This basically indexes every unique word in the language.
- Mainly three prepocessing steps are being done (can be optional)
	- Trimming sentences with words with very less frequency. (maybe useful when the text contains very uncommon names).
	- Removes the non-letter characters and induces a segments a list of punctuation. Ex: "New York City .,?" ==> "New York City . , ?"
	- Removing sentences with length < 3 or > 50 (This was the threshold they used in the paper. )

- Next, every sentence is read in order and pairs are formed as (sentence_source_version, sentence_target_version)


## 2) Data Representation:

- All the words in the sentence is converted to numbers (based on the index) and EOS symbol is attached at the end of the sentence.
- Then, a batch of sentences is selected randomnly based on batch size. Note that, since the selection is being done randomnly, we can have multiple instances of same sentence and zero instances of some sentences.
- In scenarios where one sentence is larger than the translated sentence (in the training data), we PAD the remaining words in the shorter sentence with zero.

## 3) Model Structure:

- The Encoders and Decoders (with Attention) are programmed as described in this paper (https://arxiv.org/abs/1508.04025).

## 4) Loss Function:

- The loss function is basically a softmax function which tries to maximize the probability of the predicted word (marked_cross_entropy.py). 
- The loss function we need is not entirely available in Pytorch (as the loss function needs to be masked with an indicator matrix that neglects losses for words > length of the sentence. Remember that each sentence was reshaped to have length equal to the max_length in the batch) and was programmed manually by the author. I have included a detailed mechanism of loss function in the script: (https://github.com/sl4964/MICA_NeuralMachineTranslation/blob/master/model_1/masked_cross_entropy.py)


## 5) Things to add:

- Pervasive dropout inclusion
- Add Bleu score to measure the performance of the trained model.
