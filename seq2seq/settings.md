Settings for Sutskever, Ilya, Oriol Vinyals, and Quoc V. Le. "[Sequence to sequence learning with neural networks.](https://arxiv.org/abs/1409.3215)" In proc. NIPS. 2014


# Setting
## Datasets
- WMT'14 **English** to **French** dataset
- 12M sentences consisting of 384M French words and 304M English words
- Use 160,000 of the most frequent words for the source language and 80,000 of the most frequent words for the target language
- Ever out-of-vocabulary (OOV) words was replaced with a special token "UNK" token.

## Decoding and Rescoring
- Use beam search

*To be added*

## Model and other settings
- LSTM learns much better when the source snetences are reversed (the target sentences are not reversed)
- Use deep LSTMs with 4 layers, with 1000 cells at each layer
- 1000 dimensional word embeddings
- Initialized all of the LSTM's parameters with the uniform distribution between -0.08 and 0.08
- Use **stochastic gradient descent (SGD)** without momentum, with fixed **learning rate** 0.7.
- After 5 epochs, Begin halving the learning rate every half epoch.
- Train the models for a total of 7.5 epochs
- Use **batches** of 128 sequences for the gradient and divided it the size of the batch
- Enforce hard constraint on the norm of the gradient by scaling it when its norm exceeded a threshold.
    - For each training batch, Compute L2-norm of gradients (s=||g||), where g is the gradient divided by 128. If s>5, set g = 5g/s
- Make sure that all sentences in a minibatch are roughly of the same length (How?)    
 
 # Experimental Results
 *To be added*
 
 # TODO
 - Reverse order of source language
 - Decay learning rate based on the paper. Currently, I just halves learning rate after 5 epoch
 - I must make model parallelizable because I don't have enough gpu memory to use 128 batch size
 - Make prediction function
 - Make BLEU score metric function