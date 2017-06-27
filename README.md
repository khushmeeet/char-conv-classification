# char-conv-classification
Character level convolutional neural network for text classification

# Link to the paper
[Character-level Convolutional Networks for Text
Classification](https://arxiv.org/pdf/1509.01626.pdf)

# Network Architecture
This architecture deviates slightly from the network specified in the paper. It started as a character conv, but soon it moved to word2vec, because of the lack of powerful system. It is 9 layer convolutional neural network. Convolution layers has number of filters equal to 256. Number of units at the output are 4 (depends on the dataset).
Here is the architecture table taken from the paper.
[archi1](images/conv.png)
[archi2](images/dense.png)

# Results
