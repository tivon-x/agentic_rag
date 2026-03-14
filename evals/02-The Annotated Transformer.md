# The Annotated Transformer

**Source:** https://nlp.seas.harvard.edu/2018/04/03/attention.html
**Author:** Alexander Rush (Harvard NLP)
**Date:** Apr 3, 2018

---

**Note:** There is now a [new version](http://harvardnlp.github.io/annotated-transformer) of this blog post updated for modern PyTorch.

---

The Transformer from "Attention is All You Need" has been on a lot of people's minds over the last year. Besides producing major improvements in translation quality, it provides a new architecture for many other NLP tasks. The paper itself is very clearly written, but the conventional wisdom has been that it is quite difficult to implement correctly.

In this post I present an "annotated" version of the paper in the form of a line-by-line implementation. I have reordered and deleted some sections from the original paper and added comments throughout. This document itself is a working notebook, and should be a completely usable implementation. In total there are 400 lines of library code which can process 27,000 tokens per second on 4 GPUs.

To follow along you will first need to install PyTorch. The complete notebook is also available on [github](https://github.com/harvardnlp/annotated-transformer) or on Google Colab with free GPUs.

Note this is merely a starting point for researchers and interested developers. The code here is based heavily on our OpenNMT packages.

## Background

The goal of reducing sequential computation also forms the foundation of the Extended Neural GPU, ByteNet and ConvS2S, all of which use convolutional neural networks as basic building block, computing hidden representations in parallel for all input and output positions. In these models, the number of operations required to relate signals from two arbitrary input or output positions grows in the distance between positions, linearly for ConvS2S and logarithmically for ByteNet. This makes it more difficult to learn dependencies between distant positions. In the Transformer this is reduced to a constant number of operations, albeit at the cost of reduced effective resolution due to averaging attention-weighted positions, an effect we counteract with Multi-Head Attention.

Self-attention, sometimes called intra-attention is an attention mechanism relating different positions of a single sequence in order to compute a representation of the sequence. Self-attention has been used successfully in a variety of tasks including reading comprehension, abstractive summarization, textual entailment and learning task-independent sentence representations.

To the best of our knowledge, however, the Transformer is the first transduction model relying entirely on self-attention to compute representations of its input and output without using sequence aligned RNNs or convolution.

## Model Architecture

Most competitive neural sequence transduction models have an encoder-decoder structure. Here, the encoder maps an input sequence of symbol representations to a sequence of continuous representations. Given z, the decoder then generates an output sequence of symbols one element at a time. At each step the model is auto-regressive, consuming the previously generated symbols as additional input when generating the next.

```python
class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, src_mask), src_mask,
                            tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)
```

The Transformer follows this overall architecture using stacked self-attention and point-wise, fully connected layers for both the encoder and decoder.

### Encoder and Decoder Stacks

The encoder is composed of a stack of N=6 identical layers. Each layer has two sub-layers: multi-head self-attention mechanism and position-wise fully connected feed-forward network. We employ a residual connection around each of the two sub-layers, followed by layer normalization.

The decoder is also composed of a stack of N=6 identical layers. In addition to the two sub-layers in each encoder layer, the decoder inserts a third sub-layer, which performs multi-head attention over the output of the encoder stack.

### Attention

An attention function can be described as mapping a query and a set of key-value pairs to an output. We call our particular attention "Scaled Dot-Product Attention":

Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V

Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions.

### Position-wise Feed-Forward Networks

Each of the layers in our encoder and decoder contains a fully connected feed-forward network: FFN(x) = max(0, xW1 + b1)W2 + b2

### Positional Encoding

Since our model contains no recurrence and no convolution, we add "positional encodings" to the input embeddings using sine and cosine functions of different frequencies.

## Training

We used the Adam optimizer with a warmup learning rate schedule, and label smoothing regularization.

---

*For the complete implementation with code, see the [full article](https://nlp.seas.harvard.edu/2018/04/03/attention.html) or the [updated version](http://harvardnlp.github.io/annotated-transformer).*
