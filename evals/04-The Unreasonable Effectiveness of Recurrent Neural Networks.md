# The Unreasonable Effectiveness of Recurrent Neural Networks

**Source:** https://karpathy.github.io/2015/05/21/rnn-effectiveness/
**Author:** Andrej Karpathy
**Date:** May 21, 2015

---

There's something magical about Recurrent Neural Networks (RNNs). I still remember when I trained my first recurrent network for Image Captioning. Within a few dozen minutes of training my first baby model started to generate very nice looking descriptions of images that were on the edge of making sense. Sometimes the ratio of how simple your model is to the quality of the results you get out of it blows past your expectations, and this was one of those times.

Together with this post I am also releasing [code on Github](https://github.com/karpathy/char-rnn) that allows you to train character-level language models based on multi-layer LSTMs.

## Recurrent Neural Networks

**Sequences.** A glaring limitation of Vanilla Neural Networks (and also Convolutional Networks) is that their API is too constrained: they accept a fixed-sized vector as input and produce a fixed-sized vector as output. The core reason that recurrent nets are more exciting is that they allow us to operate over *sequences* of vectors: Sequences in the input, the output, or in the most general case both.

RNN architectures support:
1. **One to one**: Vanilla mode, fixed-sized input to fixed-sized output (e.g. image classification)
2. **One to many**: Sequence output (e.g. image captioning)
3. **Many to one**: Sequence input (e.g. sentiment analysis)
4. **Many to many**: Sequence input and output (e.g. Machine Translation)
5. **Synced many to many**: (e.g. video classification)

> If training vanilla neural nets is optimization over functions, training recurrent nets is optimization over programs.

**RNN computation.** At the core, RNNs have a deceptively simple API: They accept an input vector x and give you an output vector y. However, this output vector's contents are influenced not only by the input you just fed in, but also on the entire history of inputs you've fed in in the past.

```python
class RNN:
  def step(self, x):
    # update the hidden state
    self.h = np.tanh(np.dot(self.W_hh, self.h) + np.dot(self.W_xh, x))
    # compute the output vector
    y = np.dot(self.W_hy, self.h)
    return y
```

The hidden state update: h_t = tanh(W_hh * h_{t-1} + W_xh * x_t)

## Character-Level Language Models

We'll train RNN character-level language models. We give the RNN a huge chunk of text and ask it to model the probability distribution of the next character in the sequence given a sequence of previous characters. This allows us to generate new text one character at a time.

## Fun with RNNs

### Paul Graham Generator
Trained on ~1MB of Paul Graham essays. A 2-layer LSTM with 512 hidden nodes generates plausible-sounding startup advice.

### Shakespeare
Trained on all works of Shakespeare (4.4MB). A 3-layer RNN generates text that's barely distinguishable from actual Shakespeare, complete with character names, stage directions, and iambic-ish rhythm.

### Wikipedia
Trained on the Hutter Prize 100MB dataset of raw Wikipedia. The model learns to generate valid markdown, create headings, lists, XML tags, and even hallucinate URLs.

### Algebraic Geometry (LaTeX)
Trained on a 16MB LaTeX source file on algebraic stacks/geometry. The resulting sampled LaTeX almost compiles, producing plausible-looking mathematical proofs and diagrams.

### Linux Source Code
Trained on 474MB of Linux kernel C code. The model generates syntactically near-correct C code with proper comments, includes, macros, function signatures, indentation, and even generates GNU license headers.

## Understanding What's Going On

### Evolution of Samples During Training
The model first discovers general word-space structure, then rapidly learns words (starting with short words, then longer ones). Topics and themes that span multiple words emerge only much later.

### Neuron Visualizations
Individual neurons in the hidden representation learn interpretable features:
- URL detection neurons (activate inside URLs, deactivate outside)
- Markdown environment tracking neurons (activate inside [[ ]])
- Position-tracking neurons (linear activation across scope boundaries)
- Quote detection neurons

## Further Reading

- [Minimal character-level RNN language model in Python/numpy](https://gist.github.com/karpathy/d4dee566867f8291f086) (~100 lines)
- [char-rnn code on Github](https://github.com/karpathy/char-rnn)

---

*For the full article with all code samples and visualizations, see the [original blog post](https://karpathy.github.io/2015/05/21/rnn-effectiveness/).*
