# Understanding LSTM Networks

**Source:** https://colah.github.io/posts/2015-08-Understanding-LSTMs/
**Author:** Christopher Olah
**Date:** August 27, 2015

---

## Recurrent Neural Networks

Humans don't start their thinking from scratch every second. As you read this essay, you understand each word based on your understanding of previous words. You don't throw everything away and start thinking from scratch again. Your thoughts have persistence.

Traditional neural networks can't do this, and it seems like a major shortcoming. Recurrent neural networks address this issue. They are networks with loops in them, allowing information to persist.

A recurrent neural network can be thought of as multiple copies of the same network, each passing a message to a successor. This chain-like nature reveals that recurrent neural networks are intimately related to sequences and lists.

Essential to these successes is the use of "LSTMs," a very special kind of recurrent neural network which works, for many tasks, much much better than the standard version. Almost all exciting results based on recurrent neural networks are achieved with them.

## The Problem of Long-Term Dependencies

Sometimes, we only need to look at recent information to perform the present task. For example, predicting the next word in "the clouds are in the *sky*" — it's pretty obvious the next word is going to be sky.

But there are also cases where we need more context. Consider trying to predict the last word in "I grew up in France… I speak fluent *French*." As that gap grows, RNNs become unable to learn to connect the information.

In theory, RNNs are absolutely capable of handling such "long-term dependencies." Sadly, in practice, RNNs don't seem to be able to learn them. The problem was explored in depth by Hochreiter (1991) and Bengio, et al. (1994).

Thankfully, LSTMs don't have this problem!

## LSTM Networks

Long Short Term Memory networks — usually just called "LSTMs" — are a special kind of RNN, capable of learning long-term dependencies. They were introduced by Hochreiter & Schmidhuber (1997), and were refined and popularized by many people in following work.

LSTMs are explicitly designed to avoid the long-term dependency problem. Remembering information for long periods of time is practically their default behavior, not something they struggle to learn!

All recurrent neural networks have the form of a chain of repeating modules. In standard RNNs, this repeating module will have a very simple structure, such as a single tanh layer. LSTMs also have this chain like structure, but the repeating module has a different structure — instead of having a single neural network layer, there are four, interacting in a very special way.

## The Core Idea Behind LSTMs

The key to LSTMs is the **cell state**, the horizontal line running through the top of the diagram. The cell state is kind of like a conveyor belt. It runs straight down the entire chain, with only some minor linear interactions. It's very easy for information to just flow along it unchanged.

The LSTM does have the ability to remove or add information to the cell state, carefully regulated by structures called **gates**. Gates are composed of a sigmoid neural net layer and a pointwise multiplication operation.

An LSTM has three of these gates:

### Step 1: Forget Gate
Decides what information to throw away from the cell state. It looks at h_{t-1} and x_t, and outputs a number between 0 and 1 for each number in the cell state C_{t-1}. A 1 represents "completely keep this" while a 0 represents "completely get rid of this."

### Step 2: Input Gate
Decides what new information to store in the cell state. A sigmoid layer (the "input gate layer") decides which values we'll update. A tanh layer creates a vector of new candidate values that could be added to the state.

### Step 3: Update Cell State
We multiply the old state by f_t (forgetting things we decided to forget), then add i_t * C̃_t (the new candidate values, scaled by how much we decided to update each state value).

### Step 4: Output Gate
We run a sigmoid layer which decides what parts of the cell state we're going to output. Then we put the cell state through tanh and multiply it by the output of the sigmoid gate.

## Variants on Long Short Term Memory

### Peephole Connections
Adding "peephole connections" means letting the gate layers look at the cell state (Gers & Schmidhuber, 2000).

### Coupled Forget and Input Gates
Instead of separately deciding what to forget and what to add, we make those decisions together.

### Gated Recurrent Unit (GRU)
Introduced by Cho, et al. (2014), the GRU combines the forget and input gates into a single "update gate" and merges the cell state and hidden state. The resulting model is simpler than standard LSTM models.

Greff, et al. (2015) compared popular variants and found they're all about the same. Jozefowicz, et al. (2015) tested more than ten thousand RNN architectures, finding some that worked better than LSTMs on certain tasks.

## Conclusion

LSTMs were a big step in what we can accomplish with RNNs. The natural next step is **attention** — letting every step of an RNN pick information to look at from some larger collection of information.

---

*For the full article with diagrams and visualizations, see the [original blog post](https://colah.github.io/posts/2015-08-Understanding-LSTMs/).*
