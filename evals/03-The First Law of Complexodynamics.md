# The First Law of Complexodynamics

**Source:** https://scottaaronson.blog/?p=762
**Author:** Scott Aaronson
**Date:** September 23, 2011

---

A few weeks ago, I had the pleasure of attending FQXi's Setting Time Aright conference, part of which took place on a cruise from Bergen, Norway to Copenhagen, Denmark. This conference brought together physicists, cosmologists, philosophers, biologists, psychologists, and (for some strange reason) one quantum complexity blogger to pontificate about the existence, directionality, and nature of time.

Sean Carroll delivered the opening talk of the conference, during which (among other things) he asked a beautiful question: *why does "complexity" or "interestingness" of physical systems seem to increase with time and then hit a maximum and decrease, in contrast to the entropy, which of course increases monotonically?*

My purpose, in this post, is to sketch a possible answer to Sean's question, drawing on concepts from Kolmogorov complexity.

## Background: The Second Law

We all know the Second Law, which says that the entropy of any closed system tends to increase with time until it reaches a maximum value. Here "entropy" somehow measures how "random" or "generic" or "disordered" a system is. The Second Law is almost a tautology: how could a system not tend to evolve to more "generic" configurations? If it didn't, those configurations wouldn't be generic! So the real question is not why the entropy is increasing, but why it was ever low to begin with—why did the universe's initial state at the big bang contain so much order for the universe's subsequent evolution to destroy?

The point that interests us is this: even though isolated physical systems get monotonically more entropic, they *don't* get monotonically more "complicated" or "interesting." Sean illustrated what he had in mind with the example of a coffee cup. Entropy increases monotonically from pure coffee to fully mixed, but intuitively, the "complexity" seems highest in the middle picture: the one with all the tendrils of milk. And same is true for the whole universe: shortly after the big bang, the universe was basically just a low-entropy soup of high-energy particles. A googol years from now, after the last black holes have sputtered away in bursts of Hawking radiation, the universe will basically be just a high-entropy soup of low-energy particles. But today, in between, the universe contains interesting structures such as galaxies and brains and hot-dog-shaped novelty vehicles.

## The Challenge

In answering Sean's provocative question, the challenge is twofold:

1. Come up with a plausible formal definition of "complexity."
2. Prove that the "complexity," so defined, is large at intermediate times in natural model systems, despite being close to zero at the initial time and close to zero at late times.

## Kolmogorov Complexity and Sophistication

Recall that the *Kolmogorov complexity* of a string x is the length of the shortest computer program that outputs x. In the 1970s, Kolmogorov made an observation closely related to Sean's observation above. A uniformly random string has close-to-maximal Kolmogorov complexity, but it's also one of the *least* "complex" or "interesting" strings imaginable. After all, we can describe essentially everything you'd ever want to know about the string by saying "it's random"!

This leads to the concept of *sophistication*. Given a set S of n-bit strings, let K(S) be the number of bits in the shortest computer program that outputs the elements of S. Then the sophistication of x, or Soph(x), is the smallest possible value of K(S), over all sets S such that:

1. x ∈ S and
2. K(x|S) ≥ log₂(|S|) – c, for some constant c. (In other words, one can distill all the "nonrandom" information in x just by saying that x belongs to S.)

Intuitively, Soph(x) is the length of the shortest computer program that describes, not necessarily x itself, but a set S of which x is a "random" or "generic" member.

## The Proposed Solution: Resource-Bounded Sophistication

I conjecture that the answers can be found using *resource-bounded* sophistication—what I call "complextropy":

*The number of bits in the shortest computer program that runs in n log(n) time, and that outputs a nearly-uniform sample from a set S such that (i) x∈S, and (ii) any computer program that outputs x in n log(n) time, given an oracle that provides independent, uniform samples from S, has at least log₂(|S|)-c bits.*

The key insight is imposing computational efficiency requirements in *two* places: on the sampling algorithm, and also on the algorithm that reconstructs x given the sampling oracle.

I *conjecture* that the complextropy will satisfy the "First Law of Complexodynamics," exhibiting exactly the behavior that Sean wants: small for the initial state, large for intermediate states, then small again once the mixing has finished.

This is not a hopelessly open-ended question but a relatively-bounded question about which actual theorems could be proved and actual papers published.

---

*For the full article with comments and discussion, see the [original blog post](https://scottaaronson.blog/?p=762).*
