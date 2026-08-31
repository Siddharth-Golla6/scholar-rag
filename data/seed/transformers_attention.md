# Transformers and Self-Attention

The Transformer architecture, introduced in "Attention Is All You Need" by Vaswani et al. in 2017,
replaced recurrence and convolution with self-attention as the primary mechanism for modeling
sequences. Self-attention computes a weighted sum over all positions in a sequence, where the
weights come from the similarity between query and key vectors. Every token can therefore attend
directly to every other token and capture long-range dependencies in a single layer, regardless of
distance, which recurrent networks struggle to do because of vanishing gradients.

Multi-head attention runs several attention operations in parallel so the model can attend to
different representation subspaces at once. Because attention is otherwise permutation-invariant,
positional encodings are added to inject information about token order.

Transformers parallelize well across sequence positions, which made training on very large corpora
practical and drove the rise of large language models. Beyond text, Vision Transformers (ViT) split
an image into patches and treat them as tokens, achieving strong results on image classification and
increasingly on medical imaging tasks. Transformers are often combined with convolutional or capsule
feature extractors in hybrid models to balance local detail with global context.
