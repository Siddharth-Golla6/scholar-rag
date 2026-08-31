# Capsule Networks and Dynamic Routing

Capsule networks (CapsNets), introduced by Sabour, Frosst, and Hinton in 2017, represent visual
entities as groups of neurons called capsules. A capsule's activity vector encodes both the
probability that an entity is present and its instantiation parameters such as pose, orientation,
and scale. Unlike convolutional neural networks that rely on scalar feature detectors followed by
max-pooling, capsules preserve the spatial hierarchies between parts and wholes.

The key mechanism is dynamic routing by agreement. Lower-level capsules send their outputs to
higher-level capsules whose predictions agree with them, and this agreement is strengthened over a
few routing iterations. Routing replaces pooling, which otherwise discards spatial information.

As a result, capsule networks are more robust to affine transformations and viewpoint changes and
can generalize from fewer examples. Their main drawbacks are higher computational cost and
difficulty scaling to large, high-resolution images. Hybrid architectures often combine capsule
layers with convolutional feature extractors or with attention mechanisms to capture both local
features and global part-whole structure. In medical imaging, capsule-based models have been
explored for chest X-ray classification, where preserving spatial relationships between anatomical
structures can improve interpretability and reduce false negatives.
