# Grad-CAM and Model Interpretability

Gradient-weighted Class Activation Mapping (Grad-CAM), introduced by Selvaraju et al. in 2017, is a
technique for explaining the predictions of convolutional neural networks. It uses the gradients of a
target class flowing into the final convolutional layer to produce a coarse localization heat map that
highlights the regions of an input image most responsible for the prediction. Because it operates on
the last convolutional feature maps, Grad-CAM works with a wide range of CNN architectures without
modifying or retraining them.

In medical imaging, Grad-CAM heat maps are overlaid on chest X-rays or scans to show which anatomical
regions drove a diagnosis. This helps clinicians verify that the model attends to disease-relevant
areas rather than spurious cues such as text markers or imaging artifacts, which is central to
explainable AI (XAI) and to building clinical trust.

Grad-CAM also enables weakly supervised localization: models trained only on image-level labels can
still produce region-level heat maps, avoiding the cost of pixel-level annotation. Variants such as
Grad-CAM++ and axiom-based Grad-CAM improve localization sharpness and theoretical grounding.
