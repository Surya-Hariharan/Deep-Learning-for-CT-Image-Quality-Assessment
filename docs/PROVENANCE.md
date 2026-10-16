# Provenance and attribution (research record)

This project's architecture and training recipe (RadImageNet-pretrained ResNet50 backbone, global average pooling, dropout, one-unit dense head, Adam lr 1e-3, MSE, batch 64, 30 epochs)
follow the ResNet50 baseline of Ohashi et al., "Development of a No-Reference CT Image Quality Assessment Method Using RadImageNet Pre-trained Deep Learning Models" (this is the paper that the
original adaptation, historical experiments 001-003, was based on). This study adapts that baseline to the LDCT-IQAC labels (radiologist 0-4 scores rather than synthetic VIF targets) and adds a pairwise
ranking loss (Experiment B) and a heteroscedastic Gaussian head with split-conformal intervals (Experiment C). Source attribution must appear in any submitted paper's bibliography; it is deliberately kept out of
the public README, which describes the implementation, data, experiments and results on their own terms. No claim of novelty is made in this repository: the contribution is an empirical comparison under a fixed protocol.

Other sources: RadImageNet pretrained weights (BMEII-AI/RadImageNet; terms of the authors), the LDCT-IQAC challenge dataset (organizers' terms; not redistributed), split-conformal prediction (standard method).
