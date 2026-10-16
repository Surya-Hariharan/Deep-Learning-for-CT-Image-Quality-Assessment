# Literature context

Only sources that were actually consulted are cited. Where a bibliographic detail could not be verified it is marked.
This is a short orientation, not a systematic review.

## Original method: Ohashi et al.

*"Development of a No-Reference CT Image Quality Assessment Method Using RadImageNet Pre-trained Deep Learning
Models"* (Ohashi et al.). Title and authors are as recorded in this repository's `docs/paper/`; **venue, year and DOI
could not be verified by web search**. As summarized in `docs/paper/` (from the project's working notes, not re-checked
against the primary text): a no-reference CT IQA regressor using RadImageNet-pretrained ResNet50 and InceptionResNetV2,
a Dropout-Dense(1)-Sigmoid head, Adam/MSE, evaluated with correlation after a five-parameter logistic mapping, with
labels from a VIF-based synthetic-degradation pipeline. Historical experiments 001-003 are an adaptation of the ResNet50
variant to LDCT-IQAC's human scores; they do not reproduce the paper's dataset or numbers.

## Pretraining: RadImageNet

Mei et al., *RadImageNet: An Open Radiologic Deep Learning Research Dataset for Effective Transfer Learning*,
Radiology: Artificial Intelligence 2022;4(5):e210315, DOI 10.1148/ryai.210315
([PMC9530758](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9530758/)). About 1.35 million annotated CT/MRI/ultrasound
images; reported gains over ImageNet pretraining on small transfer tasks. A 2023 letter in the same journal
([doi:10.1148/ryai.220253](https://pubs.rsna.org/doi/10.1148/ryai.220253)) reports a possible fold-handling issue in the authors' code
for their own fine-tuning experiments; this does not concern the released weights directly but is a reason not to
treat reported transfer numbers as settled. Weights/code: [BMEII-AI/RadImageNet](https://github.com/BMEII-AI/RadImageNet).

## Dataset and challenge

LDCTIQAC 2023 (MICCAI 2023): [challenge site](https://ldctiqac2023.grand-challenge.org/), data
[Zenodo 10.5281/zenodo.7833096](https://doi.org/10.5281/zenodo.7833096). The challenge and its six compared methods are
described in Lee et al., Medical Image Analysis 99 (2025) 103343 [as reported by a search result; not read]. A later
paper using the data, Celik et al., [*LMM-IQA: Image Quality Assessment for Low-Dose CT Imaging*](https://arxiv.org/html/2511.07298v1),
states the 1000/300 split, the 0-4 scale and the sum-of-correlations overall score. A 2026 arXiv preprint
([ClinReadNet, arXiv:2606.10372](https://arxiv.org/pdf/2606.10372)) is reported by a search snippet to reach
PLCC/SROCC/KROCC of 0.9507/0.9554/0.8629 on this dataset; **not read, not reproduced, and not comparable to our
numbers** without checking its protocol.

## Related ideas used here (general background, no new claims)

- Pairwise margin ranking losses are a standard learning-to-rank device; Experiment B applies one to IQA regression.
  Whether it helps this task is an empirical question tested here, not a published result we rely on.
- Heteroscedastic Gaussian regression (predicting mean and log-variance, trained with Gaussian NLL) is a standard way to
  model aleatoric uncertainty; split conformal prediction provides distribution-free marginal coverage under
  exchangeability. No specific reference was consulted for these, so none is cited.

## Contribution statement

The study's contribution is a controlled comparison of three objectives on one ResNet50 backbone with a frozen,
documented protocol and disclosed test-set exposure. It does not claim a new loss, a new architecture, or
state-of-the-art accuracy.
