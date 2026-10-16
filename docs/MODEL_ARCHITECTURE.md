# Model architecture and pipelines

Diagrams use module and function names that exist in the code (checked 2026-10-16 against
`src/ct_iqa/`). Training-only components are marked.

## 1. Shared backbone (`models.backbone.ResNet50Backbone`)

```mermaid
flowchart TD
    A["Input (B,3,224,224)"] --> B["stem: Conv7x7 s2, BN(eps=bn_eps), ReLU, MaxPool3x3 s2"]
    B --> C["stage1: 3 bottlenecks, 256 ch"]
    C --> D["stage2: 4 bottlenecks, 512 ch, stride on first 1x1"]
    D --> E["stage3: 6 bottlenecks, 1024 ch"]
    E --> F["stage4: 3 bottlenecks, 2048 ch"]
    F --> G["AdaptiveAvgPool2d(1) + flatten"]
    G --> H["features (B,2048)"]
```

53 BatchNorm layers, all constructed with the same `bn_eps` (1.001e-5 in study v2).

## 2. Experiment A: regression forward pass (`models.registry.ResNet50Model`, head kind `regression`)

```mermaid
flowchart TD
    X["image (B,1,224,224), values in [-1,1]"] --> R["repeat to 3 channels (in forward)"]
    R --> BB["ResNet50Backbone"]
    BB --> DO["dropout (p=0.5)"]
    DO --> FC["fc: Linear(2048,1)"]
    FC --> SG["sigmoid"]
    SG --> M["mean in [0,1]  (ModelOutput.mean)"]
    M --> DN["denormalize_score: x4 -> [0,4] (evaluation)"]
```

State-dict keys: `backbone.*`, `fc.weight`, `fc.bias`, identical to the historical `OhashiResNet50`
(verified structurally for all six checkpoints, see [audit/CHECKPOINT_COMPATIBILITY.md](audit/CHECKPOINT_COMPATIBILITY.md)).

## 3. Experiment B: ranking-aware training path (training only)

```mermaid
flowchart TD
    BATCH["mini-batch: images, raw scores y in [0,4]"] --> MODEL["resnet50_regression -> mean f in [0,1]"]
    BATCH --> NORM["normalize_score: y/4"]
    MODEL --> MSE["MSE(f, y/4)"]
    NORM --> MSE
    BATCH --> MASK["data.sampling.ordered_pair_mask(y, min_score_diff): B x B bool"]
    MODEL --> HINGE["hinge: max(0, margin - (f_i - f_j)) over masked pairs"]
    MASK --> HINGE
    MSE --> TOT["total = MSE + lambda * mean(hinge)"]
    HINGE --> TOT
    TOT --> BWD["backward + Adam step"]
```

At inference the ranking branch does not exist; the model is the same as in Experiment A.
With `lambda_rank = 0` the total equals the MSE term (unit-tested value and gradient).

## 4. Experiment C: mean / log-variance model and calibration

```mermaid
flowchart TD
    X["image (B,1,224,224)"] --> BB["ResNet50Backbone -> features"]
    BB --> DO["dropout (p=0.5)"]
    DO --> FM["fc: Linear(2048,1) + sigmoid -> mu_n in [0,1]"]
    DO --> FV["fc_logvar: Linear(2048,1) + clamp[-10,2] -> s_n"]
    FM --> NLL["training only: 0.5*(exp(-s_n)*(y_n-mu_n)^2 + s_n)"]
    FV --> NLL
    FM --> CONV["to_score_scale: mu = 4*mu_n, sigma = 4*exp(s_n/2)"]
    FV --> CONV
    CONV --> CAL["calibration partition: r_i = abs(y_i-mu_i)/sigma_i; q = k-th smallest, k=ceil((n+1)*0.9)"]
    CAL --> IV["interval mu +/- q*sigma (unclipped and clipped to [0,4])"]
```

A single deterministic head models aleatoric (data) uncertainty only; no epistemic uncertainty is estimated.

## 5. End-to-end pipeline and data access (`experiment.run.run_experiment`)

```mermaid
flowchart LR
    CFG["configs/base.yaml + experiment yaml"] --> LD["config.loader.load_study_config"]
    LD --> RUN["experiment.run.run_experiment"]
    MAN["split_manifest.json (frozen)"] --> RUN
    RUN --> TR["train partition (800): gradient updates"]
    RUN --> VA["validation partition (100): checkpoint selection, tuning"]
    TR --> ENG["training.engine.fit"]
    VA --> ENG
    ENG --> BEST["checkpoints/best.pt (validation-selected)"]
    BEST --> VMET["metrics_validation.json, predictions_validation.csv"]
    BEST --> CALP["calibration partition (100): Experiment C only"]
    CALP --> CALJ["calibration.json (q, coverage on validation)"]
    BEST -.->|"final-test: needs FROZEN marker and acknowledgement"| TEST["official test (300): retrospective only"]
```

The dashed edge is the only path to the official test set; `run_experiment` and `fit` contain no reference to it
(regression-tested).

## 6. Data splitting and checkpoint selection

```mermaid
flowchart TD
    ALL["1000 labeled training images"] --> PERM["numpy default_rng(2026).permutation(sorted names)"]
    PERM --> T["800 train"]
    PERM --> V["100 validation"]
    PERM --> C["100 calibration"]
    T --> UPD["optimizer updates"]
    V --> SEL["per-epoch val PLCC (raw 0-4) -> Selector (strict improvement, 1-based epochs)"]
    C --> CONF["split-conformal quantile (C only)"]
    OFFICIAL["300 official test images"] -.-> RETRO["one retrospective evaluation after protocol freeze"]
```

## 7. Experiment comparison workflow

```mermaid
flowchart LR
    A["Run A"] --> CMP["notebooks/06_comparative_analysis: paired bootstrap over validation images"]
    B["Run B (and lambda=0 ablation = A)"] --> CMP
    C["Run C (point predictions)"] --> CMP
    CMP --> DEC["differences with CIs; no winner on sub-CI differences"]
    DEC --> FREEZE["protocol frozen"] --> RETRO["retrospective test evaluation"]
```
