"""LDCT-IQAC Dataset/DataLoader for the Ohashi-ResNet50 baseline.

Audited on-disk structure (see repository dataset audit, PART 2 of the task):

    data/training/image/*.tif      (1000 files, PIL mode 'F', 512x512, float32 in [0, 1])
    data/training/train.json       ({"0000.tif": 2.8, "0001.tif": 1.8, ...}, 1000 entries)

    data/testing/images/*.tiff     (300 files, PIL mode 'F', 512x512, float32 in [0, 1])
    data/testing/test.json         ({"test000.tiff": 2.6666..., ...}, 300 entries)

Confirmed by audit:
    - Every image file has exactly one JSON entry and vice versa (no
      orphans in either direction) in both splits.
    - No `.DS_Store` or other non-image files were present in either
      image directory; the loader nonetheless filters by extension
      (`.tif`/`.tiff`) defensively rather than trusting the directory
      listing to already be clean.
    - No duplicate images (exact pixel match) were found.
    - Score range is [0.0, 4.0] in both splits. Training scores are
      multiples of 1/5 (5 raters, integer 0-4 votes averaged); testing
      scores are multiples of 1/6 (6 raters). This is the RADIOLOGIST
      QUALITY SCORE used by LDCT-IQAC -- it is NOT the VIF metric used as
      the regression target in the original Ohashi paper. Score handling
      here (SCORE_MIN/SCORE_MAX, normalize/denormalize) is specific to
      this dataset and must not be conflated with Ohashi's VIF target.

IMPORTANT: the dataset is TIFF, not PNG. The task brief assumed PNG; the
loader is written around the format actually found on disk.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

VALID_EXTENSIONS = (".tif", ".tiff")

# Score range as determined by the PART 2 dataset audit (both train and
# test JSON files: min 0.0, max 4.0). This is a property of LDCT-IQAC's
# radiologist quality score, not an assumption -- see module docstring.
SCORE_MIN = 0.0
SCORE_MAX = 4.0

INPUT_SIZE = 224  # Ohashi ResNet50 input size (see ct_iqa.models.ohashi_resnet50).


def normalize_score(raw_score: float) -> float:
    """Map a raw LDCT-IQAC score in [SCORE_MIN, SCORE_MAX] to [0, 1] for the Sigmoid head.

    Implementation choice (documented, not from the paper): linear min-max
    normalization, `(y - SCORE_MIN) / (SCORE_MAX - SCORE_MIN)`. Chosen
    because it is the simplest transformation consistent with a bounded
    Sigmoid output and an evenly-spaced rating scale; no other mapping is
    described for this dataset in the source material available to this
    project.
    """
    return (raw_score - SCORE_MIN) / (SCORE_MAX - SCORE_MIN)


def denormalize_score(normalized_score: float | np.ndarray | torch.Tensor):
    """Inverse of `normalize_score`: map a Sigmoid output in [0, 1] back to [SCORE_MIN, SCORE_MAX]."""
    return normalized_score * (SCORE_MAX - SCORE_MIN) + SCORE_MIN


class LDCTIQACLabelError(ValueError):
    """Raised when the LDCT-IQAC image directory and JSON label file are inconsistent."""


@dataclass(frozen=True)
class LDCTIQACSample:
    filename: str
    path: Path
    raw_score: float


def _load_labels(json_path: Path) -> dict[str, float]:
    if not json_path.is_file():
        raise FileNotFoundError(f"LDCT-IQAC label file not found: {json_path}")

    with json_path.open("r", encoding="utf-8") as f:
        try:
            labels = json.load(f)
        except json.JSONDecodeError as e:
            raise LDCTIQACLabelError(f"Malformed JSON in {json_path}: {e}") from e

    if not isinstance(labels, dict):
        raise LDCTIQACLabelError(
            f"Expected {json_path} to contain a JSON object mapping filename -> score, "
            f"got top-level type {type(labels).__name__}"
        )

    for filename, score in labels.items():
        if not isinstance(filename, str):
            raise LDCTIQACLabelError(f"Non-string filename key in {json_path}: {filename!r}")
        if not isinstance(score, (int, float)) or isinstance(score, bool):
            raise LDCTIQACLabelError(
                f"Non-numeric score for {filename!r} in {json_path}: {score!r}"
            )

    return labels


def _list_image_files(image_dir: Path) -> set[str]:
    if not image_dir.is_dir():
        raise FileNotFoundError(f"LDCT-IQAC image directory not found: {image_dir}")
    return {
        p.name
        for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS
    }


class LDCTIQACDataset(Dataset):
    """PyTorch Dataset over the LDCT-IQAC training or testing split.

    Fails loudly (raises) rather than silently dropping samples if:
        - the image directory or JSON label file is missing
        - the JSON is malformed or has the wrong schema
        - any image file has no JSON label, or any JSON label has no
          matching image file
        - an image fails to open, or has an unsupported PIL mode

    Each `__getitem__` returns `(image_tensor, raw_score)`:
        image_tensor: float32 tensor, shape (1, image_size, image_size),
            resized from the source 512x512 image. Pixel values are passed
            through unchanged (source TIFFs are already float32 in [0, 1];
            see module docstring) -- no additional normalization is
            applied, since no Ohashi-specific normalization statistics are
            documented for this dataset. This is an explicit implementation
            choice, not a paper detail.
        raw_score: float32 tensor (scalar), the score AS STORED IN THE
            JSON, in [SCORE_MIN, SCORE_MAX]. Callers that need the [0, 1]
            Sigmoid-compatible target must call `normalize_score`
            explicitly (done in `ct_iqa.training.trainer`), keeping the
            dataset's label space and the model's output space decoupled.
    """

    def __init__(self, image_dir: str | Path, json_path: str | Path, image_size: int = INPUT_SIZE):
        self.image_dir = Path(image_dir)
        self.json_path = Path(json_path)
        self.image_size = image_size

        labels = _load_labels(self.json_path)
        image_files = _list_image_files(self.image_dir)

        label_keys = set(labels.keys())
        images_without_labels = image_files - label_keys
        labels_without_images = label_keys - image_files
        if images_without_labels or labels_without_images:
            raise LDCTIQACLabelError(
                f"Image/label mismatch between {self.image_dir} and {self.json_path}: "
                f"{len(images_without_labels)} image(s) with no label "
                f"(e.g. {sorted(images_without_labels)[:5]}), "
                f"{len(labels_without_images)} label(s) with no image "
                f"(e.g. {sorted(labels_without_images)[:5]})"
            )

        self.samples: list[LDCTIQACSample] = [
            LDCTIQACSample(filename=fn, path=self.image_dir / fn, raw_score=float(labels[fn]))
            for fn in sorted(labels.keys())
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[index]

        with Image.open(sample.path) as im:
            if im.mode != "F":
                raise LDCTIQACLabelError(
                    f"Unsupported image mode {im.mode!r} for {sample.path} "
                    f"(expected 'F' / 32-bit float grayscale, per dataset audit)"
                )
            im = im.resize((self.image_size, self.image_size), Image.BILINEAR)
            array = np.array(im, dtype=np.float32)

        image_tensor = torch.from_numpy(array).unsqueeze(0)  # (1, H, W)
        score_tensor = torch.tensor(sample.raw_score, dtype=torch.float32)
        return image_tensor, score_tensor

    @property
    def raw_scores(self) -> list[float]:
        return [s.raw_score for s in self.samples]
