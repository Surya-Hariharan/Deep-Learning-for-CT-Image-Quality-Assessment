"""Importing the package must have no side effects (no CLI parsing, no
training, no file creation, no dataset access)."""


def test_import_package():
    import quality_aware_lung_ct  # noqa: F401


def test_import_ngpnet():
    from quality_aware_lung_ct.ngpnet import NGPNet, NGPNetConfig  # noqa: F401


def test_import_training_and_evaluation():
    from quality_aware_lung_ct.training import NGPNetTrainer  # noqa: F401
    from quality_aware_lung_ct.evaluation import evaluate  # noqa: F401


def test_import_data_png():
    from quality_aware_lung_ct.data.png import PNGDataset, build_dataloaders  # noqa: F401


def test_import_boundary_modules():
    import quality_aware_lung_ct.quality  # noqa: F401
    import quality_aware_lung_ct.uncertainty  # noqa: F401
    import quality_aware_lung_ct.fusion  # noqa: F401
    import quality_aware_lung_ct.reporting  # noqa: F401


def test_config_import_has_no_side_effects(monkeypatch):
    ''' Regression test for the author's utils/config.py, which called
    `parser.parse_args()` at import time. '''
    monkeypatch.setattr("sys.argv", ["pytest", "--some-unrelated-flag"])
    import importlib
    import quality_aware_lung_ct.ngpnet.config as config_mod
    importlib.reload(config_mod)  # must not raise or parse sys.argv
