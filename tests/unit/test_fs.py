import os

import pytest

from ct_iqa.utils.fs import replace_with_retry


def test_replace_with_retry_succeeds_after_transient_permission_error(tmp_path, monkeypatch):
    src, dst = tmp_path / "a.tmp", tmp_path / "a"
    src.write_text("x")
    real, calls = os.replace, {"n": 0}

    def flaky(s, d):
        calls["n"] += 1
        if calls["n"] < 3:
            raise PermissionError("locked")
        return real(s, d)

    monkeypatch.setattr(os, "replace", flaky)
    replace_with_retry(src, dst, attempts=5, delay=0)
    assert dst.read_text() == "x" and calls["n"] == 3


def test_replace_with_retry_reraises_persistent_error(tmp_path, monkeypatch):
    (tmp_path / "a.tmp").write_text("x")
    monkeypatch.setattr(os, "replace", lambda s, d: (_ for _ in ()).throw(PermissionError("locked")))
    with pytest.raises(PermissionError):
        replace_with_retry(tmp_path / "a.tmp", tmp_path / "a", attempts=3, delay=0)
