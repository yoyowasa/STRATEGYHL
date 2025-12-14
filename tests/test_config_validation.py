import pytest

from hlmm.config import ConfigError, load_config, normalize_config


def test_missing_required_key():
    # mode が無い場合は即例外
    with pytest.raises(ConfigError):
        normalize_config({"paths": {"data_dir": "data"}})


def test_type_mismatch_raises():
    with pytest.raises(ConfigError):
        normalize_config({"mode": "research", "paths": {"data_dir": 123}})
    with pytest.raises(ConfigError):
        normalize_config({"mode": 123, "paths": {"data_dir": "data"}})


def test_invalid_yaml_fails(tmp_path):
    bad_file = tmp_path / "bad.yaml"
    bad_file.write_text("mode: [unclosed", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(bad_file)
