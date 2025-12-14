import yaml

from hlmm.config import normalize_config


def test_defaults_are_reproducible():
    raw = {"mode": "research", "paths": {"data_dir": "data"}}
    cfg1 = normalize_config(raw)
    cfg2 = normalize_config(raw)

    assert cfg1.to_dict() == cfg2.to_dict()
    assert yaml.safe_dump(cfg1.to_dict(), sort_keys=True) == yaml.safe_dump(
        cfg2.to_dict(), sort_keys=True
    )

    assert cfg1.paths.output_dir == "outputs"
    assert cfg1.paths.log_dir == "logs"
    assert cfg1.strategy.leverage == 1.0
    assert cfg1.strategy.max_positions == 10
