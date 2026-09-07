from pathlib import Path

from scripts.prepare_splits import portable_dataset_path


def test_portable_dataset_path_is_not_absolute():
    stored = portable_dataset_path(Path("data/LSWMD.pkl"))
    assert stored == "data/LSWMD.pkl"
    assert not Path(stored).is_absolute()
