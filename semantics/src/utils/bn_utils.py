from pathlib import Path

from ArgCausalDisco.utils.data_utils import load_bnlearn_data_dag


def _looks_like_argcausaldisco(path: Path) -> bool:
    return (
        (path / "__init__.py").exists()
        and (path / "abapc.py").exists()
        and (path / "utils").is_dir()
    )


def _resolve_default_data_path() -> Path:
    gradual_root = Path(__file__).resolve().parents[2]
    embedded_root = gradual_root.parent
    if _looks_like_argcausaldisco(embedded_root):
        return embedded_root / "datasets"

    return gradual_root / "ArgCausalDisco" / "datasets"


def get_dataset(dataset_name='cancer',
                seed=42,
                sample_size=5000,
                data_path=None):
    if data_path is None:
        data_path = _resolve_default_data_path()

    X_s, B_true = load_bnlearn_data_dag(dataset_name,
                                        data_path,
                                        sample_size,
                                        seed=seed,
                                        print_info=True,
                                        standardise=True)
    return X_s, B_true
