from pathlib import Path

from utils.config import chroma_config
from utils.path_tool import get_abs_path

_REVISION_FILE_NAME = ".index_revision"


def _revision_path() -> Path:
    persist_directory = Path(get_abs_path(chroma_config["persist_directory"]))
    return persist_directory / _REVISION_FILE_NAME


def get_index_revision() -> int:
    try:
        return _revision_path().stat().st_mtime_ns
    except FileNotFoundError:
        return 0


def mark_index_updated() -> None:
    revision_path = _revision_path()
    revision_path.parent.mkdir(parents=True, exist_ok=True)
    revision_path.touch()
