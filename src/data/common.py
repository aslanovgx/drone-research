import os
import random
from pathlib import Path

import yaml
from PIL import Image

DEFAULT_DATASET_ENV = "DRONE_DATASET_ROOT"
DEFAULT_OUTPUT_ENV = "DRONE_OUTPUT_ROOT"


class DatasetError(Exception):
    pass


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_root(cli_value, env_var, label) -> Path:
    value = cli_value or os.environ.get(env_var)
    if not value:
        raise DatasetError(
            f"{label} was not provided. Pass it on the command line or set the {env_var} "
            "environment variable."
        )
    return Path(value).expanduser().resolve()


def load_config(config_path) -> dict:
    path = Path(config_path)
    if not path.is_file():
        raise DatasetError(f"Config file not found: {path}")
    try:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        raise DatasetError(f"Could not parse config file {path}: {exc}") from exc

    if not isinstance(data, dict):
        raise DatasetError(f"Config file {path} must contain a mapping at the top level.")

    for key in ("dataset", "selection", "preprocessing"):
        if key not in data:
            raise DatasetError(f"Config file {path} is missing required section '{key}'.")

    return data


def list_dataset_images(image_dir: Path, accepted_extensions, excluded_files):
    if not image_dir.is_dir():
        raise DatasetError(f"Image directory not found: {image_dir}")

    accepted = {ext.lower() for ext in accepted_extensions}
    excluded = {name.lower() for name in excluded_files}

    files = [
        p
        for p in image_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in accepted and p.name.lower() not in excluded
    ]
    files.sort(key=lambda p: p.relative_to(image_dir).as_posix())
    return files


def open_frame(path: Path, frame: int = 0) -> Image.Image:
    im = Image.open(path)
    try:
        im.seek(frame)
    except EOFError as exc:
        im.close()
        raise DatasetError(f"Image {path} does not have frame index {frame}.") from exc
    return im


def _normcase(path: Path) -> str:
    return os.path.normcase(str(Path(path).resolve()))


def is_within_or_equal(path: Path, other: Path) -> bool:
    p = _normcase(path)
    o = _normcase(other)
    return p == o or p.startswith(o + os.sep)


def paths_overlap(a: Path, b: Path) -> bool:
    return is_within_or_equal(a, b) or is_within_or_equal(b, a)


def check_no_overlap(candidate: Path, forbidden: Path, candidate_label: str, forbidden_label: str):
    if paths_overlap(candidate, forbidden):
        raise DatasetError(
            f"{candidate_label} ({candidate}) overlaps with {forbidden_label} ({forbidden}). "
            "Choose a different location."
        )


def check_safe_output_root(output_root: Path, dataset_root: Path):
    check_no_overlap(output_root, repo_root(), "Output location", "the Git repository")
    check_no_overlap(output_root, dataset_root, "Output location", "the raw dataset directory")


def resolve_safe_subdir(root: Path, configured_value, label: str, allow_root: bool = True) -> Path:
    """Resolve `configured_value` as a child of `root`, refusing anything that escapes it.

    Catches absolute paths, ".." traversal, and symlink-based escapes (Path.resolve()
    follows symlinks for every path component that exists) by checking final containment
    rather than trusting the configured string.
    """
    root_resolved = Path(root).resolve()

    if configured_value is None or not str(configured_value).strip():
        raise DatasetError(f"{label} must be a non-empty relative path.")

    raw = str(configured_value)
    candidate = Path(raw)

    if candidate.is_absolute() or candidate.drive:
        raise DatasetError(f"{label} must be a relative path, not an absolute path: {raw!r}")

    candidate_resolved = (root_resolved / candidate).resolve()

    if not is_within_or_equal(candidate_resolved, root_resolved):
        raise DatasetError(
            f"{label} {raw!r} resolves to {candidate_resolved}, which is outside the "
            f"trusted root {root_resolved}. Refusing to use an unsafe path."
        )

    if not allow_root and _normcase(candidate_resolved) == _normcase(root_resolved):
        raise DatasetError(f"{label} {raw!r} must not resolve to the root directory itself ({root_resolved}).")

    return candidate_resolved


def make_bins(n: int, k: int):
    if k <= 0:
        raise DatasetError("Selection count must be a positive integer.")
    if k > n:
        raise DatasetError(f"Requested selection count ({k}) exceeds available images ({n}).")
    base, remainder = divmod(n, k)
    bins = []
    start = 0
    for i in range(k):
        size = base + (1 if i < remainder else 0)
        end = start + size
        bins.append((start, end))
        start = end
    return bins


def stratified_select(files, count, seed):
    bins = make_bins(len(files), count)
    rng = random.Random(seed)
    selected = []
    for bin_index, (start, end) in enumerate(bins):
        idx = rng.randrange(start, end)
        selected.append({"index": idx, "bin": bin_index, "file": files[idx]})
    return selected


def random_select(files, count, seed):
    if count <= 0:
        raise DatasetError("Selection count must be a positive integer.")
    if count > len(files):
        raise DatasetError(f"Requested selection count ({count}) exceeds available images ({len(files)}).")
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(files)), count))
    return [{"index": idx, "bin": None, "file": files[idx]} for idx in indices]


def select_images(files, method, count, seed):
    if method == "stratified":
        return stratified_select(files, count, seed)
    if method == "random":
        return random_select(files, count, seed)
    raise DatasetError(f"Unknown selection method: {method!r} (expected 'stratified' or 'random').")
