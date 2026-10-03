"""Pins upstream sources and prepares role-separated MLE datasets."""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import io
import json
import os
import re
import shutil
import stat
import subprocess
import tempfile
import urllib.request
import zipfile
from collections.abc import Iterable, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

import dotenv
import yaml

from hma.benchmark.catalog import (
    CATALOG_SELECTION_NAMES,
    MLEBENCH_COMMIT,
    MLEDOJO_COMMIT,
    MLETaskSpec,
    select_catalog,
)
from hma.benchmark.integrity import (
    atomic_json,
    sha256_file,
    tree_manifest,
    verify_tree,
)
from hma.benchmark.runtime import BASE_RUNTIME_IMAGE, FULL_EVALUATOR_IMAGE, credential_home

_MLEBENCH_REPOSITORY = "https://github.com/openai/mle-bench.git"
_MLEDOJO_REPOSITORY = "https://github.com/MLE-Dojo/MLE-Dojo.git"
_DSBENCH_REPOSITORY = "liqiang888/DSBench"
_DSBENCH_REVISION = "1196d6553ec200e1262841efc89cda5954ffe89f"
_DSBENCH_FILE = "data_modeling/data.zip"
_DSBENCH_SIZE = 3_126_493_713
_DSBENCH_SHA256 = "bed0c7b202389597aadbbc1f8769f459119c1a662c3e11905c4ae05825876087"
_MAX_EXTRACTED_BYTES = 750 * 1024**3
_MAX_PREPARE_WORKER_BYTES = 1024 * 1024
_PREPARE_WORKER_ENV = "FLOWBENCH_MLE_PREPARE_WORKER_PATH"
_URL = re.compile(r"https?://\S+", re.IGNORECASE)
_ALIASES: dict[str, tuple[str, ...]] = {
    "mdu_01": ("Conway's Reverse Game of Life",),
    "mdu_05": ("University of Liverpool", "Liverpool"),
    "mdu_06": ("Porto Seguro",),
    "mdu_07": ("Santander",),
    "mdu_08": ("Santander",),
    "mdu_09": ("Santander",),
    "mdu_11": ("Kaggle LLM Science Exam",),
    "mdu_14": ("Quora",),
    "mdu_15": ("Quora",),
    "mdu_16": ("StumbleUpon",),
    "mdu_17": ("Airbus",),
    "mdu_18": ("Bengali.AI", "Bengali AI"),
    "mdu_19": ("Draper",),
}


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _data_root(value: Path | None) -> Path:
    dotenv.load_dotenv()
    raw = value or (
        Path(os.environ["FLOWBENCH_MLE_DATA_ROOT"])
        if "FLOWBENCH_MLE_DATA_ROOT" in os.environ
        else None
    )
    if raw is None:
        raise ValueError("FLOWBENCH_MLE_DATA_ROOT is not set")
    root = raw.expanduser().resolve()
    if not root.exists():
        root.mkdir(parents=True)
        root.chmod(0o700)
    # An existing root keeps its own mode: a prepared root may be published
    # read-only for reuse, and re-locking it here would revoke that on the
    # next command.
    return root


def _run(command: list[str], *, env: dict[str, str] | None = None) -> None:
    subprocess.run(command, check=True, env=env)


def _clone_pinned(target: Path, repository: str, commit: str) -> None:
    if target.exists():
        if not (target / ".git").is_dir():
            raise ValueError(f"source target exists but is not a checkout: {target}")
        head = subprocess.check_output(
            ["git", "-C", str(target), "rev-parse", "HEAD"], text=True
        ).strip()
        if head != commit:
            raise ValueError(f"source checkout has unexpected revision: {target}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    try:
        _run(["git", "clone", "--filter=blob:none", repository, str(temporary)])
        _run(["git", "-C", str(temporary), "checkout", "--detach", commit])
        os.replace(temporary, target)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _download_bytes(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    try:
        with (
            os.fdopen(descriptor, "wb") as file,
            urllib.request.urlopen(url, timeout=120) as response,
        ):
            shutil.copyfileobj(response, file, length=1024 * 1024)
        os.replace(name, destination)
    except BaseException:
        Path(name).unlink(missing_ok=True)
        raise


def _leaderboard(spec: MLETaskSpec, root: Path) -> Path:
    destination = root / "packages" / spec.task_name / "control" / "leaderboard.csv"
    if not destination.is_file():
        url = (
            "https://media.githubusercontent.com/media/openai/mle-bench/"
            f"{MLEBENCH_COMMIT}/mlebench/competitions/{spec.slug}/leaderboard.csv"
        )
        _download_bytes(url, destination)
    with destination.open(newline="", encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))
    if not rows or "score" not in rows[0]:
        raise ValueError(f"invalid MLE-bench leaderboard for {spec.experiment_id}")
    scores = [float(row["score"]) for row in rows]
    if scores != sorted(scores, reverse=spec.higher_is_better):
        raise ValueError(f"leaderboard direction mismatch for {spec.experiment_id}")
    return destination


def _protected_terms(spec: MLETaskSpec) -> list[str]:
    terms = [spec.title, spec.slug, spec.title.replace(" - ", " ")]
    terms.extend(_ALIASES.get(spec.task_name, ()))
    return list(dict.fromkeys(term for term in terms if len(term) >= 2))


def _sanitize_dojo(spec: MLETaskSpec, source: str) -> str:
    result = _URL.sub("[external link removed]", source)
    for term in sorted(_protected_terms(spec), key=len, reverse=True):
        result = re.sub(re.escape(term), "[task]", result, flags=re.IGNORECASE)
    return (
        f"De-identified task {spec.experiment_id}\n\n"
        f"Native metric: {spec.metric}. "
        f"{'Higher' if spec.higher_is_better else 'Lower'} is better.\n\n{result.strip()}\n"
    )


def _legacy_sanitize_mlebench(spec: MLETaskSpec, source: str) -> str:
    """Reproduce the retired local transform only to migrate frozen controls."""
    result = _URL.sub("[external link removed]", source)
    for term in sorted(_protected_terms(spec), key=len, reverse=True):
        result = re.sub(re.escape(term), "[task]", result, flags=re.IGNORECASE)
    return result.strip() + "\n"


def _mlebench_description(_spec: MLETaskSpec, source: str) -> str:
    """Return the pinned MLE-bench task description byte-for-byte as text."""
    return source


def _stage_control(spec: MLETaskSpec, root: Path) -> None:
    source = root / "sources"
    control = root / "packages" / spec.task_name / "control"
    control.mkdir(parents=True, exist_ok=True)
    description_path = root / "descriptions" / f"{spec.task_name}.md"
    description_path.parent.mkdir(parents=True, exist_ok=True)
    if spec.is_mlebench:
        official_description = (
            source / "mle-bench" / "mlebench" / "competitions" / spec.slug / "description.md"
        ).read_text(encoding="utf-8")
        obfuscated_description = (
            source
            / "mle-bench"
            / "mlebench"
            / "competitions"
            / spec.slug
            / "description_obfuscated.md"
        ).read_text(encoding="utf-8")
        description = _mlebench_description(spec, official_description)
        leaderboard = _leaderboard(spec, root)
        leaderboard_sha256: str | None = sha256_file(leaderboard)
    else:
        upstream_description = (
            source
            / "mle-dojo"
            / "mledojo"
            / "competitions"
            / spec.slug
            / "info"
            / "description.txt"
        ).read_text(encoding="utf-8")
        description = _sanitize_dojo(spec, upstream_description)
        leaderboard_sha256 = None
    terms = _protected_terms(spec)
    terms_path = control / "network-protected-terms.json"
    contract_path = control / "contract.json"
    expected = {
        "schema_version": 2 if spec.is_mlebench else 1,
        "experiment_id": spec.experiment_id,
        "suite": spec.suite,
        "slug": spec.slug,
        "metric": spec.metric,
        "higher_is_better": spec.higher_is_better,
        "feedback_mode": spec.feedback_mode,
        "submission_limit": spec.submission_limit,
        "upstream_commit": spec.upstream_commit,
        "description_sha256": hashlib.sha256(description.encode()).hexdigest(),
        "leaderboard_sha256": leaderboard_sha256,
    }
    if spec.is_mlebench:
        expected |= {
            "description_source": "description.md",
            "description_transform": "none",
        }
    if contract_path.is_file():
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        recorded_terms = json.loads(terms_path.read_text(encoding="utf-8"))
        has_mismatch = (
            not isinstance(contract, dict)
            or any(contract.get(key) != value for key, value in expected.items())
            or description_path.read_text(encoding="utf-8") != description
            or recorded_terms != terms
        )
        if has_mismatch:
            # Controls prepared by the earlier adapter are upgraded only when
            # every frozen byte still matches that known transform. Arbitrary
            # drift remains a hard error.
            variants: list[tuple[str, dict[str, Any]]] = []
            if spec.is_mlebench:
                transformed = _legacy_sanitize_mlebench(spec, obfuscated_description)
                schema_one = expected | {
                    "schema_version": 1,
                    "description_sha256": hashlib.sha256(transformed.encode()).hexdigest(),
                }
                schema_one.pop("description_source", None)
                schema_one.pop("description_transform", None)
                variants.append((transformed, schema_one))
                variants.append(
                    (
                        obfuscated_description,
                        expected
                        | {
                            "description_sha256": hashlib.sha256(
                                obfuscated_description.encode()
                            ).hexdigest(),
                            "description_source": "description_obfuscated.md",
                        },
                    )
                )
            current_description = description_path.read_text(encoding="utf-8")
            known_variant = any(
                current_description == candidate
                and isinstance(contract, dict)
                and all(contract.get(key) == value for key, value in candidate_contract.items())
                for candidate, candidate_contract in variants
            )
            if not known_variant or recorded_terms != terms:
                raise ValueError(f"frozen control mismatch for {spec.experiment_id}")
            description_path.write_text(description, encoding="utf-8")
            atomic_json(
                contract_path,
                {
                    **contract,
                    **expected,
                    "description_migrated_at_utc": _utc_now(),
                },
            )
    else:
        if description_path.exists() or terms_path.exists():
            raise ValueError(f"partial control exists for {spec.experiment_id}")
        description_path.write_text(description, encoding="utf-8")
        atomic_json(terms_path, terms)
        atomic_json(contract_path, {**expected, "frozen_at_utc": _utc_now()})
    control.chmod(0o700)
    for path in control.iterdir():
        path.chmod(0o600)


def _kaggle_environment() -> dict[str, str]:
    environment = dict(os.environ)
    if environment.get("KAGGLE_API_TOKEN"):
        return environment
    token_path = credential_home() / ".kaggle" / "access_token"
    if token_path.is_symlink() or not token_path.is_file():
        raise ValueError("Kaggle access token is unavailable")
    mode = stat.S_IMODE(token_path.stat().st_mode)
    if mode != 0o600 or not 1 <= token_path.stat().st_size <= 4096:
        raise ValueError("Kaggle access token must be a non-empty mode-600 file")
    environment["KAGGLE_API_TOKEN"] = token_path.read_text(encoding="utf-8").strip()
    return environment


@contextmanager
def _kaggle_download_lock(spec: MLETaskSpec, root: Path) -> Iterable[None]:
    lock_directory = root / "locks" / "kaggle"
    lock_directory.mkdir(parents=True, exist_ok=True)
    lock_path = lock_directory / f"{spec.task_name}.lock"
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


@contextmanager
def _kaggle_task_lock(spec: MLETaskSpec, root: Path) -> Iterable[None]:
    """Serialize the complete prepare/prune lifecycle for one Kaggle task."""
    lock_directory = root / "locks" / "prepare"
    lock_directory.mkdir(parents=True, exist_ok=True)
    lock_path = lock_directory / f"{spec.task_name}.lock"
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _kaggle_zip_locked(spec: MLETaskSpec, root: Path) -> Path:
    directory = root / "downloads" / "kaggle" / spec.slug
    directory.mkdir(parents=True, exist_ok=True)
    archives = list(directory.glob("*.zip"))
    partials = list(directory.glob("*.kaggle-partial"))
    if not archives or partials:
        _run(
            [
                "uv",
                "run",
                "--with",
                "kaggle==2.2.4",
                "kaggle",
                "competitions",
                "download",
                "-c",
                spec.slug,
                "-p",
                str(directory),
            ],
            env=_kaggle_environment(),
        )
        archives = list(directory.glob("*.zip"))
        partials = list(directory.glob("*.kaggle-partial"))
    if partials:
        raise ValueError(f"Kaggle download remains partial for {spec.experiment_id}")
    if len(archives) != 1 or archives[0].is_symlink():
        raise ValueError(f"expected one Kaggle archive for {spec.experiment_id}")
    return archives[0]


def _kaggle_zip(spec: MLETaskSpec, root: Path) -> Path:
    with _kaggle_download_lock(spec, root):
        return _kaggle_zip_locked(spec, root)


def _safe_extract(archive: Path, destination: Path) -> str:
    archive_sha256 = sha256_file(archive)
    identity = {
        "schema_version": 1,
        "archive_name": archive.name,
        "archive_size_bytes": archive.stat().st_size,
        "archive_sha256": archive_sha256,
    }
    marker = destination.parent / "raw-extraction.json"
    if marker.is_file():
        recorded = json.loads(marker.read_text(encoding="utf-8"))
        if recorded != identity or not destination.is_dir() or not any(destination.iterdir()):
            raise ValueError(f"raw extraction marker mismatch for {archive.name}")
        return archive_sha256
    if destination.exists() and (
        destination.is_symlink() or not destination.is_dir() or any(destination.iterdir())
    ):
        raise ValueError(f"refusing to reuse an unverified extraction: {destination}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        with zipfile.ZipFile(archive) as bundle:
            total = 0
            names: set[str] = set()
            for member in bundle.infolist():
                path = PurePosixPath(member.filename)
                file_type = member.external_attr >> 16
                total += member.file_size
                if (
                    not path.parts
                    or path.is_absolute()
                    or ".." in path.parts
                    or stat.S_ISLNK(file_type)
                    or total > _MAX_EXTRACTED_BYTES
                    or path.as_posix() in names
                ):
                    raise ValueError(f"unsafe archive member in {archive.name}")
                names.add(path.as_posix())
            bundle.extractall(temporary)
        if destination.exists():
            destination.rmdir()
        os.replace(temporary, destination)
        atomic_json(marker, identity)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return archive_sha256


def _empty_output_directories(spec: MLETaskSpec, dataset: Path) -> None:
    if spec.is_mlebench:
        public = dataset / "prepared" / "public"
        private = dataset / "prepared" / "private"
    else:
        public = dataset / "data" / "public"
        private = dataset / "data" / "private"
    for directory in (public, private):
        directory.mkdir(parents=True, exist_ok=True)
        if any(directory.iterdir()):
            raise ValueError(f"refusing to replace non-empty prepared output: {directory}")


def _prepare_worker_path() -> Path:
    raw = os.environ.get(_PREPARE_WORKER_ENV)
    if raw:
        candidate = Path(raw)
        if not candidate.is_absolute():
            raise ValueError(f"{_PREPARE_WORKER_ENV} must be an absolute path")
        if candidate.is_symlink():
            raise ValueError(f"{_PREPARE_WORKER_ENV} must not be a symlink")
    else:
        candidate = Path(__file__).resolve().parent / "prepare_worker.py"
    try:
        worker = candidate.resolve(strict=True)
    except FileNotFoundError as error:
        raise ValueError(f"prepare worker is unavailable: {candidate}") from error
    if not worker.is_file() or not 1 <= worker.stat().st_size <= _MAX_PREPARE_WORKER_BYTES:
        raise ValueError(f"invalid prepare worker: {worker}")
    return worker


def _run_prepare_worker(spec: MLETaskSpec, root: Path) -> None:
    source = root / "sources" / spec.source_name
    dataset = root / spec.data_subpath
    worker = _prepare_worker_path()
    _empty_output_directories(spec, dataset)
    if spec.slug in {
        "cassava-leaf-disease-classification",
        "siim-isic-melanoma-classification",
    }:
        _run(
            [
                "uv",
                "run",
                "--no-project",
                "--python",
                "3.11",
                "--with",
                "tensorflow==2.17.0",
                "--with",
                "pandas>=2.2",
                "--with",
                "scikit-learn>=1.5",
                "--with",
                "tqdm>=4.66",
                "--with",
                "py7zr>=0.21",
                "--with",
                "pyyaml>=6",
                "python",
                str(worker),
                "--suite",
                spec.suite,
                "--upstream",
                str(source),
                "--dataset",
                str(dataset),
                "--slug",
                spec.slug,
            ]
        )
        return
    if spec.slug == "icecube-neutrinos-in-deep-ice":
        # This is the only upstream preparer that writes parquet, and the pinned
        # evaluator image intentionally carries no parquet engine. Both writers
        # stamp their own version into the parquet footer, so the upstream
        # checksums only reproduce under this exact fastparquet/pandas pair.
        _run(
            [
                "uv",
                "run",
                "--no-project",
                "--python",
                "3.11",
                "--with",
                "fastparquet==2024.5.0",
                "--with",
                "numpy==1.26.4",
                "--with",
                "pandas==2.2.2",
                "--with",
                "scikit-learn>=1.5",
                "--with",
                "tqdm>=4.66",
                "--with",
                "py7zr>=0.21",
                "--with",
                "pyyaml>=6",
                "python",
                str(worker),
                "--suite",
                spec.suite,
                "--upstream",
                str(source),
                "--dataset",
                str(dataset),
                "--slug",
                spec.slug,
            ]
        )
        return
    if spec.slug in {
        "h-and-m-personalized-fashion-recommendations",
        "nfl-player-contact-detection",
        "tgs-salt-identification-challenge",
        "uw-madison-gi-tract-image-segmentation",
        "vinbigdata-chest-xray-abnormalities-detection",
    }:
        # The upstream checksums were produced with pandas 2.2.x, but the pinned
        # evaluator image ships pandas 3.0. That release changes CSV
        # serialization, drops DataFrame.applymap, and backs string columns with
        # Arrow arrays that scikit-learn cannot index, so run these preparers in
        # the compatible data-frame environment instead.
        _run(
            [
                "uv",
                "run",
                "--no-project",
                "--python",
                "3.11",
                "--with",
                "numpy==2.1.2",
                "--with",
                "pandas==2.2.3",
                "--with",
                "scikit-learn>=1.5",
                "--with",
                "tqdm>=4.66",
                "--with",
                "py7zr>=0.21",
                "--with",
                "pyyaml>=6",
                "python",
                str(worker),
                "--suite",
                spec.suite,
                "--upstream",
                str(source),
                "--dataset",
                str(dataset),
                "--slug",
                spec.slug,
            ]
        )
        return
    network = [] if spec.slug == "freesound-audio-tagging-2019" else ["--network", "none"]
    image = (
        FULL_EVALUATOR_IMAGE
        if spec.suite in {"mlebench_medium", "mlebench_high"}
        else BASE_RUNTIME_IMAGE
    )
    _run(
        [
            "docker",
            "run",
            "--rm",
            *network,
            "--user",
            f"{os.getuid()}:{os.getgid()}",
            "-v",
            f"{source}:/workspace/upstream:ro",
            "-v",
            f"{dataset}:/workspace/dataset:rw",
            "-v",
            f"{worker}:/workspace/prepare-worker.py:ro",
            "--entrypoint",
            "python3",
            image,
            "/workspace/prepare-worker.py",
            "--suite",
            spec.suite,
            "--upstream",
            "/workspace/upstream",
            "--dataset",
            "/workspace/dataset",
            "--slug",
            spec.slug,
        ]
    )


def _md5(path: Path) -> str:
    # MLE-bench publishes MD5 values as its exact compatibility contract.
    digest = hashlib.md5()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_mlebench(spec: MLETaskSpec, root: Path, archive: Path) -> None:
    source = root / "sources" / "mle-bench" / "mlebench" / "competitions"
    expected = yaml.safe_load((source / spec.slug / "checksums.yaml").read_text())
    if _md5(archive) != expected["zip"]:
        raise ValueError(f"official archive checksum mismatch for {spec.experiment_id}")
    dataset = root / spec.data_subpath / "prepared"
    for role in ("public", "private"):
        for relative, checksum in expected[role].items():
            if _md5(dataset / role / relative) != checksum:
                raise ValueError(f"prepared checksum mismatch for {spec.experiment_id}")


def _copy_dojo_metadata(spec: MLETaskSpec, root: Path) -> None:
    info = root / "sources" / "mle-dojo" / "mledojo" / "competitions" / spec.slug / "info"
    dataset = root / spec.data_subpath / "data"
    shutil.copy2(info / "description.txt", dataset / "public" / "description.txt")
    for name in ("public_leaderboard.csv", "private_leaderboard.csv"):
        shutil.copy2(info / name, dataset / "private" / name)


def _member_map(bundle: zipfile.ZipFile, slug: str) -> dict[str, zipfile.ZipInfo]:
    result: dict[str, zipfile.ZipInfo] = {}
    for member in bundle.infolist():
        path = PurePosixPath(member.filename)
        file_type = member.external_attr >> 16
        if path.is_absolute() or ".." in path.parts or stat.S_ISLNK(file_type):
            raise ValueError("unsafe member in frozen DSBench archive")
        parts = path.parts
        if "__MACOSX" in parts or any(part.startswith("._") for part in parts):
            continue
        for marker in ("data_resplit", "answers", "task"):
            if marker not in parts:
                continue
            index = parts.index(marker)
            suffix = parts[index:]
            if marker == "task" and suffix == ("task", f"{slug}.txt"):
                result["description.txt"] = member
            elif len(suffix) >= 3 and suffix[1] == slug:
                result[f"{marker}/{'/'.join(suffix[2:])}"] = member
    return result


def _copy_zip_member(bundle: zipfile.ZipFile, member: zipfile.ZipInfo, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with bundle.open(member) as source, destination.open("xb") as output:
        shutil.copyfileobj(source, output, length=1024 * 1024)


def _rewrite_dsbench_sample(
    bundle: zipfile.ZipFile,
    template_member: zipfile.ZipInfo,
    test: Path,
    sample: Path,
) -> None:
    temporary = sample.with_suffix(".deterministic.csv")
    with (
        io.TextIOWrapper(bundle.open(template_member), encoding="utf-8-sig", newline="") as source,
        test.open(newline="", encoding="utf-8-sig") as test_file,
        temporary.open("x", newline="", encoding="utf-8") as output,
    ):
        template_reader = csv.reader(source)
        test_reader = csv.reader(test_file)
        writer = csv.writer(output)
        header = next(template_reader)
        template = next(template_reader)
        test_header = next(test_reader)
        if len(header) < 2 or len(template) != len(header) or header[0] != test_header[0]:
            raise ValueError("DSBench sample template does not match public test IDs")
        writer.writerow(header)
        for row in test_reader:
            if not row:
                raise ValueError("DSBench public test contains an empty row")
            writer.writerow([row[0], *template[1:]])
    os.replace(temporary, sample)


def _read_manifest(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"task manifest must be an object: {path}")
    return payload


def _verify_frozen_task(spec: MLETaskSpec, root: Path, manifest: dict[str, Any]) -> None:
    dataset = root / spec.data_subpath
    scoring_data = dataset / spec.data_suffix
    if (
        manifest.get("experiment_id") != spec.experiment_id
        or manifest.get("suite") != spec.suite
        or manifest.get("upstream_commit") != spec.upstream_commit
        or not isinstance(manifest.get("data"), dict)
        or not verify_tree(scoring_data, manifest["data"])
    ):
        raise ValueError(f"frozen task integrity mismatch for {spec.experiment_id}")


def _migrate_legacy_dataset(spec: MLETaskSpec, root: Path) -> None:
    legacy = root / spec.legacy_data_subpath
    target = root / spec.data_subpath
    if not legacy.exists():
        return
    if legacy.is_symlink() or not legacy.is_dir():
        raise ValueError(f"legacy dataset is not a regular directory: {legacy}")
    if target.exists():
        if target.is_symlink() or not target.is_dir() or any(target.iterdir()):
            raise ValueError(f"neutral dataset target already exists: {target}")
        target.rmdir()
    target.parent.mkdir(parents=True, exist_ok=True)
    os.replace(legacy, target)


def _prepare_dsbench(specs: list[MLETaskSpec], root: Path) -> None:
    download_root = root / "downloads" / "dsbench"
    archive = download_root / _DSBENCH_FILE
    if not archive.is_file():
        _run(
            [
                "hf",
                "download",
                _DSBENCH_REPOSITORY,
                _DSBENCH_FILE,
                "--type",
                "dataset",
                "--revision",
                _DSBENCH_REVISION,
                "--local-dir",
                str(download_root),
            ]
        )
    if archive.stat().st_size != _DSBENCH_SIZE or sha256_file(archive) != _DSBENCH_SHA256:
        raise ValueError("DSBench archive does not match the frozen release")
    with zipfile.ZipFile(archive) as bundle:
        for spec in specs:
            dataset = root / spec.data_subpath / "data"
            public, private = dataset / "public", dataset / "private"
            manifest_path = root / "manifests" / f"{spec.task_name}.json"
            manifest = _read_manifest(manifest_path)
            legacy_manifest = bool(
                manifest and manifest.get("provenance", {}).get("sample_seed") == 0
            )
            if manifest is not None:
                _verify_frozen_task(spec, root, manifest)
                if not legacy_manifest:
                    continue
            elif (public.exists() and any(public.iterdir())) or (
                private.exists() and any(private.iterdir())
            ):
                raise ValueError(
                    f"refusing to reuse unverified DSBench output for {spec.experiment_id}"
                )
            public.mkdir(parents=True, exist_ok=True)
            private.mkdir(parents=True, exist_ok=True)
            members = _member_map(bundle, spec.slug)
            public_names = {
                key.removeprefix("data_resplit/"): value
                for key, value in members.items()
                if key.startswith("data_resplit/")
            }
            required = {"train.csv", "test.csv"}
            if not required.issubset(public_names) or "answers/test_answer.csv" not in members:
                raise ValueError(f"DSBench package is incomplete for {spec.experiment_id}")
            sample_member = public_names.get(
                "sample_submission.csv", public_names.get("sample_solution.csv")
            )
            if sample_member is None:
                raise ValueError(f"DSBench sample submission is missing for {spec.experiment_id}")
            if not legacy_manifest:
                for relative, member in public_names.items():
                    name = (
                        "sample_submission.csv" if relative == "sample_solution.csv" else relative
                    )
                    _copy_zip_member(bundle, member, public / name)
                _copy_zip_member(
                    bundle,
                    members["answers/test_answer.csv"],
                    private / "test_answer.csv",
                )
                if "description.txt" in members:
                    _copy_zip_member(bundle, members["description.txt"], public / "description.txt")
            for path in public.rglob("._*"):
                if path.is_file() and not path.is_symlink():
                    path.unlink()
            sample = public / "sample_submission.csv"
            _rewrite_dsbench_sample(bundle, sample_member, public / "test.csv", sample)
            _copy_dojo_metadata(spec, root)
            _freeze_task(
                spec,
                root,
                {
                    "source": "dojo_dsbench",
                    "source_revision": _DSBENCH_REVISION,
                    "source_archive_sha256": _DSBENCH_SHA256,
                    "sample_baseline": "upstream_template_remapped_to_public_test_ids",
                },
            )


def _freeze_task(spec: MLETaskSpec, root: Path, provenance: dict[str, Any]) -> None:
    dataset = root / spec.data_subpath
    scoring_data = dataset / spec.data_suffix
    manifest = {
        "schema_version": 1,
        "experiment_id": spec.experiment_id,
        "suite": spec.suite,
        "upstream_commit": spec.upstream_commit,
        "data": tree_manifest(scoring_data),
        "provenance": provenance,
        "frozen_at_utc": _utc_now(),
    }
    atomic_json(root / "manifests" / f"{spec.task_name}.json", manifest)


def _prune_kaggle_working_data(spec: MLETaskSpec, root: Path) -> None:
    dataset = root / spec.data_subpath
    raw = dataset / "raw"
    marker = dataset / "raw-extraction.json"
    download = root / "downloads" / "kaggle" / spec.slug
    if raw.exists():
        if raw.is_symlink() or not raw.is_dir():
            raise ValueError(f"refusing to prune unsafe raw path: {raw}")
        shutil.rmtree(raw)
    if marker.exists():
        if marker.is_symlink() or not marker.is_file():
            raise ValueError(f"refusing to prune unsafe extraction marker: {marker}")
        marker.unlink()
    if download.exists():
        if download.is_symlink() or not download.is_dir():
            raise ValueError(f"refusing to prune unsafe download path: {download}")
        for archive in download.glob("*.zip"):
            if archive.is_symlink() or not archive.is_file():
                raise ValueError(f"refusing to prune unsafe archive: {archive}")
            archive.unlink()
        if not any(download.iterdir()):
            download.rmdir()


def _prepare_kaggle_task(
    spec: MLETaskSpec, root: Path, *, prune_working_data: bool = False
) -> None:
    with _kaggle_task_lock(spec, root):
        _prepare_kaggle_task_locked(
            spec,
            root,
            prune_working_data=prune_working_data,
        )


def _prepare_kaggle_task_locked(
    spec: MLETaskSpec, root: Path, *, prune_working_data: bool = False
) -> None:
    manifest_path = root / "manifests" / f"{spec.task_name}.json"
    frozen = _read_manifest(manifest_path)
    if frozen is not None:
        _verify_frozen_task(spec, root, frozen)
        if prune_working_data:
            _prune_kaggle_working_data(spec, root)
        return
    archive = _kaggle_zip(spec, root)
    dataset = root / spec.data_subpath
    raw = dataset / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    archive_sha256 = _safe_extract(archive, raw)
    _run_prepare_worker(spec, root)
    if spec.is_mlebench:
        description = (
            root
            / "sources"
            / "mle-bench"
            / "mlebench"
            / "competitions"
            / spec.slug
            / "description.md"
        )
        shutil.copy2(description, dataset / "prepared" / "public" / "description.md")
        _verify_mlebench(spec, root, archive)
    else:
        _copy_dojo_metadata(spec, root)
    _freeze_task(
        spec,
        root,
        {"source": spec.data_source, "archive_sha256": archive_sha256},
    )
    if prune_working_data:
        _prune_kaggle_working_data(spec, root)


def _selected(suite: str, task_names: Iterable[str]) -> list[MLETaskSpec]:
    pool = list(select_catalog(suite))
    requested = set(task_names)
    selected = [spec for spec in pool if not requested or spec.task_name in requested]
    if requested and requested != {spec.task_name for spec in selected}:
        raise ValueError("--task contains an unknown or out-of-suite task name")
    return selected


def prepare_sources(root: Path, specs: list[MLETaskSpec]) -> None:
    """Pins required repositories and stages evaluator-only control files."""
    if any(spec.is_mlebench for spec in specs):
        _clone_pinned(root / "sources" / "mle-bench", _MLEBENCH_REPOSITORY, MLEBENCH_COMMIT)
    if any(not spec.is_mlebench for spec in specs):
        _clone_pinned(root / "sources" / "mle-dojo", _MLEDOJO_REPOSITORY, MLEDOJO_COMMIT)
    for spec in specs:
        _stage_control(spec, root)


def prepare_tasks(
    root: Path, specs: list[MLETaskSpec], *, prune_working_data: bool = False
) -> None:
    """Downloads, prepares, verifies, and freezes the selected datasets."""
    for spec in specs:
        _migrate_legacy_dataset(spec, root)
    dsbench = [spec for spec in specs if spec.data_source == "dojo_dsbench"]
    if dsbench:
        _prepare_dsbench(dsbench, root)
    for spec in specs:
        if spec.data_source != "dojo_dsbench":
            _prepare_kaggle_task(
                spec,
                root,
                prune_working_data=prune_working_data,
            )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare native MLE benchmark data")
    parser.add_argument("--root", type=Path)
    parser.add_argument(
        "--suite",
        choices=CATALOG_SELECTION_NAMES,
        default="both",
    )
    parser.add_argument("--task", action="append", default=[])
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument(
        "--prune-working-data",
        action="store_true",
        help="remove verified Kaggle archives and raw extraction trees",
    )
    args = parser.parse_args(argv)
    root = _data_root(args.root)
    specs = _selected(args.suite, args.task)
    prepare_sources(root, specs)
    if args.prepare:
        prepare_tasks(root, specs, prune_working_data=args.prune_working_data)
    print(
        json.dumps(
            {
                "root": str(root),
                "selected_tasks": len(specs),
                "sources_pinned": True,
                "data_prepared": args.prepare,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
