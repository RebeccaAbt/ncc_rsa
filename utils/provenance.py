import functools
import hashlib
import inspect
import json
import logging
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np


def artifact_log_path(output_path):
    """Return the same-stem text log path for an artifact."""
    output_path = Path(os.fspath(output_path))
    if output_path.suffix:
        return str(output_path.with_suffix(".log"))
    return f"{output_path}.log"


def configure_subject_logging(output_path, subject_id=None, upstream_log=None):
    """Configure a same-stem text log, optionally prefixed with a parent log."""
    output_directory = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(output_directory, exist_ok=True)
    log_path = artifact_log_path(output_path)
    logger = logging.getLogger(f"rsa_pipeline.{os.path.abspath(log_path)}")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)

    with open(log_path, "w", encoding="utf-8") as destination:
        if upstream_log and os.path.isfile(upstream_log):
            with open(upstream_log, encoding="utf-8") as source:
                upstream_text = source.read()
            destination.write("# Upstream provenance log\n")
            destination.write(upstream_text)
            if upstream_text and not upstream_text.endswith("\n"):
                destination.write("\n")
            destination.write("# Current operation\n")

    handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s %(message)s"
    ))
    logger.addHandler(handler)

    return logger, log_path


def flatten_integer_values(value):
    """Extract integer component IDs from nested JSON-compatible values."""
    if isinstance(value, bool):
        return []
    if isinstance(value, (int, np.integer)):
        return [int(value)]
    if isinstance(value, dict):
        values = []
        for item in value.values():
            values.extend(flatten_integer_values(item))
        return values
    if isinstance(value, (list, tuple)):
        values = []
        for item in value:
            values.extend(flatten_integer_values(item))
        return values
    return []

def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()

    with open(path, "rb") as file:
        while chunk := file.read(chunk_size):
            digest.update(chunk)

    return digest.hexdigest()


def _hash_array(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def serialize_value(value):
    """Convert parameters to JSON-safe values without storing large arrays."""
    if isinstance(value, np.ndarray):
        return {"type": "numpy.ndarray", "shape": list(value.shape),
                "dtype": str(value.dtype), "sha256": _hash_array(value)}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(key): serialize_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [serialize_value(item) for item in value]
    if hasattr(value, "__fspath__"):
        return os.fspath(value)
    return {"type": f"{type(value).__module__}.{type(value).__qualname__}",
            "repr": repr(value)}


def _git_info():
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], text=True,
            stderr=subprocess.DEVNULL
        ).strip())
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def _package_version(package_name):
    try:
        return version(package_name)
    except PackageNotFoundError:
        return None


def _input_record(path):
    path = os.path.abspath(os.fspath(path))
    record = {"path": path}
    if os.path.isfile(path):
        record["sha256"] = sha256_file(path)
        manifest_path = f"{path}.provenance.json"
        if os.path.isfile(manifest_path):
            with open(manifest_path, encoding="utf-8") as file:
                record["provenance"] = json.load(file)
    else:
        record["exists"] = os.path.exists(path)
    return record

def record_artifact(
    output_path,
    operation_name,
    parameters,
    input_paths,
):
    output_path = os.path.abspath(os.fspath(output_path))
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "artifact": {
            "path": output_path,
            "sha256": sha256_file(output_path),
        },
        "operation": {
            "name": operation_name,
            "status": "success",
        },
        "parameters": serialize_value(parameters),
        "inputs": [_input_record(path) for path in input_paths],
        "code": {"git": _git_info()},
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "packages": {
                name: _package_version(name)
                for name in ("numpy", "mne", "rsatoolbox", "joblib")
            },
        },
    }

    manifest_path = f"{output_path}.provenance.json"

    temporary_path = f"{manifest_path}.tmp"
    with open(temporary_path, "w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)
    os.replace(temporary_path, manifest_path)
    return manifest_path
        
def provenance_step(operation_name=None):
    def decorator(function):
        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            bound = inspect.signature(function).bind(*args, **kwargs)
            bound.apply_defaults()

            parameters = {
                name: serialize_value(value)
                for name, value in bound.arguments.items()
            }

            return function(*args, **kwargs)

        return wrapper

    return decorator

def save_with_provenance(
    object_to_save,
    output_path,
    operation_name,
    parameters,
    input_paths,
):
    import joblib

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    joblib.dump(object_to_save, output_path)

    record_artifact(
        output_path=output_path,
        operation_name=operation_name,
        parameters=parameters,
        input_paths=input_paths,
    )



	