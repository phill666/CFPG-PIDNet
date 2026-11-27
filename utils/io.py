"""Atomic experiment metadata writes and content hashes."""
import hashlib
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def atomic_text(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(value, encoding='utf-8-sig' if path.suffix == '.csv' else 'utf-8')
    os.replace(temporary, path)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
