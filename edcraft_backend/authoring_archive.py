"""Archive the quiesced, file-backed authoring store before an empty database cutover."""

import argparse
import hashlib
import io
import os
import tarfile
import tempfile
from pathlib import Path


def archive_store(source: Path, destination: Path) -> None:
    """Publish a private checksum-verified archive, without replacing an existing backup."""
    source = source.resolve(strict=True)
    destination = destination.absolute()
    if destination.resolve().is_relative_to(source):
        raise ValueError("Choose a backup destination outside the authoring store.")
    if destination.exists():
        raise FileExistsError(destination)
    if not all((source / name).is_dir() for name in ("jobs", "artifacts")):
        raise ValueError("Expected the jobs and artifacts directories of an authoring store.")
    with tempfile.NamedTemporaryFile(dir=destination.parent) as temporary:
        hashes = {}
        with tarfile.open(fileobj=temporary, mode="w:gz") as archive:
            for path in sorted(source.rglob("*")):
                name = path.relative_to(source).as_posix()
                if name == "SHA256SUMS" or any(char in name for char in "\r\n\\"):
                    raise ValueError(f"Unexpected archive member name: {name!r}")
                if path.is_symlink() or not (path.is_dir() or path.is_file()):
                    raise ValueError(f"Expected a directory or regular file: {name!r}")
                if path.is_file():
                    with path.open("rb") as file:
                        hashes[name] = hashlib.file_digest(file, "sha256").hexdigest()
                archive.add(path, arcname=name, recursive=False)
            manifest = "".join(f"{digest}  {name}\n" for name, digest in hashes.items()).encode()
            info = tarfile.TarInfo("SHA256SUMS")
            info.size = len(manifest)
            info.mode = 0o600
            archive.addfile(info, io.BytesIO(manifest))
        temporary.flush()
        os.fsync(temporary.fileno())
        with tarfile.open(temporary.name) as archive:
            for name, digest in hashes.items():
                file = archive.extractfile(name)
                if file is None or hashlib.file_digest(file, "sha256").hexdigest() != digest:
                    raise ValueError(f"Archived bytes differ from the source: {name!r}")
        os.link(temporary.name, destination)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    archive_store(args.source, args.destination)
