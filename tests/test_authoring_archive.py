import hashlib
import tarfile

import pytest

from edcraft_backend.authoring_archive import archive_store


def test_archive_round_trip_and_no_overwrite(tmp_path):
    source = tmp_path / "authoring"
    (source / "jobs").mkdir(parents=True)
    (source / "artifacts" / "one").mkdir(parents=True)
    (source / "jobs/one.json").write_bytes(b'{"state": "completed"}\n')
    original = b'{ "checked":true }\n'
    (source / "artifacts/one/artifact.json").write_bytes(original)
    destination = tmp_path / "offline.tar.gz"
    archive_store(source, destination)
    assert destination.stat().st_mode & 0o777 == 0o600
    with tarfile.open(destination) as archive:
        archive.extractall(tmp_path / "restored", filter="data")
    restored = tmp_path / "restored"
    assert (restored / "artifacts/one/artifact.json").read_bytes() == original
    digest = hashlib.sha256(original).hexdigest()
    assert f"{digest}  artifacts/one/artifact.json" in (restored / "SHA256SUMS").read_text()
    with pytest.raises(FileExistsError):
        archive_store(source, destination)


def test_archive_refuses_links_and_destination_inside_store(tmp_path):
    (tmp_path / "source/jobs").mkdir(parents=True)
    (tmp_path / "source/artifacts").mkdir()
    with pytest.raises(ValueError, match="outside"):
        archive_store(tmp_path / "source", tmp_path / "source/bad.tar.gz")
    (tmp_path / "source/jobs/link").symlink_to(tmp_path / "source/artifacts")
    with pytest.raises(ValueError, match="regular|link"):
        archive_store(tmp_path / "source", tmp_path / "backup.tar.gz")
    assert not (tmp_path / "backup.tar.gz").exists()
