"""Run pinned MiniLM sentence embeddings locally on the CPU."""

from __future__ import annotations

import os
import platform
from collections import OrderedDict
from functools import lru_cache
from hashlib import file_digest, sha256
from math import ceil
from pathlib import Path
from tempfile import NamedTemporaryFile
from threading import Lock
from typing import cast

import httpx
import numpy as np
import numpy.typing as npt
import onnxruntime as ort  # type: ignore[import-untyped]
from tokenizers import Tokenizer

_REVISION = "826711e54e001c83835913827a843d8dd0a1def9"
_SOURCE = f"https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/{_REVISION}"
_TOKENIZER = "tokenizer.json", "be50c3628f2bf5bb5e3a7f17b1f74611b2561a3a27eeab05e5aa30f411572037"
_ARM_MODEL = (
    "onnx/model_qint8_arm64.onnx",
    "4278337fd0ff3c68bfb6291042cad8ab363e1d9fbc43dcb499fe91c871902474",
)
_X86_MODEL = (
    "onnx/model_quint8_avx2.onnx",
    "b941bf19f1f1283680f449fa6a7336bb5600bdcd5f84d10ddc5cd72218a0fd21",
)
_CONTENT_TOKENS = 254  # MiniLM accepts 256 tokens, including CLS and SEP.
_WINDOW_STEP = 192
_MAX_WINDOWS_PER_PAGE = 8
_CACHED_PAGES = 1024
_INFERENCE_LOCK = Lock()


class SemanticRetrievalUnavailable(RuntimeError):
    """The local semantic ranking model could not be loaded."""


def _model_directory() -> Path:
    return Path(os.environ.get("EDCRAFT_MATERIAL_EMBEDDING_DIR", ".data/material-embedding"))


def _model_asset() -> tuple[str, str]:
    return _ARM_MODEL if platform.machine().lower() in {"arm64", "aarch64"} else _X86_MODEL


def _has_digest(path: Path, expected: str) -> bool:
    try:
        with path.open("rb") as asset:
            return file_digest(asset, "sha256").hexdigest() == expected
    except FileNotFoundError:
        return False


def prepare_assets() -> None:
    """Download and verify the two public model files, independently of retrieval."""
    directory = _model_directory()
    directory.mkdir(parents=True, exist_ok=True)
    assets = {"tokenizer.json": _TOKENIZER, "model.onnx": _model_asset()}
    with httpx.Client(follow_redirects=True, timeout=60) as client:
        for filename, (source_path, digest) in assets.items():
            target = directory / filename
            if _has_digest(target, digest):
                continue
            with NamedTemporaryFile(dir=directory, delete_on_close=False) as temporary:
                with client.stream("GET", f"{_SOURCE}/{source_path}") as response:
                    response.raise_for_status()
                    for block in response.iter_bytes():
                        temporary.write(block)
                temporary.flush()
                downloaded = Path(temporary.name)
                if not _has_digest(downloaded, digest):
                    raise RuntimeError(f"Checksum mismatch for {filename}.")
                # Close the file before replacing it on Windows.
                temporary.close()
                downloaded.replace(target)


class _MiniLM:
    def __init__(self, tokenizer: Tokenizer, session: ort.InferenceSession) -> None:
        self._page_cache: OrderedDict[bytes, npt.NDArray[np.float32]] = OrderedDict()
        tokenizer.no_truncation()
        tokenizer.no_padding()
        cls_id = tokenizer.token_to_id("[CLS]")
        sep_id = tokenizer.token_to_id("[SEP]")
        pad_id = tokenizer.token_to_id("[PAD]")
        if cls_id is None or sep_id is None or pad_id is None:
            raise ValueError("The MiniLM tokenizer is missing special tokens.")
        self.cls_id, self.sep_id, self.pad_id = cls_id, sep_id, pad_id
        self.tokenizer = tokenizer
        self.session = session

    def tokenize(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False).ids

    def encode(self, windows: tuple[list[int], ...]) -> npt.NDArray[np.float32]:
        width = max(map(len, windows)) + 2
        input_ids = np.full((len(windows), width), self.pad_id, dtype=np.int64)
        attention = np.zeros_like(input_ids)
        for index, window in enumerate(windows):
            length = len(window) + 2
            input_ids[index, :length] = [self.cls_id, *window, self.sep_id]
            attention[index, :length] = 1
        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention,
            "token_type_ids": np.zeros_like(input_ids),
        }
        hidden = cast(npt.NDArray[np.float32], self.session.run(["last_hidden_state"], inputs)[0])
        mask = attention[..., None].astype(np.float32)
        pooled = (hidden * mask).sum(axis=1) / mask.sum(axis=1)
        norm = np.linalg.norm(pooled, axis=1, keepdims=True)
        return np.asarray(pooled / np.maximum(norm, np.finfo(np.float32).eps), dtype=np.float32)

    def page_vectors(self, texts: list[str]) -> list[npt.NDArray[np.float32]]:
        # The cache is local to this encoder. Keep no original source text in it.
        keys = [sha256(text.encode()).digest() for text in texts]
        unique_pages = dict(zip(keys, texts, strict=True))
        vectors = {}
        for key, text in unique_pages.items():
            vector = self._page_cache.get(key)
            if vector is None:
                # Quantized activation ranges are batch-dependent. Each page has its own batch.
                vector = self.encode(tuple(_windows(self.tokenize(text))))
            vectors[key] = self._page_cache[key] = vector
            self._page_cache.move_to_end(key)
        while len(self._page_cache) > _CACHED_PAGES:
            self._page_cache.popitem(last=False)
        return [vectors[key] for key in keys]


@lru_cache(maxsize=1)
def _load_model(directory: Path) -> _MiniLM:
    try:
        tokenizer_path = directory / "tokenizer.json"
        model_path = directory / "model.onnx"
        if not _has_digest(tokenizer_path, _TOKENIZER[1]) or not _has_digest(
            model_path, _model_asset()[1]
        ):
            raise FileNotFoundError("Missing or invalid MiniLM assets.")
        ort.disable_telemetry_events()
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        session = ort.InferenceSession(str(model_path), options, providers=["CPUExecutionProvider"])
        return _MiniLM(Tokenizer.from_file(str(tokenizer_path)), session)
    except Exception as exc:
        raise SemanticRetrievalUnavailable(
            "The local semantic retrieval model is unavailable. Check the backend model setup."
        ) from exc


def _windows(ids: list[int]) -> list[list[int]]:
    last_start = max(0, len(ids) - _CONTENT_TOKENS)
    if not last_start:
        return [ids]
    count = min(_MAX_WINDOWS_PER_PAGE, ceil(last_start / _WINDOW_STEP) + 1)
    starts = [round(last_start * index / (count - 1)) for index in range(count)]
    return [ids[start : start + _CONTENT_TOKENS] for start in starts]


def semantic_scores(texts: list[str], topic: str) -> list[float]:
    """Give each page its highest passage cosine with the topic embedding."""
    with _INFERENCE_LOCK:
        model = _load_model(_model_directory())
        query = model.encode((model.tokenize(topic)[:_CONTENT_TOKENS],))[0]
        return [
            float(np.clip((vectors @ query).max(), -1, 1)) for vectors in model.page_vectors(texts)
        ]


if __name__ == "__main__":
    prepare_assets()
