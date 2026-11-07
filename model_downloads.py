"""Download only the four public RevealLayer artifacts into this node folder."""

import hashlib
import logging
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models" / "reveallayer"
REPOSITORY = "qihoo360/RevealLayer"
REVISION = "79c1969fe7d1d7f768d76e24fac3df89c495d586"
FILES = {
    "pytorch_lora_weights.safetensors": (298933416, "f5069b954306f882d384c059fbc07b931d2d4d5c4d6d4d16e8d888f8c512bc7c"),
    "layer_pe.pt": (75312, "c3144b547bf8dad8246ee0070c4ea37b7d12cf5aa54f0aacae45ba3cc12cf6f0"),
    "Refiner.pt": (56997104, "752aff6cb62a44e35098b7e21eabf93744eb204e4ebb6d9615e3d1daffe40d3c"),
    "xvae/transparent_decoder_ckpt.pth": (341128512, "44653f514096dedf906354310d21ae9a62c812d79ac69f8dc4e6d7b8575ee8c3"),
}
_VERIFIED = set()


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verified(path, size, expected_hash):
    if not path.is_file() or path.stat().st_size != size:
        return False
    signature = (str(path), size, path.stat().st_mtime_ns, expected_hash)
    if signature not in _VERIFIED:
        if sha256(path) != expected_hash:
            return False
        _VERIFIED.add(signature)
    return True


def download_file(path, url, size, expected_hash, progress, interrupted):
    if verified(path, size, expected_hash):
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    if partial.exists() and partial.stat().st_size >= size:
        if verified(partial, size, expected_hash):
            partial.replace(path)
            return
        partial.unlink()
    offset = partial.stat().st_size if partial.exists() else 0
    headers = {"User-Agent": "TBG-RevealLayer/0.1.0"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    interrupted()
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=60) as response:
        if response.status != 206:
            offset = 0
        elif response.headers.get("Content-Range", "").split("-")[0] != f"bytes {offset}":
            raise RuntimeError(f"Invalid resume response for {path.name}; retry the node.")
        completed = offset
        progress(completed, size)
        with partial.open("ab" if offset else "wb") as stream:
            while True:
                interrupted()
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                stream.write(chunk)
                completed += len(chunk)
                if completed > size:
                    raise RuntimeError(f"Download larger than expected: {path.name}")
                progress(completed, size)
    if not verified(partial, size, expected_hash):
        if partial.stat().st_size == size:
            partial.unlink()
        raise RuntimeError(f"Incomplete or invalid RevealLayer download: {path.name}. Queue again to retry.")
    partial.replace(path)


def ensure_reveallayer_models(node_id=None):
    # These dependencies are imported after the first-run dependency check.
    from filelock import FileLock
    import comfy.model_management
    import comfy.utils

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for filename, (size, expected_hash) in FILES.items():
        path = MODEL_DIR / filename
        with FileLock(str(MODEL_DIR / ".download.lock")):
            if verified(path, size, expected_hash):
                continue
            logging.info("TBG RevealLayer: downloading %s from %s", filename, REPOSITORY)
            bar = comfy.utils.ProgressBar(size, node_id=node_id)
            download_file(
                path,
                f"https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/{filename}?download=true",
                size, expected_hash,
                lambda value, total: bar.update_absolute(value, total=total),
                comfy.model_management.throw_exception_if_processing_interrupted,
            )
    return MODEL_DIR, MODEL_DIR / "xvae" / "transparent_decoder_ckpt.pth"
