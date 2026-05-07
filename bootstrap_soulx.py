"""Download SoulX-Singer weights + NLTK data needed at runtime.

Idempotent — each step exits early if its artefact is already present.
Run once before the first inference call (HF Spaces, Docker, or local
fresh clone). Safe to call from app.py on startup.

Local dev that already has a SoulX install at $SOULX_ROOT can skip the
weights step (handled automatically by the existence check).
"""
from pathlib import Path
import os
import sys

ROOT = Path(__file__).parent.resolve()
SOULX_ROOT = Path(os.environ.get("SOULX_ROOT", str(ROOT / "vendor" / "SoulX-Singer")))
WEIGHTS_DIR = SOULX_ROOT / "pretrained_models" / "SoulX-Singer"
HF_REPO = "Soul-AILab/SoulX-Singer"

# g2p_en needs these NLTK corpora.
NLTK_PACKAGES = ("cmudict", "averaged_perceptron_tagger_eng")


def ensure_nltk():
    """Download NLTK data g2p_en relies on if not already cached."""
    import nltk
    for pkg in NLTK_PACKAGES:
        try:
            # different packages live under different lookup paths
            for prefix in ("corpora", "taggers", "tokenizers"):
                try:
                    nltk.data.find(f"{prefix}/{pkg}")
                    break
                except LookupError:
                    continue
            else:
                raise LookupError
            print(f"[bootstrap] nltk:{pkg} already present")
        except LookupError:
            print(f"[bootstrap] nltk download: {pkg}")
            nltk.download(pkg, quiet=True)


def ensure_soulx_weights():
    """Download SoulX-Singer weights if not already present."""
    target = WEIGHTS_DIR / "model.pt"
    if target.exists():
        print(f"[bootstrap] weights already present at {target}")
        return
    print(f"[bootstrap] downloading {HF_REPO} -> {WEIGHTS_DIR}")
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id=HF_REPO,
        local_dir=str(WEIGHTS_DIR),
        local_dir_use_symlinks=False,
    )
    if not target.exists():
        print(f"[bootstrap] WARNING: {target} not found after download. "
              f"Repo layout may differ from expected.", file=sys.stderr)
        sys.exit(1)
    print(f"[bootstrap] weights done")


def main():
    ensure_nltk()
    ensure_soulx_weights()


if __name__ == "__main__":
    main()
