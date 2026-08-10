from pathlib import Path

from huggingface_hub import snapshot_download


MODEL_NAME = "BAAI/bge-small-en-v1.5"

PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "embedding"
    / "bge-small-en-v1.5"
)


def main():
    print(
        f"Downloading model: {MODEL_NAME}"
    )

    print(
        f"Target directory: {MODEL_DIR}"
    )

    snapshot_download(
        repo_id=MODEL_NAME,
        local_dir=MODEL_DIR,
        ignore_patterns=[
            "onnx/*",
            "pytorch_model.bin",
        ],
    )

    print(
        "Embedding model downloaded successfully."
    )


if __name__ == "__main__":
    main()