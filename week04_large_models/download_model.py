from constants import MODEL_DIR, MODEL_NAME
from huggingface_hub import snapshot_download


def download_model(
    model_name: str = MODEL_NAME,
    model_dir: str = MODEL_DIR,
) -> None:
    snapshot_download(
        repo_id=model_name,
        local_dir=model_dir,
    )


if __name__ == "__main__":
    download_model()
