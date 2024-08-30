import os

from dotenv import load_dotenv
from huggingface_hub import login, snapshot_download
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_fixed

DATASET_ID = "HuggingFaceFW/fineweb-edu"
LOCAL_DIR = "data/fineweb-edu"
ALLOW_PATTERN = "sample/"


def should_retry(exception):
    # OSError: [Errno 28] No space left on device
    if isinstance(exception, OSError) and exception.errno == 28:
        return False
    return True


def after_retry(retry_state):
    print(
        f"({retry_state.attempt_number:02d}/{retry_state.retry_object.stop.max_attempt_number:02d}) Exception: {retry_state.outcome.exception()}"
    )


@retry(
    stop=stop_after_attempt(20),
    wait=wait_fixed(10),
    retry=retry_if_exception(should_retry),
    after=after_retry,
)
def download_with_backoff(*args, **kwargs):
    return snapshot_download(*args, **kwargs)


if __name__ == "__main__":
    load_dotenv()
    login(token=os.environ["HF_TOKEN"])

    download_with_backoff(
        repo_id=DATASET_ID,
        repo_type="dataset",
        local_dir=LOCAL_DIR,
        allow_patterns=ALLOW_PATTERN,
    )
