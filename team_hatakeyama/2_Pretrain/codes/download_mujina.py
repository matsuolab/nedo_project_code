# インターネットでダウンロードできる時に実行
from huggingface_hub import snapshot_download
download_path = snapshot_download(repo_id="misdelivery/Mujina-test")