from azure.identity import ClientSecretCredential
from azure.storage.blob import BlobServiceClient
import streamlit as st
from pathlib import Path
import pickle, tempfile, json

_LOCAL_CACHE = Path(tempfile.gettempdir()) / "nzmap_cache"
_LOCAL_CACHE.mkdir(exist_ok=True)

@st.cache_resource(show_spinner=False)
def get_container_client():
    cfg = st.secrets["azure"]
    cred = ClientSecretCredential(
        tenant_id=cfg["tenant_id"],
        client_id=cfg["client_id"],
        client_secret=cfg["client_secret"],
    )
    url = f"https://{cfg['storage_account']}.blob.core.windows.net"
    client = BlobServiceClient(account_url=url, credential=cred)
    return client.get_container_client(cfg["container"])

def _blob_name(local_path: Path) -> str:
    parts = local_path.parts
    if "assets" in parts:
        idx = parts.index("assets")
        return "/".join(parts[idx+1:])
    return str(local_path)

def load_pkl_blob(local_path: Path):
    cache_path = _LOCAL_CACHE / _blob_name(local_path)
    if cache_path.exists():
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    try:
        data = get_container_client().download_blob(_blob_name(local_path)).readall()
    except Exception:
        return None
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        f.write(data)
    return pickle.loads(data)

def load_json_blob(local_path: Path):
    if local_path.exists():
        with open(local_path) as f:
            return json.load(f)
    try:
        data = get_container_client().download_blob(_blob_name(local_path)).readall()
        return json.loads(data)
    except Exception:
        return {}
