import http.cookiejar
import json
import os
import re
import shutil
import urllib.error
import urllib.request

import numpy as np
from scipy.io import loadmat

from task_and_baseline import baseline, build_task_helpers


DATA_FILE_ID = "1BBHVSI4KB-B8OX46eN1Nm4ARCeq6Rui4"
DATA_FILE = "challenge.mat"
helpers = None


def _request(opener, url):
    return opener.open(
        urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0"},
        )
    )


def _confirm_token_from_cookies(cookie_jar):
    for cookie in cookie_jar:
        if cookie.name.startswith("download_warning"):
            return cookie.value
    return None


def _confirm_token_from_html(html):
    patterns = (
        r'name="confirm"\s+value="([^"]+)"',
        r"confirm=([0-9A-Za-z_]+)",
    )
    for pattern in patterns:
        match = re.search(pattern, html)
        if match:
            return match.group(1)
    return None


def _stream_download(opener, url, destination):
    with _request(opener, url) as response:
        content_type = response.headers.get("Content-Type", "")
        if "text/html" in content_type:
            return response.read(512_000).decode("utf-8", errors="ignore")

        tmp_destination = f"{destination}.part"
        with open(tmp_destination, "wb") as f:
            shutil.copyfileobj(response, f)
        os.replace(tmp_destination, destination)
        return None


def _download_with_gdown(path):
    try:
        import gdown
    except ImportError:
        return False

    verify = os.environ.get("SMILES_ALLOW_INSECURE_DOWNLOAD") != "1"
    result = gdown.download(id=DATA_FILE_ID, output=path, quiet=False, verify=verify)
    return bool(result) and os.path.exists(path)


def _download_with_stdlib(path):
    cookie_jar = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cookie_jar))
    base_url = f"https://drive.google.com/uc?export=download&id={DATA_FILE_ID}"

    html = _stream_download(opener, base_url, path)
    if html is None:
        return True

    token = _confirm_token_from_cookies(cookie_jar) or _confirm_token_from_html(html)
    if token:
        confirmed_url = f"{base_url}&confirm={token}"
        html = _stream_download(opener, confirmed_url, path)
        if html is None:
            return True

    return False


def ensure_dataset(path=DATA_FILE):
    if os.path.exists(path):
        return path

    errors = []
    try:
        if _download_with_gdown(path):
            return path
    except Exception as exc:  # Google Drive failures vary by environment.
        errors.append(exc)

    try:
        if _download_with_stdlib(path):
            return path
    except urllib.error.URLError as exc:
        errors.append(exc)

    message = (
        "Could not download challenge.mat. Place it next to applicant_solution.py "
        "or rerun with internet access. If your network uses a local TLS proxy, "
        "set SMILES_ALLOW_INSECURE_DOWNLOAD=1 for the download step."
    )
    if errors:
        raise RuntimeError(message) from errors[-1]
    raise RuntimeError(message)


def _rank1_projection(band_matrix):
    cov = band_matrix.conj().T @ band_matrix / band_matrix.shape[0]
    _, vecs = np.linalg.eigh(cov)
    shared = band_matrix @ vecs[:, -1]
    denom = np.vdot(shared, shared) + 1e-30
    return np.column_stack(
        [
            (np.vdot(shared, band_matrix[:, ch]) / denom) * shared
            for ch in range(band_matrix.shape[1])
        ]
    )


def _band_matrix(x):
    score_filter = helpers["score_filter"]
    return np.column_stack([score_filter(x[:, ch]) for ch in range(x.shape[1])])


def _estimate_external_rank1(residual):
    band_residual = _band_matrix(residual)
    rank1 = _rank1_projection(band_residual)

    # The scorer filters the removed component once more. Fit a single complex
    # scale against that exact filtered view to avoid systematic under/overshoot.
    filtered_rank1 = _band_matrix(rank1)
    denom = np.vdot(filtered_rank1, filtered_rank1) + 1e-30
    scale = np.vdot(filtered_rank1, band_residual) / denom
    max_gain = 2.0
    if abs(scale) > max_gain:
        scale *= max_gain / abs(scale)
    return scale * rank1


def your_canceller(tx_n, rx):
    """Cancel TX-driven leakage, then remove the shared external interferer."""
    del tx_n
    if helpers is None:
        raise RuntimeError("helpers must be initialised before calling your_canceller")

    fit_tx_prediction = helpers["fit_tx_prediction"]

    tx_prediction = fit_tx_prediction(rx)
    external_prediction = _estimate_external_rank1(rx - tx_prediction)

    refined_tx_prediction = fit_tx_prediction(rx - external_prediction)
    refined_external_prediction = _estimate_external_rank1(rx - refined_tx_prediction)

    return rx - refined_tx_prediction - refined_external_prediction


def main():
    global helpers

    ensure_dataset()
    data = loadmat(DATA_FILE, simplify_cells=True)
    tx = data["tx"].astype(np.complex128)
    rx = data["rx"].astype(np.complex128)
    fs = float(data["Fs"])
    n_samples, _ = tx.shape

    tx_n = tx / (np.sqrt(np.mean(np.abs(tx) ** 2, axis=0, keepdims=True)) + 1e-30)
    helpers = build_task_helpers(tx_n, fs, n_samples)

    print("\n=== Baseline ===")
    baseline_reds, baseline_avg = helpers["score"](
        rx, baseline(tx_n, rx, helpers["fit_tx_prediction"]), label="baseline"
    )

    print("=== Your Solution ===")
    yours_reds, yours_avg = helpers["score"](rx, your_canceller(tx_n, rx), label="yours")

    results = {
        "baseline": {
            "per_channel_db": baseline_reds,
            "average_db": baseline_avg,
        },
        "yours": {
            "per_channel_db": yours_reds,
            "average_db": yours_avg,
        },
    }

    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
