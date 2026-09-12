"""Restore the exact original UCI mathematics CSV. Standard library only."""

from pathlib import Path
import hashlib
import io
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
URL = "https://archive.ics.uci.edu/static/public/320/student+performance.zip"
EXPECTED = "e47f9ee225e1ee6e69b7564e6dac7123e80b8486677fe111f351964cef5dec80"


def main():
    with urllib.request.urlopen(URL, timeout=30) as response:
        downloaded = response.read()
    with zipfile.ZipFile(io.BytesIO(downloaded)) as outer:
        with zipfile.ZipFile(io.BytesIO(outer.read("student.zip"))) as inner:
            csv_bytes = inner.read("student-mat.csv")
    if hashlib.sha256(csv_bytes).hexdigest() != EXPECTED:
        raise ValueError("Source CSV checksum changed; the existing file was not overwritten.")
    destination = ROOT / "data/student-mat.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(csv_bytes)
    print("Restored data/student-mat.csv from UCI; checksum verified.")


if __name__ == "__main__":
    main()
