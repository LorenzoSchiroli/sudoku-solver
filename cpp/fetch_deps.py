from pathlib import Path
import urllib.request
import zipfile
import tarfile

ROOT = Path(__file__).resolve().parents[1]
DEPS = ROOT / "deps"

def download(url: str, dst: Path):
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url}")
    urllib.request.urlretrieve(url, dst)

def extract_zip(src: Path, dst: Path):
    if dst.exists():
        return
    print(f"Extracting {src}")
    dst.mkdir(parents=True)
    with zipfile.ZipFile(src) as z:
        z.extractall(dst)

def extract_tgz(src: Path, dst: Path):
    if dst.exists():
        return
    print(f"Extracting {src}")
    dst.mkdir(parents=True)
    with tarfile.open(src) as t:
        t.extractall(dst)

# =========================================================
# OpenCV
# =========================================================

# Android
opencv_android_zip = DEPS / "opencv/opencv-android.zip"
download(
    "https://github.com/opencv/opencv/releases/download/4.13.0/opencv-4.13.0-android-sdk.zip",
    opencv_android_zip,
)
extract_zip(opencv_android_zip, DEPS / "opencv/android")

# iOS
opencv_ios_zip = DEPS / "opencv/opencv-ios.zip"
download(
    "https://github.com/opencv/opencv/releases/download/4.13.0/opencv-4.13.0-ios-framework.zip",
    opencv_ios_zip,
)
extract_zip(opencv_ios_zip, DEPS / "opencv/ios")

# macOS: dev-only (Homebrew / system)
# =========================================================
# ONNX Runtime
# =========================================================

# ---------- Android ----------
# AAR from Maven Central (C++ users must extract manually)

onnx_android_aar = DEPS / "onnxruntime/onnxruntime-android.aar"
download(
    "https://repo1.maven.org/maven2/com/microsoft/onnxruntime/onnxruntime-android/1.23.2/onnxruntime-android-1.23.2.aar",
    onnx_android_aar,
)
extract_zip(onnx_android_aar, DEPS / "onnxruntime/android")

# ---------- iOS ----------
# Official path is CocoaPods.
# We download the framework zip only if you explicitly want no Pods.

onnx_ios_zip = DEPS / "onnxruntime/onnxruntime-ios.zip"
download(
    "https://github.com/csukuangfj/onnxruntime-libs/releases/download/v1.23.2/onnxruntime-ios-static-xcframework-1.23.2.zip",
    onnx_ios_zip,
)
extract_zip(onnx_ios_zip, DEPS / "onnxruntime/ios")
