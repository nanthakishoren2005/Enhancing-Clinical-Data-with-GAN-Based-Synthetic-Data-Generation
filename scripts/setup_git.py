"""
Download and extract portable MinGit for Windows (no admin rights needed).
"""
import urllib.request
import zipfile
import shutil
import os
from pathlib import Path

target_dir = Path.home() / "AppData" / "Local" / "Programs" / "Git"
git_exe = target_dir / "cmd" / "git.exe"

if not git_exe.exists():
    target_dir.mkdir(parents=True, exist_ok=True)
    url = "https://github.com/git-for-windows/git/releases/download/v2.47.1.windows.1/MinGit-2.47.1-64-bit.zip"
    zip_path = target_dir / "mingit.zip"
    
    print(f"Downloading MinGit from {url}...")
    headers = {"User-Agent": "Mozilla/5.0"}
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req) as response, open(zip_path, "wb") as out_file:
        shutil.copyfileobj(response, out_file)
        
    print(f"Extracting {zip_path} to {target_dir}...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(target_dir)
        
    if zip_path.exists():
        zip_path.unlink()
        
    print("MinGit installed successfully!")
else:
    print(f"MinGit already present at {git_exe}")

print(f"Git executable location: {git_exe}")
