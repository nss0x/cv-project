# Google Colab Setup Guide

This guide helps you run the CV project in Google Colab without zipping/uploading for every change.

## Option 1: GitHub + Colab (RECOMMENDED - No Re-uploading)

### First-Time Setup (5 minutes)

1. **Create a GitHub repository** and push this project:
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/cv-project.git
   git push -u origin main
   ```

2. **Open Google Colab** and create a new notebook

3. **In Colab, run these cells:**

   **Cell 1: Clone the repository**
   ```python
   !git clone https://github.com/YOUR_USERNAME/cv-project.git
   %cd cv-project
   ```

   **Cell 2: First-time setup (install dependencies)**
   ```python
   !python colab_setup.py --install
   ```

   **Cell 3: Mount Google Drive (optional, for saving results)**
   ```python
   !python colab_setup.py --mount-drive
   ```

   **Cell 4: Run training**
   ```python
   !python colab_setup.py --epochs 12 --batch-size 16 --train
   ```

### For Subsequent Changes (NO RE-UPLOADING NEEDED!)

1. **On your local machine**: Make code changes, then:
   ```bash
   git add .
   git commit -m "Your changes"
   git push
   ```

2. **In Colab**: Just update and rerun:
   ```python
   !git pull
   !python colab_setup.py --epochs 12 --batch-size 16 --train
   ```

That's it! No zipping, no uploading. Just `git push` and `git pull`.

---

## Option 2: Google Drive (For Non-Git Users)

### First-Time Setup

1. **Compress and upload** your project to Google Drive once:
   ```bash
   # On your machine
   Compress-Archive -Path . -DestinationPath cv-project.zip
   # Upload cv-project.zip to Google Drive root folder
   ```

2. **In Colab Cell 1: Mount and extract**
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   
   import subprocess
   subprocess.run(['unzip', '-q', '/content/drive/My Drive/cv-project.zip'])
   %cd cv-project
   ```

3. **Cell 2: Install dependencies**
   ```python
   !python colab_setup.py --install
   ```

4. **Cell 3: Run training**
   ```python
   !python colab_setup.py --epochs 12 --batch-size 16 --train
   ```

### For Changes (Still need to re-upload, sorry 😅)

- Make changes locally → zip → upload to Drive → extract in Colab
- Not as smooth as GitHub, but you only extract once!

---

## Option 3: Direct Upload (Simplest, but Re-upload Each Time)

1. **In Colab**: Upload the project folder directly
   ```python
   from google.colab import files
   files.upload()  # Select cv-project.zip
   !unzip -q cv-project.zip
   %cd cv-project
   ```

2. **Cell 2-4: Same as Option 2 above**

---

## Complete Colab Notebook Template

```python
# ============================================================
# CELL 1: Clone from GitHub (or upload zip if not using GitHub)
# ============================================================
!git clone https://github.com/YOUR_USERNAME/cv-project.git
%cd cv-project

# ============================================================
# CELL 2: Install dependencies (run once)
# ============================================================
!python colab_setup.py --install

# ============================================================
# CELL 3: Mount Google Drive (optional)
# ============================================================
!python colab_setup.py --mount-drive

# ============================================================
# CELL 4: Run training
# ============================================================
!python colab_setup.py --epochs 12 --batch-size 16 --train

# ============================================================
# CELL 5: Pull latest changes (GitHub only)
# ============================================================
!git pull
!python colab_setup.py --epochs 12 --batch-size 16 --train
```

---

## Custom Training Parameters

Adjust training parameters easily:

```python
# 10 epochs, batch size 32
!python colab_setup.py --epochs 10 --batch-size 32 --train

# Different dataset
!python colab_setup.py --dataset chest_xray --epochs 12 --train

# All options
!python colab_setup.py --help
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError` | Run `!python colab_setup.py --install` again |
| `FileNotFoundError` for data | Make sure you have downloaded the dataset locally first |
| CUDA not available | Normal - Colab CPU/TPU will be used automatically |
| Drive mount fails | You're not in Colab, or mount failed - skip this step |

---

## Summary

| Method | First Setup | Each Change | Best For |
|--------|------------|------------|----------|
| **GitHub** | 5 min | 30 sec (git push/pull) | Active development ✅ |
| **Google Drive** | 10 min | 3 min (zip upload) | Occasional changes |
| **Direct Upload** | 3 min | 3 min (re-upload) | One-off use |

**Recommendation: Use GitHub** - Once it's set up, you literally just `git push` → `git pull` → `run`. No zipping ever again! 🚀
