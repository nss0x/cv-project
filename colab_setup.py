"""
Automated setup and training script for Google Colab
Run this once to install dependencies, then just rerun training after code changes
"""
import subprocess
import sys
import os

def install_dependencies():
    """Install all required packages for Colab"""
    print("=" * 60)
    print("INSTALLING DEPENDENCIES")
    print("=" * 60)
    
    packages = [
        'torch==2.11.0',
        'torchvision==0.16.0',
        'scikit-learn==1.8.0',
        'numpy==1.26.4',
        'pandas==2.3.0',
        'matplotlib==3.10.0',
        'torch-summary==1.4.5',
    ]
    
    for package in packages:
        print(f"\nInstalling {package}...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', package])
    
    print("\n✅ All dependencies installed!")

def mount_google_drive():
    """Mount Google Drive in Colab"""
    try:
        from google.colab import drive
        print("\n" + "=" * 60)
        print("MOUNTING GOOGLE DRIVE")
        print("=" * 60)
        drive.mount('/content/drive')
        print("✅ Google Drive mounted at /content/drive")
        return True
    except ImportError:
        print("⚠️  Not in Google Colab - Google Drive mount skipped")
        return False

def run_training(epochs=12, batch_size=16, dataset='brain_tumor'):
    """Run training with specified parameters"""
    print("\n" + "=" * 60)
    print(f"STARTING TRAINING ({epochs} epochs, batch size {batch_size})")
    print("=" * 60)
    
    cmd = [
        sys.executable, 
        'src/main.py',
        '--mode', 'baseline',
        '--dataset', dataset,
        '--epochs', str(epochs),
        '--batch-size', str(batch_size)
    ]
    
    subprocess.run(cmd)

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--install', action='store_true', help='Install dependencies')
    parser.add_argument('--mount-drive', action='store_true', help='Mount Google Drive')
    parser.add_argument('--epochs', type=int, default=12, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--dataset', type=str, default='brain_tumor', help='Dataset name')
    parser.add_argument('--train', action='store_true', help='Run training')
    
    args = parser.parse_args()
    
    if args.install:
        install_dependencies()
    
    if args.mount_drive:
        mount_google_drive()
    
    if args.train:
        run_training(args.epochs, args.batch_size, args.dataset)
    
    # Default: do first-time setup
    if not (args.install or args.mount_drive or args.train):
        print("COLAB SETUP SCRIPT - First time setup")
        print("\nUsage:")
        print("  python colab_setup.py --install          # Install dependencies (run once)")
        print("  python colab_setup.py --mount-drive      # Mount Google Drive (run once)")
        print("  python colab_setup.py --train            # Run training")
        print("  python colab_setup.py --epochs 12 --batch-size 16 --train")
        print("\nExample first-time setup in Colab:")
        print("  !python colab_setup.py --install")
        print("  !python colab_setup.py --mount-drive")
        print("  !python colab_setup.py --epochs 12 --batch-size 16 --train")
