from setuptools import setup, find_packages
import sys
import platform
import subprocess

# Check Python version
if sys.version_info < (3, 8):
    sys.exit('Python >= 3.8 is required')

def check_system_dependencies():
    """Check for required system-level dependencies"""
    system = platform.system()
    if system == 'Darwin':  # macOS
        try:
            # Check if Xcode command line tools are installed
            subprocess.run(['xcode-select', '-p'], check=True, capture_output=True)
        except subprocess.CalledProcessError:
            sys.exit('Xcode command line tools are required. Install with: xcode-select --install')
    elif system == 'Linux':
        required_packages = ['build-essential', 'python3-dev']
        try:
            subprocess.run(['dpkg', '-l'] + required_packages, check=True, capture_output=True)
        except subprocess.CalledProcessError:
            sys.exit(f'Required system packages missing. Install with: sudo apt-get install {" ".join(required_packages)}')

def main():
    check_system_dependencies()
    
    setup(
        name="astros-analytics",
        version="0.1.0",
        packages=find_packages(),
        python_requires=">=3.8",
        install_requires=[
            line.strip()
            for line in open('requirements-core.txt')
            if line.strip() and not line.startswith('-r')
        ],
        extras_require={
            'full': [
                line.strip()
                for line in open('requirements-full.txt')
                if line.strip() and not line.startswith('-r')
            ]
        }
    )

if __name__ == '__main__':
    main() 