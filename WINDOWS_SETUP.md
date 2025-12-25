# Setup Instructions for Windows

Windows-specific setup instructions for the SNS Comic Analysis project.

## Prerequisites

1. **Python 3.7 or newer**
   - Download from: https://www.python.org/downloads/
   - During installation, CHECK "Add Python to PATH"

2. **Git** (optional, for cloning)
   - Download from: https://git-scm.com/download/win

3. **GPU Support (Optional)**
   - NVIDIA GPU with CUDA capability
   - NVIDIA CUDA Toolkit: https://developer.nvidia.com/cuda-downloads
   - cuDNN: https://developer.nvidia.com/cudnn

## Installation Steps

### 1. Clone or Download Repository

**Option A: Using Git (recommended)**
```powershell
git clone https://github.com/yourusername/sns-comic-analysis.git
cd sns-comic-analysis
```

**Option B: Download ZIP**
- Download ZIP from GitHub
- Extract to desired location
- Open PowerShell in that folder

### 2. Create Virtual Environment

```powershell
# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1

# If you get execution policy error, run:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### 3. Install Dependencies

```powershell
# Upgrade pip first
python -m pip install --upgrade pip

# Install requirements
pip install -r requirements.txt
```

**Note**: PyTorch installation may take 5-10 minutes as it's large (~2GB)

### 4. Configure API Keys (Optional)

For GPT-4 analysis:

```powershell
# Copy example environment file
Copy-Item .env.example .env

# Open .env in your editor and add your OpenAI API key
notepad .env
```

Or set environment variable directly:

```powershell
$env:OPENAI_API_KEY = "your-api-key-here"
```

## Running the Analysis

```powershell
# Activate virtual environment first (if not already active)
.\venv\Scripts\Activate.ps1

# Run with default settings
python clip_revenue_analysis.py

# Use GPU (if available)
python clip_revenue_analysis.py --device cuda

# Include GPT-4 analysis
python clip_revenue_analysis.py --gpt_api_key $env:OPENAI_API_KEY
```

## Common Issues & Solutions

### Issue: "python is not recognized"
**Solution**: Python not in PATH
- Reinstall Python and CHECK "Add Python to PATH"
- Or use `py` instead: `py -m venv venv`

### Issue: Execution policy error
**Solution**:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Issue: Virtual environment not activating
**Solution**: Use full path to activate script
```powershell
& ".\venv\Scripts\Activate.ps1"
```

### Issue: CUDA/GPU not detected
**Solution**: Use CPU instead
```powershell
python clip_revenue_analysis.py --device cpu
```

### Issue: Out of memory error
**Solution**: CPU is more memory-efficient than GPU for this dataset
```powershell
python clip_revenue_analysis.py --device cpu
```

### Issue: Module not found (e.g., "No module named 'torch'")
**Solution**: Ensure virtual environment is activated
```powershell
# Check if venv is active (should show (venv) in prompt)
# If not:
.\venv\Scripts\Activate.ps1

# Then reinstall:
pip install -r requirements.txt
```

## Deactivating Virtual Environment

When done working:

```powershell
deactivate
```

## Next Steps

1. See [QUICKSTART.md](QUICKSTART.md) for first run examples
2. Read [README.md](README.md) for full documentation
3. Check [CONTRIBUTING.md](CONTRIBUTING.md) to contribute improvements

---

**Having trouble?** Open an issue on GitHub with:
- Your Python version: `python --version`
- Your OS: Windows version
- Error message/traceback
