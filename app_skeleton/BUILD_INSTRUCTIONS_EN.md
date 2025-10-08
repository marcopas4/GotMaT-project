# Build Instructions - RAG Prefettura Executable

## Prerequisites

Before creating the executable, make sure you have:

1. **Python 3.11+** installed on your system
2. **All required packages** installed:
   ```bash
   pip install -r requirements.txt
   ```

## Creating the Executable

### Step 1: Install PyInstaller

If you haven't already:
```bash
pip install pyinstaller
```

### Step 2: Run the Build Script

```bash
python build_exe.py
```

The script will automatically perform all necessary steps:
- ✅ Clean previous build folders
- ✅ Create Streamlit configuration
- ✅ Generate custom .spec file
- ✅ Build executable with PyInstaller
- ✅ Copy necessary files (data, configurations)
- ✅ Create batch launcher
- ✅ Generate README for end users

### Step 3: Test the Executable

1. Go to the `dist/` folder
2. Double click on `Avvia_RAG_Prefettura.bat` (Windows)
3. Or run `./RAG_Prefettura` (Linux/Mac)
4. The application should start and open the browser at http://localhost:8501

## Generated Files Structure

After the build, you'll find in the `dist/` folder:

```
dist/
├── RAG_Prefettura.exe              # Main executable (~200-500 MB)
├── Avvia_RAG_Prefettura.bat        # Automatic launcher
├── README.txt                       # Instructions for end users
├── src/                             # Embedded source code
├── data/                            # Knowledge base and data
│   ├── faiss_index/                # Vector indices
│   └── knowledge_base/             # Base documents
└── .streamlit/                     # Streamlit configurations
    └── config.toml
```

## Distribution

To distribute the application to end users:

1. **Copy the entire `dist/` folder** to a USB drive or compress it into a ZIP file
2. Users will simply need to:
   - Extract/copy the folder to their computer
   - Double click on `Avvia_RAG_Prefettura.bat`

**Note**: The executable includes all necessary Python dependencies, so users do **NOT** need to have Python installed!

## File Size

- **Base executable**: ~200-300 MB
- **With fine-tuned model**: depends on model size (can reach several GB)
- **Total dist/ folder**: varies based on knowledge base data

## Customization

### Change the Icon

1. Create or download an `.ico` file (recommended size: 256x256)
2. Rename it to `icon.ico` and place it in the main folder
3. The script will automatically include it in the executable

### Modify Settings

You can modify the `.streamlit/config.toml` file before building to customize:
- Application port (default: 8501)
- Color theme
- Other Streamlit configurations

## Troubleshooting

### Build Failed

If the build fails, check:

1. **Are all dependencies installed?**
   ```bash
   pip install -r requirements.txt
   ```

2. **Is PyInstaller up to date?**
   ```bash
   pip install --upgrade pyinstaller
   ```

3. **Sufficient disk space?**
   - At least 2-3 GB of free space is needed for the build process

### Executable Won't Start

1. **Antivirus**: Some antiviruses block PyInstaller executables
   - Add an exception for `RAG_Prefettura.exe`

2. **Missing Dependencies**: If Tesseract OCR is missing
   - Include Tesseract in the dist/ folder or
   - Ask users to install it separately

3. **Console Test**: Run from command prompt to see errors
   ```cmd
   cd dist
   RAG_Prefettura.exe
   ```

### Excessive Size

To reduce size:

1. **Exclude unnecessary files** by modifying `build_exe.py`:
   ```python
   excludes=['matplotlib', 'jupyter', 'notebook', 'pytest'],
   ```

2. **Use UPX** to compress the executable (already enabled)

3. **Remove unused models** from the data/ folder

## Building for Other Operating Systems

**Important**: PyInstaller creates executables only for the system on which it's run:
- Build on Windows → .exe file (for Windows)
- Build on Linux → Linux executable
- Build on macOS → macOS app

Cross-compilation is not possible!

## Updates

To update the application:

1. Modify the source code
2. Re-run `python build_exe.py`
3. Redistribute the updated `dist/` folder

Users will need to replace their folder with the new version.

## Performance

The executable might be slightly slower at startup compared to running with Python directly:
- **First startup**: 10-30 seconds (temporary files extraction)
- **Subsequent startups**: 5-15 seconds

This is normal for PyInstaller applications.

## Alternative Build Methods

### Method 1: Automatic Script (Recommended)

**Windows:**
```bash
build.bat
```

**Linux/Mac:**
```bash
chmod +x build.sh
./build.sh
```

### Method 2: Direct Python Script
```bash
python build_exe.py
```

### Method 3: Manual PyInstaller (Advanced)

```bash
# Clean previous builds
rm -rf build/ dist/ *.spec

# Create .spec file
pyi-makespec --onefile --windowed \
    --add-data "src:src" \
    --add-data "data:data" \
    --add-data ".streamlit:.streamlit" \
    --hidden-import streamlit \
    --hidden-import torch \
    --hidden-import transformers \
    --name RAG_Prefettura \
    app.py

# Build
pyinstaller RAG_Prefettura.spec
```

## Advanced Configuration

### Modify .spec File

The `build_exe.py` script creates a custom `.spec` file. You can modify it for:

- **Add hidden imports**:
  ```python
  hiddenimports=[
      'your_module',
      'another_module',
  ],
  ```

- **Exclude modules**:
  ```python
  excludes=[
      'tkinter',
      'matplotlib',
  ],
  ```

- **Add binary files**:
  ```python
  binaries=[
      ('path/to/file.dll', '.'),
  ],
  ```

### Optimize Build Size

1. **Use virtual environment** with only necessary packages
2. **Remove development dependencies** (pytest, jupyter, etc.)
3. **Use quantized models** (GGUF format)
4. **Enable UPX compression** (already enabled in script)

### Add Splash Screen (Optional)

Modify the .spec file to add a splash screen:

```python
splash = Splash(
    'path/to/splash_image.png',
    binaries=a.binaries,
    datas=a.datas,
    text_pos=(10, 50),
    text_size=12,
    text_color='white'
)
```

## Testing Checklist

Before distributing, verify:

- [ ] Application runs locally with `streamlit run app.py`
- [ ] Build completes without errors
- [ ] Executable starts correctly
- [ ] Document upload features work
- [ ] Model responses are correct
- [ ] `data/` folder contains knowledge base
- [ ] Fine-tuned model is included (if applicable)
- [ ] README for end users is clear
- [ ] Tested on a clean computer without Python

## Security Considerations

### Code Signing (Recommended for Production)

To avoid antivirus false positives:

1. **Windows**: Use signtool.exe with a code signing certificate
   ```cmd
   signtool sign /f certificate.pfx /p password RAG_Prefettura.exe
   ```

2. **macOS**: Use codesign
   ```bash
   codesign --force --sign "Developer ID" RAG_Prefettura.app
   ```

### Whitelist Application

For enterprise distribution:
- Add hash to antivirus whitelist
- Use group policy to distribute
- Contact IT department for approval

## Docker Alternative

If executable size is too large, consider Docker:

```bash
# Build Docker image
docker build -t rag-prefettura .

# Run container
docker-compose up
```

Users will need Docker Desktop installed.

## Continuous Integration

For automated builds:

### GitHub Actions Example

```yaml
name: Build Executable

on:
  push:
    tags:
      - 'v*'

jobs:
  build:
    runs-on: windows-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.11'
      - run: pip install -r requirements.txt
      - run: python build_exe.py
      - uses: actions/upload-artifact@v2
        with:
          name: rag-prefettura-exe
          path: dist/
```

## Support Resources

### Official Documentation
- **PyInstaller**: https://pyinstaller.org/
- **Streamlit**: https://docs.streamlit.io/
- **Python**: https://docs.python.org/

### Common Issues Database
- PyInstaller FAQ: https://pyinstaller.org/en/stable/faq.html
- Streamlit Forums: https://discuss.streamlit.io/

### Getting Help

1. **Check logs**: Look in the `logs/` folder
2. **Run in console**: See error messages directly
3. **Test script**: Run `python test_changes.py`
4. **Community**: Ask on StackOverflow with tags: `pyinstaller`, `streamlit`

## Final Notes

### What's Included
✅ All Python dependencies  
✅ Source code  
✅ Data files  
✅ Streamlit configuration  
✅ Required libraries  

### What's NOT Included
❌ Python interpreter (embedded)  
❌ System-specific libraries (may need installation)  
❌ Tesseract OCR (must be installed separately)  

### Recommendations
- Test on target operating system before distribution
- Provide installation guide for end users
- Include contact information for support
- Version your builds (e.g., v1.0.0, v1.0.1)
- Keep source code backed up

### Minimum System Requirements

**End Users Need:**
- Windows 10+ / Linux (Ubuntu 20.04+) / macOS 10.15+
- 8GB RAM (16GB recommended)
- 2GB free disk space
- Multi-core CPU (4+ cores recommended)
- Internet connection (for first-time setup)

**Developers Need:**
- All of the above, plus:
- Python 3.11+
- 10GB free disk space (for build process)
- Administrator privileges (for some installations)

---

**Version**: 1.0.0  
**Compatibility**: Windows 10/11, Linux, macOS, Python 3.11+  
**Last Updated**: October 8, 2025

---

For questions or issues, refer to the complete documentation in `CHANGES_SUMMARY.md` and `README_CHANGES.md`.

Good luck! 🚀
