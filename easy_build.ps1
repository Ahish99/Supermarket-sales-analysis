
# 1. Create a clean virtual environment
if (-not (Test-Path "env")) {
    Write-Host "Creating virtual environment 'env'..."
    python -m venv env
}

# 2. Upgrade pip and Install dependencies
Write-Host "Installing dependencies..."
.\env\Scripts\python -m pip install --upgrade pip
.\env\Scripts\python -m pip install -r requirements.txt

# Force uninstall of potential conflict packages (ignoring errors if not found)
# This prevents PyInstaller from picking up ghost packages from system or dependencies
Write-Host "Ensuring no conflicting Qt bindings..."
.\env\Scripts\python -m pip uninstall -y PyQt5 PySide6 PyQt6 PySide2 qtpy pyqtgraph

# 3. Clean previous builds
if (Test-Path "build") { Remove-Item "build" -Recurse -Force -ErrorAction SilentlyContinue }
if (Test-Path "dist") { Remove-Item "dist" -Recurse -Force -ErrorAction SilentlyContinue }
if (Test-Path "SalesApp.spec") { Remove-Item "SalesApp.spec" -Force -ErrorAction SilentlyContinue }

# 4. Prepare PyInstaller arguments
$buildArgs = @(
    "--noconfirm",
    "--onefile",
    "--windowed",
    "--name", "SalesApp",
    "--clean",
    "--exclude-module", "PyQt5",
    "--exclude-module", "PySide6",
    "--exclude-module", "PySide2",
    "--exclude-module", "PyQt4",
    "--exclude-module", "PyQt6",
    "--exclude-module", "conda"
)

# Add assets
$assets = "SM 1 Bg.jpeg", "SM 2 Bg.jpeg", "SM 3 Bg.jpeg", "logo new.png", "Button 1.jpeg", "Button 2.jpeg", "Button 3.jpeg", "Button 4.jpeg", "Button 5.jpeg", "Button 6.jpeg", "Button 7.jpeg", "Button 8.jpeg", "Button 9.jpeg", "Button 10.jpeg", "Button 11.jpeg", "Button 12.jpeg"

foreach ($asset in $assets) {
    if (Test-Path $asset) {
        $buildArgs += "--add-data"
        $buildArgs += "$asset;."
    }
}

$buildArgs += "sales_app.py"

# 5. Run Build
Write-Host "Building SalesApp with PyInstaller (in clean env)..."
# Using call operator '&' avoids quoting issues with cmd /c
& .\env\Scripts\pyinstaller.exe $buildArgs

if ($LASTEXITCODE -eq 0) {
    Write-Host "`nBuild Successful! Executable is in dist/SalesApp.exe"
}
else {
    Write-Host "`nBuild Failed!"
}
