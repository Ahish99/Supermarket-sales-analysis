
Write-Host "Cleaning up previous build artifacts..."
if (Test-Path "build") { Remove-Item "build" -Recurse -Force -ErrorAction SilentlyContinue }
if (Test-Path "dist") { Remove-Item "dist" -Recurse -Force -ErrorAction SilentlyContinue }
if (Test-Path "SalesApp.spec") { Remove-Item "SalesApp.spec" -Force -ErrorAction SilentlyContinue }

$assets = "SM 1 Bg.jpeg", "SM 2 Bg.jpeg", "SM 3 Bg.jpeg", "logo new.png", "Button 1.jpeg", "Button 2.jpeg", "Button 3.jpeg", "Button 4.jpeg", "Button 5.jpeg", "Button 6.jpeg", "Button 7.jpeg", "Button 8.jpeg", "Button 9.jpeg", "Button 10.jpeg", "Button 11.jpeg"

$data_args = @()
foreach ($asset in $assets) {
    if (Test-Path $asset) {
        $data_args += "--add-data"
        $data_args += "$asset;."
    }
}

Write-Host "Building SalesApp with PyInstaller..."
# Exclude Qt bindings AND conda to prevent conflicts/errors
$excludes = "--exclude-module PyQt5 --exclude-module PySide6 --exclude-module PySide2 --exclude-module PyQt4 --exclude-module PyQt6 --exclude-module conda"

$arg_str = "--noconfirm --onefile --windowed --name SalesApp --clean $excludes"
foreach ($asset in $assets) {
    if (Test-Path $asset) {
        $arg_str += " --add-data `"$asset;.`""
    }
}
$arg_str += " sales_app.py"

cmd /c "pyinstaller $arg_str"

if ($LASTEXITCODE -eq 0) {
    Write-Host "Build Successful! Executable is in dist/SalesApp.exe"
}
else {
    Write-Host "Build Failed!"
}
