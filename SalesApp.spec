# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['sales_app.py'],
    pathex=[],
    binaries=[],
    datas=[('SM 1 Bg.jpeg', '.'), ('SM 2 Bg.jpeg', '.'), ('SM 3 Bg.jpeg', '.'), ('logo new.png', '.'), ('Button 1.jpeg', '.'), ('Button 2.jpeg', '.'), ('Button 3.jpeg', '.'), ('Button 4.jpeg', '.'), ('Button 5.jpeg', '.'), ('Button 6.jpeg', '.'), ('Button 7.jpeg', '.'), ('Button 8.jpeg', '.'), ('Button 9.jpeg', '.'), ('Button 10.jpeg', '.'), ('Button 11.jpeg', '.'), ('Button 12.jpeg', '.')],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['PyQt5', 'PySide6', 'PySide2', 'PyQt4', 'PyQt6', 'conda'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='SalesApp',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
