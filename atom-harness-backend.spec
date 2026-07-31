from pathlib import Path

from PyInstaller.building.build_main import Analysis, COLLECT, EXE, PYZ


root = Path(SPECPATH)
data_files = [
    (str(root / "atom-language-model.json"), "."),
    (
        str(root / "causal_world_outputs" / "atom_causal_world_evidence.jsonl"),
        "causal_world_outputs",
    ),
    (
        str(root / "causal_world_outputs" / "atom_causal_world_model.json"),
        "causal_world_outputs",
    ),
    (
        str(root / "primitive_forge_outputs" / "atom_primitive_graph.json"),
        "primitive_forge_outputs",
    ),
    (
        str(
            root
            / "atom_causal_memory_rust"
            / "target"
            / "release"
            / "atom-causal-memory.exe"
        ),
        "rust",
    ),
]

analysis = Analysis(
    [str(root / "atom_harness_desktop_backend.py")],
    pathex=[str(root)],
    binaries=[],
    datas=data_files,
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["torch"],
    noarchive=False,
    optimize=1,
)
python_archive = PYZ(analysis.pure)
executable = EXE(
    python_archive,
    analysis.scripts,
    [],
    exclude_binaries=True,
    name="atom-harness-backend",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
)
bundle = COLLECT(
    executable,
    analysis.binaries,
    analysis.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="atom-harness-backend",
)
