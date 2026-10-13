"""Phase 6 verification tests for Wokwi ESP32 demonstration and C++ parity."""
import json
from pathlib import Path
import subprocess
import pytest


def test_demo_vectors_development_only():
    """Verify demo vectors originate strictly from development split, never test split."""
    meta_path = Path("esp32_demo/demo_vectors_metadata.json")
    header_path = Path("esp32_demo/src/demo_vectors.h")

    assert meta_path.exists(), "demo_vectors_metadata.json must exist"
    assert header_path.exists(), "demo_vectors.h must exist"

    with meta_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    vectors = data["vectors"]
    assert len(vectors) == 5

    # Check that each vector belongs strictly to development partition
    for v in vectors:
        assert v["split"] == "development"
        assert v["source_group_id"] == "g0001"
        assert v["source_group_id"] not in [
            "g0000", "g0004", "g0005", "g0009", "g0015",
            "g0020", "g0022", "g0024", "g0029", "g0032",
            "g0033", "g0034", "g0042", "g0046", "g0062",
            "g0067", "g0068", "g0070", "g0077", "g0078"
        ]  # No test group IDs


def test_wokwi_project_structure():
    """Verify all required Wokwi and PlatformIO configuration files exist."""
    required_files = [
        Path("esp32_demo/platformio.ini"),
        Path("esp32_demo/wokwi.toml"),
        Path("esp32_demo/diagram.json"),
        Path("esp32_demo/src/main.cpp"),
        Path("esp32_demo/src/adaptive_core.h"),
        Path("esp32_demo/src/demo_vectors.h"),
        Path("esp32_demo/README.md"),
    ]

    for p in required_files:
        assert p.exists(), f"Required file missing: {p}"


def test_cpp_test_harness_compilation_and_parity():
    """Compile and run the standalone C++ test harness to verify Python/C++ parity."""
    harness_src = Path("esp32_demo/test_harness.cpp")
    harness_exe = Path("esp32_demo/test_harness.exe")

    # Compile using g++
    compile_cmd = [
        "C:\\MinGW\\bin\\g++.exe",
        "-std=c++11",
        "-O2",
        "-I", "esp32_demo",
        str(harness_src),
        "-o", str(harness_exe),
    ]

    try:
        comp_res = subprocess.run(compile_cmd, capture_output=True, text=True)
        assert comp_res.returncode == 0, f"Compilation failed: {comp_res.stderr}"
        assert harness_exe.exists(), "test_harness.exe was not created"

        # Execute harness and assert exit code 0
        run_res = subprocess.run([str(harness_exe)], capture_output=True, text=True)
        assert run_res.returncode == 0, f"Execution failed:\n{run_res.stdout}\n{run_res.stderr}"
        assert "100% Python/C++ Parity Achieved" in run_res.stdout
    finally:
        if harness_exe.exists():
            harness_exe.unlink()
