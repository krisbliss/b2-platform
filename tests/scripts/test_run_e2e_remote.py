"""Exercise the real shell wrapper with an offline harness stand-in."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


SCRIPTS = Path(__file__).resolve().parent


@pytest.fixture
def remote_workspace(tmp_path, monkeypatch):
    scripts = tmp_path / "tests" / "scripts"
    scripts.mkdir(parents=True)
    for name in ("run_e2e_remote.sh", "clean_results_to_markdown.py"):
        shutil.copyfile(SCRIPTS / name, scripts / name)
    bin_dir = tmp_path / ".venv-testplan" / "bin"
    bin_dir.mkdir(parents=True)
    (bin_dir / "python").symlink_to(sys.executable)
    uv = bin_dir / "uv"
    uv.write_text("#!/bin/sh\nexit 0\n")
    uv.chmod(0o755)
    harness = tmp_path / "scripts" / "e2e_cases.py"
    harness.parent.mkdir()
    harness.write_text('''
import argparse
import json
import os
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--base-url', required=True)
parser.add_argument('--out-dir')
parser.add_argument('--debug-tools', action='store_true')
parser.add_argument('--secret')
parser.add_argument('--country')
parser.add_argument('--kind')
args = parser.parse_args()
assert args.debug_tools
for flag in ('ALLOW_UNAUTHENTICATED_WEBHOOK', 'E2E_DISABLE_WHATSAPP_OUTBOUND', 'ENABLE_E2E_DEBUG'):
    assert flag not in os.environ, flag
Path('invocation.json').write_text(json.dumps(vars(args)))
if os.environ.get('WRITE_REPORT', '1') == '1':
    report = Path(args.out_dir) / '20261009T120000000000Z' / 'results.json'
    report.parent.mkdir(parents=True)
    report.write_text('invalid' if os.environ.get('BAD_REPORT') else '{"results": []}')
raise SystemExit(int(os.environ.get('HARNESS_STATUS', '0')))
''')
    for key in (
        "ALLOW_UNAUTHENTICATED_WEBHOOK", "E2E_DISABLE_WHATSAPP_OUTBOUND",
        "ENABLE_E2E_DEBUG", "E2E_OUT_DIR", "HARNESS_STATUS", "WRITE_REPORT", "BAD_REPORT",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("E2E_LOAD_ENV", "0")
    monkeypatch.setenv("BASE_URL", "https://adapter.example")
    monkeypatch.setenv("WEBHOOK_SECRET", "test-secret")
    return tmp_path


def run_remote(root, *args):
    return subprocess.run(
        ["bash", str(root / "tests/scripts/run_e2e_remote.sh"), *args],
        cwd=root.parent, capture_output=True, text=True, timeout=10,
    )


@pytest.mark.parametrize("status", [0, 7])
@pytest.mark.parametrize("output_mode", ["default", "env", "argument", "equals"])
def test_generates_report_and_preserves_status(remote_workspace, monkeypatch, status, output_mode):
    root = remote_workspace
    monkeypatch.setenv("HARNESS_STATUS", str(status))
    args = ["--country", "Indonesia", "--kind", "real"]
    output = root / "e2e_runs/remote"
    if output_mode == "env":
        output = root / "env reports"
        monkeypatch.setenv("E2E_OUT_DIR", str(output))
    elif output_mode == "argument":
        output = root / "relative reports"
        args += ["--out-dir", "relative reports"]
    elif output_mode == "equals":
        output = root / "absolute reports"
        args += [f"--out-dir={output}"]
    result = run_remote(root, *args)
    assert result.returncode == status, result.stdout + result.stderr
    assert len(list(output.glob("*/results_report.md"))) == 1
    invocation = json.loads((root / "invocation.json").read_text())
    assert invocation["base_url"] == "https://adapter.example"
    assert invocation["secret"] == "test-secret"
    assert invocation["country"] == "Indonesia"
    assert invocation["kind"] == "real"


@pytest.mark.parametrize("status", [0, 7])
def test_stale_report_is_not_converted(remote_workspace, monkeypatch, status):
    root = remote_workspace
    stale = root / "e2e_runs/remote/20260101T000000Z/results.json"
    stale.parent.mkdir(parents=True)
    stale.write_text('{"results": []}')
    monkeypatch.setenv("WRITE_REPORT", "0")
    monkeypatch.setenv("HARNESS_STATUS", str(status))
    result = run_remote(root)
    assert result.returncode == (status or 1)
    assert "did not produce a new results.json" in result.stderr
    assert not stale.with_name("results_report.md").exists()


@pytest.mark.parametrize("status", [0, 7])
def test_converter_failure_does_not_hide_harness_failure(remote_workspace, monkeypatch, status):
    monkeypatch.setenv("BAD_REPORT", "1")
    monkeypatch.setenv("HARNESS_STATUS", str(status))
    result = run_remote(remote_workspace)
    assert result.returncode == (status or 1)
    assert "invalid JSON" in result.stderr


def test_base_url_is_required(remote_workspace, monkeypatch):
    monkeypatch.delenv("BASE_URL")
    result = run_remote(remote_workspace)
    assert result.returncode == 1
    assert "set BASE_URL" in result.stderr
    assert not (remote_workspace / "invocation.json").exists()
