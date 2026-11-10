"""Git metadata must not prevent experiments when tracked text uses other encodings."""
import json
import subprocess
import sys

import pytest

from avm_bringup.experiments import runtime_manifest


def test_runtime_manifest_with_gbk_git_diff(tmp_path):
    def git(*args):
        return subprocess.run(["git", *args], cwd=tmp_path, capture_output=True, check=True)

    git("init")
    document = tmp_path / "dev_log.md"
    document.write_bytes("将标定算法加到系统\n".encode("gbk"))
    git("add", "dev_log.md")
    git("-c", "user.name=Regression Test", "-c", "user.email=test@example.invalid",
        "-c", "commit.gpgsign=false", "commit", "-m", "Initial log")
    document.write_bytes(document.read_bytes() + "将企业的场景加到系统\n".encode("gbk")
                         + "UTF-8 中文\n".encode("utf-8"))
    raw_diff = git("diff", "--no-ext-diff", "--binary").stdout
    with pytest.raises(UnicodeDecodeError):
        raw_diff.decode("utf-8")

    manifest = runtime_manifest(tmp_path)

    assert manifest["git_commit"].strip() == git("rev-parse", "HEAD").stdout.decode().strip()
    assert "dev_log.md" in manifest["git_status"]
    assert manifest["git_diff"] == raw_diff.decode("utf-8", errors="backslashreplace")
    assert r"\xbd" in manifest["git_diff"]
    assert "UTF-8 中文" in manifest["git_diff"]
    assert json.loads(json.dumps(manifest))["git_diff"] == manifest["git_diff"]


@pytest.mark.parametrize("returncode", [0, 1])
def test_runtime_manifest_with_non_utf8_stderr(tmp_path, monkeypatch, returncode):
    run = subprocess.run

    def git_with_non_utf8_stderr(args, **kwargs):
        return run([sys.executable, "-c",
                    "import sys; sys.stdout.buffer.write(b'valid output\\n'); "
                    "sys.stderr.buffer.write(bytes([0xbd])); sys.exit(%d)" % returncode],
                   **kwargs)

    monkeypatch.setattr("avm_bringup.experiments.subprocess.run", git_with_non_utf8_stderr)
    manifest = runtime_manifest(tmp_path)
    for key in ("git_commit", "git_status", "git_diff"):
        assert manifest[key] == ("valid output\n" if returncode == 0 else None)
