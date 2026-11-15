import os
import shutil
import subprocess

import pytest

from nexus_open_swe_runtime import cli


def test_pinned_git_history_ignores_replacements_and_grafts(tmp_path):
    git = shutil.which("git")
    assert git is not None
    repo = tmp_path / "repo"
    repo.mkdir()

    def run(*args, env=None, check=True, input=None):
        return subprocess.run(
            [git, "-C", str(repo), *args],
            check=check,
            capture_output=True,
            text=True,
            env=env,
            input=input,
        )

    run("init", "-q")
    run("config", "user.name", "Test User")
    run("config", "user.email", "test@example.com")
    blob = run("hash-object", "-w", "--stdin", input="content")
    tree = run("mktree", input=f"100644 blob {blob.stdout.strip()}\tfile\n")
    orphan = run("commit-tree", tree.stdout.strip(), input="orphan\n").stdout.strip()
    parent = run("commit-tree", tree.stdout.strip(), input="parent\n").stdout.strip()
    run("replace", "--graft", orphan, parent)
    graft_file = repo / ".git" / "info" / "grafts"
    graft_file.write_text(f"{orphan} {parent}\n", encoding="utf-8")

    assert run("rev-parse", "--verify", "--quiet", f"{orphan}^").returncode == 0
    caller_env = dict(os.environ)
    caller_env["GIT_NO_REPLACE_OBJECTS"] = "0"
    caller_env["GIT_GRAFT_FILE"] = str(graft_file)
    pinned_env = cli._pinned_git_env(caller_env)
    assert pinned_env["GIT_NO_REPLACE_OBJECTS"] == "1"
    assert pinned_env["GIT_GRAFT_FILE"] == os.devnull
    result = subprocess.run(
        [
            git,
            *cli._PINNED_GIT_ARGS,
            "-C",
            str(repo),
            "rev-parse",
            "--verify",
            "--quiet",
            f"{orphan}^",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=pinned_env,
    )
    assert result.returncode == 1


def test_git_call_sites_ignore_replacements_and_grafts(tmp_path, monkeypatch):
    git = shutil.which("git")
    assert git is not None

    def make_repo(path):
        path.mkdir()

        def run(*args, input=None):
            return subprocess.run(
                [git, "-C", str(path), *args], capture_output=True, text=True,
                check=True, input=input,
            )

        run("init", "-q")
        run("config", "user.name", "Test User")
        run("config", "user.email", "test@example.com")
        blob = run("hash-object", "-w", "--stdin", input="content")
        tree = run("mktree", input=f"100644 blob {blob.stdout.strip()}\tfile\n")
        orphan = run("commit-tree", tree.stdout.strip(), input="orphan\n").stdout.strip()
        parent = run("commit-tree", tree.stdout.strip(), input="parent\n").stdout.strip()
        return run, orphan, parent

    repo = tmp_path / "hostile"
    run, orphan, parent = make_repo(repo)
    run("config", "core.useReplaceRefs", "true")
    run("config", "core.commitGraph", "true")
    run("replace", "--graft", orphan, parent)
    graft_file = repo / ".git" / "info" / "grafts"
    graft_file.write_text(f"{orphan} {parent}\n", encoding="utf-8")
    monkeypatch.setenv("GIT_NO_REPLACE_OBJECTS", "0")
    monkeypatch.setenv("GIT_GRAFT_FILE", str(graft_file))

    with pytest.raises(cli.RuntimeErrorBounded):
        cli._git_output(repo, "rev-parse", "--verify", "--quiet", f"{orphan}^")
    with pytest.raises(cli.RuntimeErrorBounded):
        cli._git_temp_output(
            repo,
            ["rev-parse", "--verify", "--quiet", f"{orphan}^"],
            {"GIT_NO_REPLACE_OBJECTS": "0", "GIT_GRAFT_FILE": str(graft_file)},
        )

    clean = tmp_path / "clean"
    _, clean_orphan, _ = make_repo(clean)
    assert cli._git_output(clean, "rev-parse", "--verify", clean_orphan) == clean_orphan
    assert cli._git_temp_output(clean, ["rev-parse", "--verify", clean_orphan], {}) == clean_orphan


def test_git_call_sites_pass_pinned_argv_and_env(tmp_path, monkeypatch):
    calls = []

    class Result:
        stdout = "100644 blob " + "a" * 40 + "\tfile\n"

    def capture(argv, **kwargs):
        calls.append((argv, kwargs.get("env")))
        if "ls-tree" in argv:
            return Result()
        if "cat-file" in argv:
            result = Result()
            result.stdout = b"content"
            return result
        result = Result()
        result.stdout = "ok\n"
        return result

    monkeypatch.setattr(cli.subprocess, "run", capture)
    monkeypatch.setenv("GIT_NO_REPLACE_OBJECTS", "0")
    monkeypatch.setenv("GIT_GRAFT_FILE", "/tmp/hostile-grafts")
    monkeypatch.setenv("GIT_NO_LAZY_FETCH", "0")
    cli._git_output(tmp_path, "status")
    cli._git_temp_output(
        tmp_path, ["status"],
        {
            "GIT_NO_REPLACE_OBJECTS": "0",
            "GIT_GRAFT_FILE": "/tmp/hostile-grafts",
            "GIT_NO_LAZY_FETCH": "0",
        },
    )
    cli._git_blob_sha256(tmp_path, "a" * 40, "file")

    git_calls = [call for call in calls if call[0][0] == shutil.which("git")]
    assert len(git_calls) == 4
    for argv, env in git_calls:
        assert argv[1:9] == [
            "-c", "core.useReplaceRefs=false",
            "-c", "core.commitGraph=false",
            "-c", "core.fsmonitor=false",
            "-c", "core.hooksPath=/dev/null",
        ]
        assert argv[9] == "-C"
        assert env["GIT_NO_REPLACE_OBJECTS"] == "1"
        assert env["GIT_GRAFT_FILE"] == os.devnull
        assert env["GIT_NO_LAZY_FETCH"] == "1"


def test_git_wrappers_ignore_hostile_repo_local_fsmonitor(tmp_path, monkeypatch):
    git = shutil.which("git")
    assert git is not None
    repo = tmp_path / "hostile-fsmonitor"
    repo.mkdir()
    subprocess.run([git, "-C", str(repo), "init", "-q"], check=True)
    marker = tmp_path / "fsmonitor-ran"
    monitor = tmp_path / "fsmonitor"
    monitor.write_text(
        "#!/bin/sh\nprintf ran > \"$FSmonitor_MARKER\"\nprintf '\\n'\n",
        encoding="utf-8",
    )
    monitor.chmod(0o755)
    hooks = tmp_path / "hooks"
    hooks.mkdir()
    hook = hooks / "pre-commit"
    hook.write_text(
        "#!/bin/sh\nprintf ran > \"$HOOK_MARKER\"\n",
        encoding="utf-8",
    )
    hook.chmod(0o755)
    subprocess.run(
        [git, "-C", str(repo), "config", "core.fsmonitor", str(monitor)], check=True
    )
    subprocess.run(
        [git, "-C", str(repo), "config", "core.hooksPath", str(hooks)], check=True
    )

    monkeypatch.setenv("FSmonitor_MARKER", str(marker))
    cli._git_output(repo, "status")

    assert not marker.exists()
