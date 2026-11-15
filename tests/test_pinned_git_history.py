import os
import shutil
import subprocess

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
