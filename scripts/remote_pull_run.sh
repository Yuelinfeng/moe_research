#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: remote_pull_run.sh <config.env>" >&2
  exit 2
fi

CONFIG_ENV="$1"
# The local orchestrator uploads a shell-quoted env file. Keeping the remote
# bootstrap pure Bash avoids requiring Python before the conda environment exists.
source "$CONFIG_ENV"

: "${REPO_PATH:?missing REPO_PATH}"
: "${RUNS_ROOT:?missing RUNS_ROOT}"
: "${RUN_COMMAND:?missing RUN_COMMAND}"

GIT_REMOTE="${GIT_REMOTE:-origin}"
BRANCH="${BRANCH:-main}"
RUN_LABEL="${RUN_LABEL:-moe_run}"
SETUP_COMMAND="${SETUP_COMMAND:-}"
EXPECTED_COMMIT="${EXPECTED_COMMIT:-}"
REPO_URL="${REPO_URL:-}"
FALLBACK_BUNDLE="${FALLBACK_BUNDLE:-}"

fetch_target_ref() {
  if git -c http.version=HTTP/1.1 fetch "$GIT_REMOTE" "$BRANCH"; then
    return 0
  fi

  if [[ -n "$FALLBACK_BUNDLE" && -f "$FALLBACK_BUNDLE" ]]; then
    echo "git fetch from $GIT_REMOTE/$BRANCH failed; using uploaded bundle fallback" >&2
    git fetch "$FALLBACK_BUNDLE" HEAD
    return 0
  fi

  return 1
}

if [[ ! -d "$REPO_PATH/.git" ]]; then
  if [[ -n "$REPO_URL" ]]; then
    if [[ -e "$REPO_PATH" ]]; then
      BACKUP_PATH="${REPO_PATH}.invalid_$(date +%Y%m%d_%H%M%S)"
      echo "remote repo path exists but is not a git repository; moving it to $BACKUP_PATH" >&2
      mv "$REPO_PATH" "$BACKUP_PATH"
    fi
    mkdir -p "$(dirname "$REPO_PATH")"
    if ! git -c http.version=HTTP/1.1 clone "$REPO_URL" "$REPO_PATH"; then
      if [[ -n "$FALLBACK_BUNDLE" && -f "$FALLBACK_BUNDLE" ]]; then
        echo "git clone failed; initializing repository from uploaded bundle fallback" >&2
        mkdir -p "$REPO_PATH"
        cd "$REPO_PATH"
        git init
        git remote add "$GIT_REMOTE" "$REPO_URL" || true
        git fetch "$FALLBACK_BUNDLE" HEAD
        git checkout -B "$BRANCH" FETCH_HEAD
      else
        exit 10
      fi
    fi
  else
    echo "remote repo path is not a git repository: $REPO_PATH" >&2
    echo "set remote.repoUrl to let the runner clone it automatically" >&2
    exit 10
  fi
fi

cd "$REPO_PATH"

if [[ -n "$(git status --porcelain)" ]]; then
  echo "remote repository has local changes; aborting before git pull" >&2
  git status --short >&2
  exit 20
fi

fetch_target_ref
if git rev-parse --verify "$BRANCH" >/dev/null 2>&1; then
  git checkout "$BRANCH"
else
  git checkout -B "$BRANCH" FETCH_HEAD
fi
git merge --ff-only FETCH_HEAD

COMMIT="$(git rev-parse HEAD)"
SHORT_COMMIT="$(git rev-parse --short HEAD)"

if [[ -n "$EXPECTED_COMMIT" && "$COMMIT" != "$EXPECTED_COMMIT" ]]; then
  echo "remote commit mismatch after pull" >&2
  echo "expected: $EXPECTED_COMMIT" >&2
  echo "actual:   $COMMIT" >&2
  exit 30
fi

SAFE_LABEL="$(printf '%s' "$RUN_LABEL" | tr -c 'A-Za-z0-9_.-' '_')"
STARTED_AT="$(date -Is)"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ID="${SAFE_LABEL}_${SHORT_COMMIT}_${STAMP}"
RUN_DIR="${RUNS_ROOT%/}/$RUN_ID"

mkdir -p "$RUN_DIR"

echo "AIRS_RUN_ID=$RUN_ID"
echo "AIRS_RUN_DIR=$RUN_DIR"

git status --short > "$RUN_DIR/git_status_before.txt"
git log -1 --format=fuller > "$RUN_DIR/commit.txt"
printf '#!/usr/bin/env bash\nset -euo pipefail\n%s\n' "$RUN_COMMAND" > "$RUN_DIR/run_command.sh"
chmod +x "$RUN_DIR/run_command.sh"

write_manifest() {
  local exit_code="$1"
  local finished_at="$2"

  cat > "$RUN_DIR/manifest.env" <<EOF
run_id=$RUN_ID
run_dir=$RUN_DIR
repo_path=$REPO_PATH
branch=$BRANCH
git_remote=$GIT_REMOTE
commit=$COMMIT
short_commit=$SHORT_COMMIT
run_label=$RUN_LABEL
started_at=$STARTED_AT
finished_at=$finished_at
exit_code=$exit_code
EOF

  if command -v python3 >/dev/null 2>&1 || command -v python >/dev/null 2>&1; then
    local python_bin
    python_bin="$(command -v python3 || command -v python)"
    "$python_bin" - "$RUN_DIR/manifest.json" <<PY
import json
import os
import sys

payload = {
    "run_id": os.environ["RUN_ID"],
    "run_dir": os.environ["RUN_DIR"],
    "repo_path": os.environ["REPO_PATH"],
    "branch": os.environ["BRANCH"],
    "git_remote": os.environ["GIT_REMOTE"],
    "commit": os.environ["COMMIT"],
    "short_commit": os.environ["SHORT_COMMIT"],
    "run_label": os.environ["RUN_LABEL"],
    "run_command": os.environ["RUN_COMMAND"],
    "started_at": os.environ["STARTED_AT"],
    "finished_at": "$finished_at",
    "exit_code": int("$exit_code"),
}

with open(sys.argv[1], "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2)
    f.write("\\n")
PY
  fi
}

export RUN_ID RUN_DIR COMMIT SHORT_COMMIT REPO_PATH BRANCH GIT_REMOTE RUN_LABEL RUN_COMMAND STARTED_AT

set +e
(
  cd "$REPO_PATH"
  export RUN_ID RUN_DIR COMMIT SHORT_COMMIT REPO_PATH
  if [[ -n "$SETUP_COMMAND" ]]; then
    eval "$SETUP_COMMAND"
  fi
  eval "$RUN_COMMAND"
) > >(tee "$RUN_DIR/stdout.log") 2> >(tee "$RUN_DIR/stderr.log" >&2)
STATUS=$?
set -e

printf '%s\n' "$STATUS" > "$RUN_DIR/exit_code.txt"
write_manifest "$STATUS" "$(date -Is)"

exit "$STATUS"
