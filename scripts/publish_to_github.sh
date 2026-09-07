#!/usr/bin/env bash
# Create the GitHub repository (private by default) and push this directory.
#
#   scripts/publish_to_github.sh                       # after `gh auth login`
#   GITHUB_TOKEN=ghp_xxx scripts/publish_to_github.sh  # or with a personal access token
#   scripts/publish_to_github.sh my-repo-name --public # custom name / visibility
#
# The token needs the "repo" scope (classic) or "Contents: read/write" +
# "Administration: read/write" (fine-grained) to create the repository.
set -euo pipefail
REPO="${1:-cocitation-prospective-concepts}"
VIS="${2:---private}"
export PATH="$HOME/.local/bin:$PATH"
cd "$(dirname "$0")/.."

if [ -n "${GITHUB_TOKEN:-}" ]; then export GH_TOKEN="$GITHUB_TOKEN"; fi
if ! gh auth status >/dev/null 2>&1; then
  echo "GitHub CLI is not authenticated. Run 'gh auth login' or set GITHUB_TOKEN." >&2
  exit 1
fi

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  git init -b main
  git add -A
  git commit -m "Initial release of data and code"
fi

gh repo create "$REPO" "$VIS" --source=. --remote=origin --push \
  --description "Data and code accompanying the manuscript: prospective technology concepts generated from predicted co-citation links"
echo
echo "Repository URL:"
gh repo view "$REPO" --json url -q .url
