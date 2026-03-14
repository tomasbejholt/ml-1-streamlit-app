#!/bin/bash
# Pull latest course updates from the upstream repo

git fetch upstream 2>/dev/null || {
    echo "No upstream remote found."
    echo "Run: git remote add upstream <UPSTREAM_REPO_URL>"
    exit 1
}
git merge upstream/main --no-edit
