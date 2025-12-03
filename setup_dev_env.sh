#!/usr/bin/env bash

# Go to repo root to avoid madness
cd "$(git rev-parse --show-toplevel)"

# Ensure pre-commit exists and is executable; then just make sure it runs the script we have in our repo
# We also run setup_hooks in case we want to add any later hooks (that way any working hook will install
# the others)
touch .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit
echo "./setup_dev_env.sh && ./hooks/pre-commit || exit 1" > .git/hooks/pre-commit

# Same thing for pre-push
touch .git/hooks/pre-push
chmod +x .git/hooks/pre-push
echo "./setup_dev_env.sh && ./hooks/pre-push || exit 1" > .git/hooks/pre-push
