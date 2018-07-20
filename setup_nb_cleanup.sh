#!/bin/bash

# Script to set up repo for automatic cleanup of ipynb files
# as a git pre-commit hook.
# Run once in your repo as: ./setup_nb_cleanup.sh

# Check for dependencies
brew list >/dev/null 2>&1 || echo "Install homebrew first :-)"
jq >/dev/null 2>&1 || brew install jq

# Set up git config
grep nbstrip_full ~/.gitconfig >/dev/null 2>&1 || cat >> ~/.gitconfig <<CONFIG
[filter "nbstrip_full"]
clean = "jq --indent 1 \
        '(.cells[] | select(has(\"outputs\")) | .outputs) = []  \
        | (.cells[] | select(has(\"execution_count\")) | .execution_count) = null  \
        | .metadata = {\"language_info\": {\"name\": \"python\", \"pygments_lexer\": \"ipython3\"}} \
        | .cells[].metadata = {} \
        '"
smudge = cat
required = true
CONFIG

# Set up gitattributes
grep nbstrip_full .gitattributes >/dev/null 2>&1 || echo "*.ipynb filter=nbstrip_full" >> .gitattributes

# Set up bash alias
grep nbstrip_jq ~/.bashrc >/dev/null 2>&1 || cat >> ~/.bashrc <<ALIAS
alias nbstrip_jq="jq --indent 1 \
    '(.cells[] | select(has(\"outputs\")) | .outputs) = []  \
    | (.cells[] | select(has(\"execution_count\")) | .execution_count) = null  \
    | .metadata = {\"language_info\": {\"name\": \"python\", \"pygments_lexer\": \"ipython3\"}} \
    | .cells[].metadata = {} \
    '"
ALIAS

exit 0

