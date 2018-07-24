#!/bin/bash

# Script to set up repo for automatic cleanup of ipynb files
# as a git pre-commit hook.
# Run once in your repo as: ./setup_nb_cleanup.sh

# Check for dependencies
brew list >/dev/null 2>&1 || echo "Install homebrew first :-)"
jq -h >/dev/null 2>&1 || brew install jq

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

# Install nbconvert if needed
pip freeze | grep nbconvert >/dev/null 2>&1
if [ $? -ne 0 ] ; then
    pip install nbconvert
fi

# Set up py script export
if [ ! -e jupyter_notebook_config.py ] ; then
	cat > jupyter_notebook_config.py <<NB_TO_PY
# Source: http://jupyter-notebook.readthedocs.io/en/latest/extending/savehooks.html

import io
import os
from notebook.utils import to_api_path

_script_exporter = None

def script_post_save(model, os_path, contents_manager, **kwargs):
    """convert notebooks to Python script after save with nbconvert

    replaces ipython notebook --script
    """
    from nbconvert.exporters.script import ScriptExporter

    if model['type'] != 'notebook':
        return

    global _script_exporter

    if _script_exporter is None:
        _script_exporter = ScriptExporter(parent=contents_manager) 

    log = contents_manager.log

    base, ext = os.path.splitext(os_path)
    py_fname = base + '.py'
    script, resources = _script_exporter.from_filename(os_path)
    script_fname = base + resources.get('output_extension', '.txt')
    log.info("Saving script %s", to_api_path(script_fname, contents_manager.root_dir))

    with io.open(script_fname, 'w', encoding='utf-8') as f:
        f.write(script)

c.FileContentsManager.post_save_hook = script_post_save
NB_TO_PY

fi

exit 0

