"""
Registry loading and model path resolution utilities.

Design
------
model_registry.yaml holds paths *relative to model_root*.  At runtime the
model_root can be overridden from any page's sidebar; the value is shared
across all pages via st.session_state["model_root"].

Resolution rules
----------------
  onnx, label, param_files entries  →  joined with model_root   (when relative)
  color_map                          →  joined with repo root     (when relative)
  Any absolute path                  →  used as-is

Sidebar widget
--------------
Call render_model_root_sidebar(registry) once per page.  It renders the
"Model Root" input and an optional expander that shows per-file resolved
paths (read-only) so the user can verify what will be loaded.
"""

import os

import streamlit as st
import yaml

# Paths resolved relative to the repository root (not model_root)
_REPO_ROOT_KEYS: frozenset[str] = frozenset({"color_map"})
# Paths resolved relative to model_root
_MODEL_ROOT_KEYS: frozenset[str] = frozenset({"onnx", "label"})

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REGISTRY_PATH = os.path.join(_REPO_ROOT, "config", "model_registry.yaml")


@st.cache_data
def load_registry() -> dict:
    with open(_REGISTRY_PATH) as f:
        return yaml.safe_load(f)


def get_model_root(registry: dict) -> str:
    """Return the current model root (session_state > registry default)."""
    return st.session_state.get("model_root", registry.get("model_root", ""))


def resolve_model_config(cfg: dict, model_root: str) -> dict:
    """
    Return a copy of cfg with all paths resolved to absolute strings.

    onnx / label / param_files values  → os.path.join(model_root, path)
    color_map                           → os.path.join(repo_root, path)
    """
    resolved = {**cfg}

    for key in _MODEL_ROOT_KEYS:
        if key in resolved:
            resolved[key] = _join(model_root, resolved[key])

    if "param_files" in resolved:
        resolved["param_files"] = {
            k: _join(model_root, v)
            for k, v in resolved["param_files"].items()
        }

    for key in _REPO_ROOT_KEYS:
        if key in resolved:
            resolved[key] = _join(_REPO_ROOT, resolved[key])

    return resolved


def render_model_root_sidebar(registry: dict) -> str:
    """
    Render the shared Model Root input at the top of the sidebar.

    The widget key is "model_root" so st.session_state["model_root"] holds
    the value and persists across page navigations automatically.

    Returns the current model root string.
    """
    st.sidebar.text_input(
        "Model Root",
        value=registry.get("model_root", ""),
        key="model_root",
        help=(
            "Root directory that contains all model subdirectories.\n"
            "Relative paths in model_registry.yaml are resolved against this."
        ),
    )
    return st.session_state["model_root"]


def render_resolved_paths_expander(resolved_cfg: dict) -> None:
    """
    Show resolved file paths in a collapsed expander so users can verify
    what will actually be loaded without cluttering the normal sidebar view.
    """
    path_keys = [k for k in ("onnx", "label", "color_map") if k in resolved_cfg]
    param_files = resolved_cfg.get("param_files", {})

    if not path_keys and not param_files:
        return

    with st.sidebar.expander("Resolved paths", expanded=False):
        for key in path_keys:
            path = resolved_cfg[key]
            exists = os.path.exists(path)
            icon = "✅" if exists else "❌"
            st.markdown(f"**{key}**  \n`{path}` {icon}")
        for name, path in param_files.items():
            exists = os.path.exists(path)
            icon = "✅" if exists else "❌"
            st.markdown(f"**{name}**  \n`{path}` {icon}")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _join(base: str, path: str) -> str:
    """Join base + path only when path is relative; otherwise return path as-is."""
    if not path or os.path.isabs(path):
        return path
    return os.path.join(base, path)
