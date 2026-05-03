"""
Registry loading and model path resolution utilities.

Design
------
model_registry.yaml stores paths *relative to model_root*.
At runtime model_root is overridable from any page's sidebar; the value is
shared across all pages via st.session_state["model_root"].

Key change vs. absolute-path era
---------------------------------
`onnx` is no longer stored in the registry.  Instead each model entry has
`model_dir` (directory under model_root).  The sidebar shows a selectbox of
all .onnx files found in that directory so the user can pick the exact file.
This handles model versioning / filename changes without touching the registry.

Resolution rules
----------------
  model_dir, label, param_files entries  →  joined with model_root (if relative)
  color_map                               →  joined with repo root  (if relative)
  Any absolute path                       →  used as-is
"""

import glob
import os

import streamlit as st
import yaml

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REGISTRY_PATH = os.path.join(_REPO_ROOT, "config", "model_registry.yaml")

# Keys resolved against model_root
_MODEL_ROOT_KEYS: frozenset[str] = frozenset({"model_dir", "label"})
# Keys resolved against repo root
_REPO_ROOT_KEYS: frozenset[str] = frozenset({"color_map"})


@st.cache_data
def load_registry() -> dict:
    with open(_REGISTRY_PATH) as f:
        return yaml.safe_load(f)


def get_model_root(registry: dict) -> str:
    return st.session_state.get("model_root", registry.get("model_root", ""))


def resolve_model_config(cfg: dict, model_root: str) -> dict:
    """
    Return a copy of cfg with all paths resolved to absolute strings.

    model_dir / label / param_files  → os.path.join(model_root, path)
    color_map                         → os.path.join(repo_root, path)
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


# ---------------------------------------------------------------------------
# Sidebar widgets
# ---------------------------------------------------------------------------

def render_model_root_sidebar(registry: dict) -> str:
    """
    Render the shared Model Root input at the top of the sidebar.
    Uses key="model_root" so st.session_state["model_root"] persists
    across page navigations automatically.
    """
    st.sidebar.text_input(
        "Model Root",
        value=registry.get("model_root", ""),
        key="model_root",
        help=(
            "Root directory containing all model subdirectories.\n"
            "Relative paths in model_registry.yaml are resolved against this."
        ),
    )
    return st.session_state["model_root"]


def render_onnx_selector(model_dir: str, key: str) -> str | None:
    """
    Scan model_dir for .onnx files and render a selectbox in the sidebar.

    - Directory missing or empty  → warning, returns None
    - Exactly one file            → shown as caption (no selectbox noise)
    - Multiple files              → selectbox for the user to choose
    """
    if not os.path.isdir(model_dir):
        st.sidebar.warning(f"Directory not found:\n`{model_dir}`")
        return None

    onnx_files = sorted(glob.glob(os.path.join(model_dir, "*.onnx")))
    if not onnx_files:
        st.sidebar.warning(f"No .onnx files found in:\n`{model_dir}`")
        return None

    if len(onnx_files) == 1:
        st.sidebar.caption(f"Model: `{os.path.basename(onnx_files[0])}`")
        return onnx_files[0]

    return st.sidebar.selectbox(
        "ONNX Model",
        onnx_files,
        format_func=os.path.basename,
        key=key,
    )


def render_resolved_paths_expander(
    resolved_cfg: dict,
    selected_onnx: str | None = None,
) -> None:
    """
    Show resolved file paths in a collapsed expander with ✅/❌ existence check.
    Pass selected_onnx separately since it comes from render_onnx_selector.
    """
    with st.sidebar.expander("Resolved paths", expanded=False):
        if selected_onnx:
            _show_path("onnx", selected_onnx)
        if "model_dir" in resolved_cfg:
            _show_path("model_dir", resolved_cfg["model_dir"], is_dir=True)
        for key in ("label", "color_map"):
            if key in resolved_cfg:
                _show_path(key, resolved_cfg[key])
        for name, path in resolved_cfg.get("param_files", {}).items():
            _show_path(name, path)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _join(base: str, path: str) -> str:
    if not path or os.path.isabs(path):
        return path
    return os.path.join(base, path)


def _show_path(label: str, path: str, is_dir: bool = False) -> None:
    exists = os.path.isdir(path) if is_dir else os.path.isfile(path)
    icon = "✅" if exists else "❌"
    st.markdown(f"**{label}**  \n`{path}` {icon}")
