import random
from typing import Sequence

# First 8 entries match common Autoware label order (UNKNOWN, CAR, TRUCK, …)
_DEFAULT_COLORS: dict[int, tuple[int, int, int]] = {
    0: (160, 160, 160),
    1: (0, 200, 0),
    2: (0, 80, 255),
    3: (255, 80, 0),
    4: (0, 220, 220),
    5: (200, 0, 200),
    6: (220, 220, 0),
    7: (128, 128, 255),
}


def class_color_map(n_classes: int, seed: int = 42) -> dict[int, tuple[int, int, int]]:
    """
    Return a stable color dict for n_classes class IDs.
    The first 8 IDs use hand-picked colors; the rest are randomly generated.
    """
    rng = random.Random(seed)
    colors: dict[int, tuple[int, int, int]] = {}
    for i in range(n_classes):
        if i in _DEFAULT_COLORS:
            colors[i] = _DEFAULT_COLORS[i]
        else:
            colors[i] = tuple(rng.randint(50, 230) for _ in range(3))
    return colors


def sidebar_class_legend(
    st,
    class_names: Sequence[str],
    colors: dict[int, tuple[int, int, int]],
) -> None:
    """Render a colored class legend in a Streamlit sidebar."""
    st.sidebar.markdown("### Class Legend")
    for i, name in enumerate(class_names):
        r, g, b = colors.get(i, (200, 200, 200))
        st.sidebar.markdown(
            f"<span style='color:rgb({r},{g},{b})'>■</span> {i}: {name}",
            unsafe_allow_html=True,
        )
