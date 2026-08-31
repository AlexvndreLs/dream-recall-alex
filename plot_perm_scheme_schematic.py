"""Schematic diagram of the two permutation schemes (Fixed-effects vs Random-effects).

A purely illustrative figure (no external data read) intended for a presentation slide:
it demonstrates side-by-side the mechanics of the two permutation strategies on a small
toy dataset containing 4 subjects with 3 epochs each.

    Column 1 (Original)        : Each subject has a single group label color (HR / LR);
                                 all of its epochs carry the exact same color.
    Column 2 (Epoch-level, FFX): Labels are shuffled epoch by epoch, breaking the subject
                                 blocks apart. A single subject can end up with a mix of
                                 HR and LR epochs.
    Column 3 (Subject-level, RFX): Labels are shuffled at the subject block level. Each
                                 subject block remains homogeneous; only the block's overall
                                 label is reassigned.

Usage:
    python3 plot_perm_scheme_schematic.py

Requires no external data dependencies and runs in any Python environment.
"""

import argparse
import os
import sys
from typing import List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Colors (HR = High Recaller, LR = Low Recaller)
COLOR_HR: str = "#c0392b"  # Red
COLOR_LR: str = "#2c7fb8"  # Blue

NUMBER_OF_SUBJECTS: int = 4
NUMBER_OF_EPOCHS: int = 3

# Original state: S1, S2 = HR; S3, S4 = LR
ORIGINAL_LABELS: List[str] = ["HR", "HR", "LR", "LR"]

# A sample epoch-level permutation (FFX): all 12 labels are shuffled without
# regard to subject boundaries. The total count (6 HR, 6 LR) is preserved but dispersed.
# Rows = subjects, Columns = epochs.
FFX_LABELS: List[List[str]] = [
    ["HR", "LR", "HR"],
    ["LR", "LR", "HR"],
    ["HR", "LR", "LR"],
    ["HR", "LR", "HR"],
]

# A sample subject-level permutation (RFX): block labels are permuted as whole units.
# Here S1 -> LR, S2 -> HR, S3 -> HR, S4 -> LR (block homogeneity is strictly preserved).
RFX_LABELS: List[str] = ["LR", "HR", "HR", "LR"]


def get_label_color(label: str) -> str:
    """Returns the hex color corresponding to a given experimental group label.

    Args:
        label (str): The subject or epoch group label ("HR" or "LR").

    Returns:
        str: Hex color string.
    """
    return COLOR_HR if label == "HR" else COLOR_LR


def draw_column(
    ax: plt.Axes, x_offset: float, labels_per_epoch: List[List[str]], title: str
) -> None:
    """Draws a single schematic column representing subject blocks and epoch cells.

    Args:
        ax (plt.Axes): The Matplotlib Axes object on which to render elements.
        x_offset (float): The horizontal position (x-coordinate) for the column.
        labels_per_epoch (List[List[str]]): A nested list where each element represents
            a subject's list of epoch labels.
        title (str): The title displayed above the column.

    Raises:
        ValueError: If labels_per_epoch does not match the expected dimensions.
    """
    if len(labels_per_epoch) != NUMBER_OF_SUBJECTS:
        raise ValueError(
            f"Expected data for {NUMBER_OF_SUBJECTS} subjects, "
            f"got {len(labels_per_epoch)}."
        )

    cell_width: float = 0.9
    cell_height: float = 0.7
    gap_between_subjects: float = 0.35  # Vertical spacing between subject blocks

    for subject_idx in range(NUMBER_OF_SUBJECTS):
        if len(labels_per_epoch[subject_idx]) != NUMBER_OF_EPOCHS:
            raise ValueError(
                f"Subject {subject_idx + 1} must contain exactly {NUMBER_OF_EPOCHS} epochs."
            )

        y_base: float = (NUMBER_OF_SUBJECTS - 1 - subject_idx) * (
            NUMBER_OF_EPOCHS * cell_height + gap_between_subjects
        )

        for epoch_idx in range(NUMBER_OF_EPOCHS):
            y_pos: float = y_base + (NUMBER_OF_EPOCHS - 1 - epoch_idx) * cell_height
            current_label: str = labels_per_epoch[subject_idx][epoch_idx]

            ax.add_patch(
                Rectangle(
                    (x_offset, y_pos),
                    cell_width,
                    cell_height * 0.9,
                    facecolor=get_label_color(current_label),
                    edgecolor="white",
                    linewidth=1.5,
                )
            )

        # Subject label positioned to the left of each subject block
        ax.text(
            x_offset - 0.15,
            y_base + (NUMBER_OF_EPOCHS * cell_height) / 2 - cell_height * 0.05,
            f"S{subject_idx + 1}",
            ha="right",
            va="center",
            fontsize=11,
            fontweight="bold",
        )

    # Column title centered directly above the top block
    top_y: float = NUMBER_OF_SUBJECTS * (NUMBER_OF_EPOCHS * 0.7 + 0.35) - 0.15
    ax.text(
        x_offset + 0.45,
        top_y + 0.55,
        title,
        ha="center",
        va="bottom",
        fontsize=12,
        fontweight="bold",
    )


def measure_content_extent(fig, ax) -> tuple:
    """Measures the bounding box of every drawn element, in data coordinates.

    Text extents are only known once the figure has been rendered, and they are
    expressed in pixels; they are converted back to data units here. The legend
    is deliberately ignored: it is anchored in axes coordinates and must not
    influence the framing of the content.

    Args:
        fig: The Matplotlib Figure being rendered.
        ax (plt.Axes): The Axes holding the schematic.

    Returns:
        tuple: (x_min, x_max, y_min, y_max) in data coordinates.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    to_data = ax.transData.inverted()

    xs: List[float] = []
    ys: List[float] = []

    for artist in list(ax.patches) + list(ax.texts):
        box = artist.get_window_extent(renderer)
        corners = to_data.transform(
            [[box.x0, box.y0], [box.x1, box.y1]]
        )
        xs.extend([corners[0][0], corners[1][0]])
        ys.extend([corners[0][1], corners[1][1]])

    if not xs:
        raise RuntimeError("Nothing was drawn, cannot compute the framing.")

    return min(xs), max(xs), min(ys), max(ys)


def generate_schematic_figure(
    output_path: str, size: float = 10.0, dpi: int = 200
) -> None:
    """Constructs and saves the permutation schemes schematic figure.

    The output canvas is strictly square (1:1). The three columns are centred
    inside that square, the drawing keeps an equal aspect ratio so the cells
    stay square, and the legend sits at the bottom, horizontally centred on the
    canvas rather than on the drawn content.

    Args:
        output_path (str): File path where the output image will be saved.
        size (float): Side of the square canvas, in inches.
        dpi (int): Output resolution. Final image is size * dpi pixels a side.

    Raises:
        IOError: If directory creation or image saving fails due to I/O issues.
    """
    output_directory: str = os.path.dirname(output_path)
    if output_directory:
        try:
            os.makedirs(output_directory, exist_ok=True)
        except OSError as error:
            raise IOError(
                f"Failed to create target directory '{output_directory}': {error}"
            ) from error

    fig, ax = plt.subplots(figsize=(size, size))
    # The axes fills the whole canvas: combined with equal data spans below,
    # this guarantees an exact 1:1 output without relying on tight_layout.
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)

    # Construct complete grids for original and RFX states
    original_grid: List[List[str]] = [
        [ORIGINAL_LABELS[s]] * NUMBER_OF_EPOCHS for s in range(NUMBER_OF_SUBJECTS)
    ]
    rfx_grid: List[List[str]] = [
        [RFX_LABELS[s]] * NUMBER_OF_EPOCHS for s in range(NUMBER_OF_SUBJECTS)
    ]

    try:
        draw_column(ax, 0.0, original_grid, "Original")
        draw_column(ax, 7.0, FFX_LABELS, "Epoch-level\npermutation (FFX)")
        draw_column(ax, 14.0, rfx_grid, "Subject-level\npermutation (RFX)")
    except ValueError as val_err:
        plt.close(fig)
        raise RuntimeError(f"Error drawing schematic columns: {val_err}") from val_err

    # Legend construction
    legend_handles: List[Rectangle] = [
        Rectangle((0, 0), 1, 1, facecolor=COLOR_HR, edgecolor="white"),
        Rectangle((0, 0), 1, 1, facecolor=COLOR_LR, edgecolor="white"),
    ]
    # Anchored in axes coordinates, so it is centred on the square canvas and
    # not on the bounding box of the three columns.
    ax.legend(
        legend_handles,
        ["High Recaller", "Low Recaller"],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.025),
        ncol=2,
        frameon=False,
        fontsize=11,
    )

    # Key descriptive annotations underneath the corresponding scheme columns
    ax.text(
        7.45,
        -0.75,
        "subject blocks broken,\nepochs mixed together",
        ha="center",
        va="top",
        fontsize=9,
        color="#c0392b",
        style="italic",
    )
    ax.text(
        14.45,
        -0.75,
        "subject blocks intact,\nonly block labels change",
        ha="center",
        va="top",
        fontsize=9,
        color="#2c7fb8",
        style="italic",
    )

    ax.set_aspect("equal")
    ax.axis("off")

    # The column titles and the italic annotations are much wider than the
    # coloured cells themselves, and their width in data units depends on the
    # limits we are trying to compute. Rather than guessing, the extents are
    # measured on a real render and the limits refined; two passes are enough
    # to converge, a third is kept as a safety margin.
    for _ in range(3):
        x_min, x_max, y_min, y_max = measure_content_extent(fig, ax)

        content_width: float = x_max - x_min
        content_height: float = y_max - y_min
        largest: float = max(content_width, content_height)

        side_margin: float = 0.04 * largest
        legend_band: float = 0.13 * largest  # empty strip kept for the legend

        # Square span: whichever constraint binds, the other axis is padded.
        span: float = max(content_width + 2 * side_margin,
                          content_height + legend_band + side_margin)

        x_centre: float = (x_min + x_max) / 2
        y_low: float = y_min - legend_band

        ax.set_xlim(x_centre - span / 2, x_centre + span / 2)
        ax.set_ylim(y_low, y_low + span)

    try:
        # No bbox_inches="tight": cropping to the ink would destroy the 1:1
        # ratio we just set up.
        fig.savefig(output_path, dpi=dpi, facecolor="white")
        print(f"[OK] Figure successfully written to: {output_path}")
    except OSError as save_error:
        raise IOError(f"Could not write figure to '{output_path}': {save_error}") from save_error
    finally:
        plt.close(fig)


def main() -> None:
    """Parses command-line arguments and runs the figure generation pipeline."""
    parser = argparse.ArgumentParser(
        description="Generate a conceptual schematic comparing FFX and RFX permutation schemes."
    )
    parser.add_argument(
        "--out",
        default=("/home/alouis/dream-recall-alex/final_plotted_figures/"
                 "fig02_perm_scheme.png"),
        help="Path where the output PNG file will be saved.",
    )
    parser.add_argument(
        "--size",
        type=float,
        default=10.0,
        help="Side of the square canvas, in inches (default: 10).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Output resolution (default: 200, i.e. 2000x2000 px at size 10).",
    )
    args = parser.parse_args()

    try:
        generate_schematic_figure(
            output_path=args.out, size=args.size, dpi=args.dpi
        )
    except Exception as err:
        print(f"[ERROR] Execution failed: {err}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()