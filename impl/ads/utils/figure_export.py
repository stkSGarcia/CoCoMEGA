"""Save matplotlib figures for ACM publication (PDF + high-resolution PNG)."""
import pathlib

PUBLICATION_RASTER_DPI = 600
PUBLICATION_RASTER_SUFFIX = ".png"


def save_publication_figure(fig, path, dpi=PUBLICATION_RASTER_DPI, **kwargs):
    """Save a figure for publication.

    When *path* ends with ``.pdf``, also writes a lossless high-resolution PNG
    alongside (same stem). Raster-only paths are saved at *dpi*.
    """
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        fig.savefig(path, **kwargs)
        raster_kwargs = dict(kwargs)
        raster_kwargs["dpi"] = dpi
        raster_kwargs.setdefault("bbox_inches", "tight")
        fig.savefig(path.with_suffix(PUBLICATION_RASTER_SUFFIX), **raster_kwargs)
    else:
        save_kwargs = dict(kwargs)
        save_kwargs.setdefault("dpi", dpi)
        fig.savefig(path, **save_kwargs)
