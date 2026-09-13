import io
import numpy as np
import matplotlib as mpl
import cairosvg
from collections import namedtuple
from pathlib import Path
from PIL import ImageChops, Image


mpl.use('SVG')
mpl.rcParams['svg.hashsalt'] = 'fixed_salt_123'
mpl.rcParams['svg.fonttype'] = 'none'
np.random.seed(42)

PKG_DATA_DIR = Path(__file__).parent / "data"


def images_identical(img1_path: Path, img2_path: Path):
    def rasterize(svg_path):
        png_data = cairosvg.svg2png(url=svg_path)
        return Image.open(io.BytesIO(png_data)).convert("RGB")
    def compare_visual(svg1, svg2):
        img1 = rasterize(svg1)
        img2 = rasterize(svg2)
        diff = ImageChops.difference(img1, img2)
        return diff.getbbox() is None  # True if identical
    return compare_visual(str(img1_path), str(img2_path))


class ResultFigures:

    generated: Path
    reference: Path

    def __init__(self, stem: str):
        self.generated = PKG_DATA_DIR / f'{stem}_gen.svg'
        self.reference = PKG_DATA_DIR / f'{stem}_ref.svg'
        # assert self.reference.exists()