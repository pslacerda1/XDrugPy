import io
import numpy as np
import matplotlib as mpl
import cairosvg
from pathlib import Path
from PIL import Image, ImageChops, ImageStat


PKG_DATA_DIR = Path(__file__).parent / "data"


def images_identical(img1_path: Path, img2_path: Path, rms_threshold: float = 20.0) -> bool:
    def rasterize(img_path) -> Image.Image:
        png_data = cairosvg.svg2png(url=str(img_path))
        return Image.open(io.BytesIO(png_data)).convert("RGB")

    img1 = rasterize(img1_path)
    img2 = rasterize(img2_path)

    img1.save(img1_path.with_suffix(".png"))
    img2.save(img2_path.with_suffix(".png"))

    # Garante que têm o mesmo tamanho para evitar erros no comparador
    if img1.size != img2.size:
        return False

    # Calcula a diferença absoluta pixel a pixel
    diff = ImageChops.difference(img1, img2)

    # Analisa estatísticas da diferença
    stat = ImageStat.Stat(diff)
    # stat.rms retorna uma lista com o RMS por canal (R, G, B). Tiramos a média.
    mean_rms = sum(stat.rms) / len(stat.rms)

    return mean_rms <= rms_threshold


class ResultFigures:

    generated: Path
    reference: Path

    def __init__(self, stem: str):
        self.generated = PKG_DATA_DIR / f'{stem}_gen.svg'
        self.reference = PKG_DATA_DIR / f'{stem}_ref.svg'
