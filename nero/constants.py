from __future__ import annotations
from typing import Tuple


def percentage_rgb(r: float, g: float, b: float) -> Colour:
    max_rgb = 255.0
    return r / max_rgb, g / max_rgb, b / max_rgb


Colour = Tuple[float, float, float]

AGH_RED_COLOUR = percentage_rgb(167.0, 25.0, 4.0)
AGH_GREEN_COLOUR = percentage_rgb(0.0, 105.0, 60.0)
AGH_GREY_COLOUR = percentage_rgb(30.0, 30.0, 30.0)
AGH_LIGHT_GREY_COLOUR = percentage_rgb(130.0, 130.0, 130.0)

STANDARD_ALPHA = 0.2
STANDARD_LINE_WIDTH = 1.5

# storage_path = <local path>
storage_path = "/net/tscratch/people/plgglegeza/nero"

CACHE_DIR = f"{storage_path}/cache"
DOWNLOADS_DIR = f"{storage_path}/downloads"
PICKLES_DIR = f"{storage_path}/pickles"
LOGS_DIR = f"{storage_path}/results/logs"
CSV_DIR = f"{storage_path}/results/csv"

INSIGHT_BINS_NO = 500
