import os
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from habitat.core.utils import try_cv2_import
cv2 = try_cv2_import()

COLOR_PALETTE = {}

COLOR_PALETTE["none"] = [0, 0, 0]

COLOR_PALETTE["chair"] = [255, 102, 102]
COLOR_PALETTE["table"] = [255, 178, 102]
COLOR_PALETTE["bed"] = [178, 255, 102]
COLOR_PALETTE["sink"] = [153, 255, 204]
COLOR_PALETTE["tv_monitor"] = [153, 255, 255]
COLOR_PALETTE["toilet"] = [153, 153, 255]
COLOR_PALETTE["potted_plant"] = [204, 153, 255]
COLOR_PALETTE["plant"] = [204, 153, 255]
COLOR_PALETTE["book"] = [255, 51, 153]

COLOR_PALETTE["node"] = [0, 0, 0]
COLOR_PALETTE["object_center_gt"] = [0, 0, 0]
COLOR_PALETTE["room_center_gt"] = [0, 0, 0]
COLOR_PALETTE["door_center_gt"] = [0, 0, 0]
COLOR_PALETTE["trajectory"] = [0, 0, 0]
COLOR_PALETTE["major"] = [0, 0, 0]

def draw_rectangle(
    top_down_map: np.ndarray,
    corners: List[Tuple],
    object: str = "none"
) -> None:
    r"""Draw rectangle on top_down_map (in place)
    Args:
        top_down_map: A colored version of the map.
        corners: corners
        object: category
    """
    if object not in COLOR_PALETTE : object = "none"
    cv2.rectangle(
        top_down_map,
        corners[0][::-1],
        corners[1][::-1],
        COLOR_PALETTE[object],
        -1
    )

def draw_circle(
    top_down_map: np.ndarray,
    center: Tuple,
    radius: int = 5,
    node_type: str = "node"
) -> None:
    r"""Draw path on top_down_map (in place) with specified color.
    Args:
        top_down_map: A colored version of the map.
        color: color code of the path, from TOP_DOWN_MAP_COLORS.
        path_points: list of points that specify the path to be drawn
        thickness: thickness of the path.
    """
    cv2.circle(
        top_down_map,
        center,
        radius,
        COLOR_PALETTE[node_type]
    )
