import math
from dataclasses import dataclass


@dataclass
class GridDimensions2D:
    width: int
    height: int


def find_cortical_sheet_size(area: float):
    length = int(math.sqrt(area))  # Starting with a square shape
    while area % length != 0:
        length -= 1

    breadth = area // length

    return GridDimensions2D(width=breadth, height=length)
