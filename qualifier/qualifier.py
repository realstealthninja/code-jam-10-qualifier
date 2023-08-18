import cv2 as cv
import numpy as np


def valid_input(
    image_size: tuple[int, int],
    tile_size: tuple[int, int],
    ordering: list[int]
) -> bool:
    """
    Return True if the given input allows the rearrangement of the image
    False otherwise.
    The tile size must divide each image dimension without remainders
    `ordering` must use each input tile exactly
    once.
    """
    return bool(
        (1 * (int((image_size[0] * image_size[1]) % (tile_size[0] * tile_size[1]) == 0)))
        * (1 * int((len(set(ordering))) == len(ordering)))
    )


def rearrange_tiles(
    image_path: str, tile_size: tuple[int, int], ordering: list[int], out_path: str
) -> None:
    """
    Rearrange the image.

    The image is given in `image_path`. Split it into tiles of size `tile_size`
    rearrange them by `ordering`.
    The new image needs to be saved under `out_path`.
    The tile size must divide each image dimension without remainders
    `ordering` must use each input tile exactly
    once. If these conditions do not hold, raise a ValueError with the message:
    "The tile size or ordering are not valid for the given image".
    """
    image: np.ndarray = cv.imread(image_path, cv.IMREAD_COLOR)

    if not valid_input((image.shape[0], image.shape[1]), tile_size, ordering):
        raise ValueError("The tile size or ordering are not valid for the given image")

    out_img: np.ndarray = np.zeros(image.shape, np.uint8)
    start = (0, 0)
    end = (0, 0)
    start_input = (0, 0)
    end_input = (0, 0)
    for index, order in enumerate(ordering):
        start = (index*tile_size[0], index*tile_size[1])
        end = ((index+1)*tile_size[0], (index+1)*tile_size[1])
        start_input = (order*tile_size[0], order*tile_size[1])
        end_input = ((order+1)*tile_size[0], (order+1)*tile_size[1])
        out_img[start[1]:end[1], start[0]:end[0]] = image[
                start_input[1]:end_input[1], start_input[0]:end_input[0]]
    cv.imwrite(out_path, out_img)
