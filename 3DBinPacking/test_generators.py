import json
import random
import os
from geometry import Box

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DATA_FILE = os.path.join(_MODULE_DIR, "last_test_data.json")
DEFAULT_SEED = 42

def generate_unit_cubes(container_width, container_height, container_depth):
    boxes = []
    total = int(container_width) * int(container_height) * int(container_depth)
    for i in range(1, total + 1):
        boxes.append(Box(i, 1, 1, 1))
    return container_width, container_height, container_depth, boxes


def generate_perfect_cubes(container_side, number_of_boxes):
    n = round(number_of_boxes ** (1.0 / 3.0))
    if n ** 3 != number_of_boxes:
        raise ValueError(
            f"Number of boxes ({number_of_boxes}) must be a perfect cube "
            f"(1, 8, 27, 64, 125, ...). Closest: {n**3}."
        )

    side = container_side / n
    boxes = []
    for i in range(1, number_of_boxes + 1):
        boxes.append(Box(i, side, side, side))
    return container_side, container_side, container_side, boxes


def generate_mixed_fit(container_side, number_of_boxes, seed=DEFAULT_SEED):
    rng = random.Random(seed)

    pieces = [(container_side, container_side, container_side)]

    while len(pieces) < number_of_boxes:

        pieces.sort(key=lambda p: p[0] * p[1] * p[2], reverse=True)
        biggest = pieces.pop(0)
        w, h, d = biggest

        max_dim = max(w, h, d)
        if max_dim < 2:
            pieces.append(biggest)
            break

        if w == max_dim:
            cut = rng.uniform(1, w - 1)
            piece_a = (cut, h, d)
            piece_b = (w - cut, h, d)
        elif h == max_dim:
            cut = rng.uniform(1, h - 1)
            piece_a = (w, cut, d)
            piece_b = (w, h - cut, d)
        else:
            cut = rng.uniform(1, d - 1)
            piece_a = (w, h, cut)
            piece_b = (w, h, d - cut)

        pieces.append(piece_a)
        pieces.append(piece_b)

    boxes = []
    for i, (w, h, d) in enumerate(pieces, start=1):
        boxes.append(Box(i, round(w, 2), round(h, 2), round(d, 2)))
    return container_side, container_side, container_side, boxes

def generate_realistic(container_width, container_height, container_depth,
                       number_of_boxes, seed=DEFAULT_SEED):
    rng = random.Random(seed)
    boxes = []
    for i in range(1, number_of_boxes + 1):
        if i % 3 == 0:
            box = Box(i, rng.randint(2, 3), rng.randint(4, 5), rng.randint(2, 3))
        elif i % 3 == 1:
            box = Box(i, rng.randint(4, 5), rng.randint(2, 3), rng.randint(2, 3))
        else:
            dim = rng.randint(2, 4)
            box = Box(i, dim, dim, dim)
        boxes.append(box)
    return container_width, container_height, container_depth, boxes

def save_test_data(container_width, container_height, container_depth, boxes, test_type):
    data = {
        "test_type": test_type,
        "container": {
            "width": container_width,
            "height": container_height,
            "depth": container_depth,
        },
        "boxes": [
            {"id": box.id, "width": box.width, "height": box.height, "depth": box.depth}
            for box in boxes
        ],
    }
    with open(TEST_DATA_FILE, "w") as f:
        json.dump(data, f, indent=2)


def load_test_data():
    if not os.path.exists(TEST_DATA_FILE):
        return None

    with open(TEST_DATA_FILE, "r") as f:
        data = json.load(f)

    container = data["container"]
    boxes = [
        Box(b["id"], b["width"], b["height"], b["depth"])
        for b in data["boxes"]
    ]
    return (
        container["width"],
        container["height"],
        container["depth"],
        boxes,
        data["test_type"],
    )


def test_data_exists():
    return os.path.exists(TEST_DATA_FILE)