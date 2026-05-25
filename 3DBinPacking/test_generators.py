"""
Generatoare de date de test pentru 3D Bin Packing.

4 tipuri de teste:
    1. Unit Cubes        - container LxlxH, cutii 1x1x1 (sanity check, fill teoretic 100%)
    2. Perfect Cubes     - container cub LxLxL, n^3 cutii de (L/n)^3 (sanity check, fill teoretic 100%)
    3. Mixed Fit         - container cub LxLxL, cutii mixte cu volum total = L^3 (discriminativ)
    4. Realistic         - dimensiuni alese de user, cutii random (discriminativ)

Toate testele cu randomizare folosesc seed fix pentru reproducibilitate.
"""

import json
import random
import os
from geometry import Box

TEST_DATA_FILE = "last_test_data.json"
DEFAULT_SEED = 42


# ============================================================
# GENERATORS
# ============================================================

def generate_unit_cubes(container_width, container_height, container_depth):
    """
    Test 1: Cutii 1x1x1 care umplu perfect containerul.
    Nr cutii = W * H * D. Fill teoretic 100%.
    """
    boxes = []
    total = int(container_width) * int(container_height) * int(container_depth)
    for i in range(1, total + 1):
        boxes.append(Box(i, 1, 1, 1))
    return container_width, container_height, container_depth, boxes


def generate_perfect_cubes(container_side, number_of_boxes):
    """
    Test 2: Cub LxLxL spart in n^3 cutii egale de (L/n)^3.
    Validare: number_of_boxes trebuie sa fie cub perfect.
    """
    # verificam ca number_of_boxes e cub perfect
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
    """
    Test 3: Cub LxLxL spart in N bucati cu volume mixte.
    Algoritm: recursive splitting - alegem o cutie la intamplare, o spargem pe o axa.
    Suma volumelor = L^3 (fill teoretic 100% daca algoritmul gaseste asezarea).
    """
    rng = random.Random(seed)

    # incepem cu o singura "cutie" = containerul intreg
    # reprezentam fiecare cutie ca (w, h, d)
    pieces = [(container_side, container_side, container_side)]

    while len(pieces) < number_of_boxes:
        # alegem cea mai mare cutie (ca sa avem ce sparge)
        # sortam descrescator dupa volum si luam prima
        pieces.sort(key=lambda p: p[0] * p[1] * p[2], reverse=True)
        biggest = pieces.pop(0)
        w, h, d = biggest

        # alegem axa pe care spargem (cea mai lunga, ca sa avem loc)
        max_dim = max(w, h, d)
        if max_dim < 2:
            # nu mai putem sparge nimic util, oprim
            pieces.append(biggest)
            break

        # alegem un punct de taiere intre 1 si max_dim - 1
        # (ca sa avem ambele bucati cu dim >= 1)
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

    # rotunjim dimensiunile la 2 zecimale ca sa fie mai usor de vizualizat
    boxes = []
    for i, (w, h, d) in enumerate(pieces, start=1):
        boxes.append(Box(i, round(w, 2), round(h, 2), round(d, 2)))
    return container_side, container_side, container_side, boxes


def generate_realistic(container_width, container_height, container_depth,
                       number_of_boxes, seed=DEFAULT_SEED):
    """
    Test 4: Cutii random cu dimensiuni variate (cum era logica veche).
    Seed fix => reproducibil pentru comparatie GA vs SA.
    """
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


# ============================================================
# SAVE / LOAD
# ============================================================

def save_test_data(container_width, container_height, container_depth, boxes, test_type):
    """Salveaza datele intr-un fisier JSON simplu."""
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
    """Incarca datele din fisier. Returneaza (container_dims, boxes, test_type) sau None daca nu exista."""
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
    """Verifica daca exista date de test salvate."""
    return os.path.exists(TEST_DATA_FILE)