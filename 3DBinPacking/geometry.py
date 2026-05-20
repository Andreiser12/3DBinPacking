
import itertools

class Box:
    def __init__(self, box_id: int, w: float, h: float, d: float):
        self.id: int = box_id
        
        self.width: float = h
        self.height: float = h
        self.depth: float = d
        
        self.x: float = None
        self.y: float = None
        self.z: float = None
        
        self.is_placed: bool = False
        
        
    def place(self, x: float, y: float, z: float) -> None:
        self.x = x
        self.y = y
        self.z = z
        self.is_placed = True
        
    def get_allowed_orientations(self) -> list:
        dims = (self.width, self.height, self.depth)
        unique_orientations = list(set(itertools.permutations(dims)))
        return unique_orientations
    
    def rotate(self, new_w: float, new_h: float, new_d: float) -> None:
        self.width = new_w
        self.height = new_h
        self.depth = new_d
        
        
class Container:
    def __init__(self, w: float, h: float, d: float):
        self.width: float = w
        self.height: float = h
        self.depth: float = d
        
        self.placed_boxes: list = []

    def is_inside_boundaries(self, box) -> bool:
        if box.x < 0 or box.x + box.width > self.width:
            return False
        if box.y < 0 or box.y + box.height > self.height:
            return False
        if box.z < 0 or box.z + box.depth > self.depth:
            return False
        return True

    def check_intersection(self, box1, box2) -> bool:
        separated_x = (box1.x + box1.width <= box2.x) or (box2.x + box2.width <= box1.x)
        separated_y = (box1.y + box1.height <= box2.y) or (box2.y + box2.height <= box1.y)
        separated_z = (box1.z + box1.depth <= box2.z) or (box2.z + box2.depth <= box1.z)
        
        if separated_x or separated_y or separated_z:
            return False
            
        return True

    def can_place_box(self, box) -> bool:
        if not self.is_inside_boundaries(box):
            return False
            
        for placed_box in self.placed_boxes:
            if self.check_intersection(box, placed_box):
                return False
                
        return True

    def add_box(self, box) -> bool:
        if self.can_place_box(box):
            self.placed_boxes.append(box)
            return True
        return False
        
        