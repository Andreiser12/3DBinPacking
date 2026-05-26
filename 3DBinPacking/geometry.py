import itertools

EPSILON = 1e-6

class Box:
    def __init__(self, box_id: int, width: float, height: float, depth: float):
        self.id: int = box_id
        self.width: float = width 
        self.height: float = height
        self.depth: float = depth
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
        dimensions = (self.width, self.height, self.depth)
        unique_orientations = list(set(itertools.permutations(dimensions)))
        return unique_orientations
    
    
    def rotate(self, new_width: float, new_height: float, new_depth: float) -> None:
        self.width = new_width
        self.height = new_height
        self.depth = new_depth
        
        
class Container:
    def __init__(self, width: float, height: float, depth: float):
        self.width: float = width
        self.height: float = height
        self.depth: float = depth
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


    def is_stable(self, box) -> bool:
        if box.z <= EPSILON:
            return True

        center_x = box.x + box.width / 2.0
        center_y = box.y + box.height / 2.0

        for placed_box in self.placed_boxes:
            placed_top_z = placed_box.z + placed_box.depth

            if abs(placed_top_z - box.z) > EPSILON:
                continue

            inside_x = placed_box.x - EPSILON <= center_x <= placed_box.x + placed_box.width + EPSILON
            inside_y = placed_box.y - EPSILON <= center_y <= placed_box.y + placed_box.height + EPSILON

            if inside_x and inside_y:
                return True

        return False


    def can_place_box(self, box) -> bool:
        if not self.is_inside_boundaries(box):
            return False
            
        for placed_box in self.placed_boxes:
            if self.check_intersection(box, placed_box):
                return False

        if not self.is_stable(box):
            return False
                
        return True
    

    def add_box(self, box) -> bool:
        if self.can_place_box(box):
            self.placed_boxes.append(box)
            return True
        return False