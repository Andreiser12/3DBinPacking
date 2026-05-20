from geometry import Box, Container

class ExtremePointsHeuristic:
    def __init__(self, container: Container):
        self.container = container
        self.extreme_points_list: list = [(0.0, 0.0, 0.0)]
        
    def sort_extreme_points(self) -> None:
        self.extreme_points_list.sort(key=lambda point: (point[2], point[1], point[0]))
        
    def remove_duplicates(self) -> None:
        self.extreme_points_list = list(set(self.extreme_points_list))
        
    def generate_new_extreme_points(self, box: Box) -> None:
        extreme_point_right = (box.x + box.width, box.y, box.z)
        extreme_point_top = (box.x, box.y + box.height, box.z)
        extreme_point_front = (box.x, box.y, box.z + box.depth)
        
        for extreme_point in [extreme_point_right, extreme_point_top, extreme_point_front]:
            if (extreme_point[0] < self.container.width and 
                extreme_point[1] < self.container.height and 
                extreme_point[2] < self.container.depth):
                self.extreme_points_list.append(extreme_point)
                
    def pack_box(self, box: Box) -> bool:
        self.remove_duplicates()
        self.sort_extreme_points()

        for extreme_point in self.extreme_points_list:
            extreme_point_x, extreme_point_y, extreme_point_z = extreme_point
        
            orientations = box.get_allowed_orientations()
            
            for (width, height, depth) in orientations:
                box.rotate(width, height, depth)
                box.place(extreme_point_x, extreme_point_y, extreme_point_z)
                
                if self.container.can_place_box(box):
                    self.container.add_box(box)
                    self.extreme_points_list.remove(extreme_point)
                    self.generate_new_extreme_points(box)
                    return True
        
        return False