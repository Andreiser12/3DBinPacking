from geometry import Box, Container

class ExtremePointsHeuristic:
    def __init__(self, container: Container):
        self.container = container
        self.ep_list: list = [(0.0, 0.0, 0.0)]
        
        
    def sort_eps(self) -> None:
        self.ep_list.sort(key= lambda p: (p[2], p[1], p[0]))
        
        
    def remove_duplicates(self) -> None:
        self.ep_list= list(set(self.ep_list))
        
        
    def generate_new_eps(self, box: Box) -> None:
        ep_right = (box.x + box.width, box.y, box.z)
        ep_top = (box.x, box.y + box.height, box.z)
        ep_front = (box.x, box.y, box.z + box.depth)
        
        for ep in [ep_right, ep_top, ep_front]:
            if ep[0] < self.container.width and ep[1] < self.container.height and ep[2] < self.container.depth:
                self.ep_list.append(ep)
                
                
    def pack_box(self, box: Box) -> bool:
        self.remove_duplicates()
        self.sort_eps()

        for ep in self.ep_list:
            ep_x, ep_y, ep_z = ep
        
            orientations = box.get_allowed_orientations()
            
            for (w, h, d) in orientations:
                box.rotate(w, h, d)
                box.place(ep_x, ep_y, ep_z)
                
                if self.container.can_place_box(box):
                    self.container.add_box(box)
                    self.ep_list.remove(ep)
                    self.generate_new_eps(box)
                    return True
        
        return False
        

