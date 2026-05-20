import customtkinter as ctk
import random
import numpy as np 
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d.art3d import Poly3DCollection 
from geometry import Container, Box
from constructive import ExtremePointsHeuristic
from genetic import run_genetic_algorithm
from simulated_annealing import run_simulated_annealing
import threading
import queue

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("3D Bin Packing Optimizer")
        self.state('zoomed')
        self.update()

        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.current_width = 0.0
        self.current_height = 0.0
        self.current_depth = 0.0
        self.best_order = []
        self.best_orientations = []
        self.best_boxes_reference = []

        self.left_frame = ctk.CTkFrame(self, width=250, corner_radius=0)
        self.left_frame.grid(row=0, column=0, rowspan=2, sticky="nsew", padx=10, pady=10)
        
        ctk.CTkLabel(self.left_frame, text="Optimization Method:").pack(anchor="w", padx=10)
        self.method_variable = ctk.StringVar(value="Genetic Algorithm")
        self.dropdown_method = ctk.CTkOptionMenu(self.left_frame, values=["Genetic Algorithm", "Simulated Annealing"], variable=self.method_variable)
        self.dropdown_method.pack(fill="x", padx=10, pady=(0, 10))
        
        ctk.CTkLabel(self.left_frame, text="Dimensions (W, H, D):").pack(anchor="w", padx=10)
        self.entry_dimensions = ctk.CTkEntry(self.left_frame)
        self.entry_dimensions.pack(fill="x", padx=10, pady=(0, 10))
        self.entry_dimensions.insert(0, "10, 10, 10")
        
        ctk.CTkLabel(self.left_frame, text="Number of Test Boxes:").pack(anchor="w", padx=10)
        self.entry_boxes = ctk.CTkEntry(self.left_frame)
        self.entry_boxes.pack(fill="x", padx=10, pady=(0, 10))
        self.entry_boxes.insert(0, "15")

        ctk.CTkLabel(self.left_frame, text="Population Size (GA):").pack(anchor="w", padx=10)
        self.entry_population = ctk.CTkEntry(self.left_frame)
        self.entry_population.pack(fill="x", padx=10, pady=(0, 10))
        self.entry_population.insert(0, "50")

        ctk.CTkLabel(self.left_frame, text="Generations (GA):").pack(anchor="w", padx=10)
        self.entry_generations = ctk.CTkEntry(self.left_frame)
        self.entry_generations.pack(fill="x", padx=10, pady=(0, 10))
        self.entry_generations.insert(0, "40")

        self.btn_update_environment = ctk.CTkButton(self.left_frame, text="Apply Dimensions", command=self.init_empty_plot)
        self.btn_update_environment.pack(fill="x", padx=10, pady=10)

        ctk.CTkLabel(self.left_frame, text="Execution Statistics:", font=ctk.CTkFont(size=16, weight="bold")).pack(pady=(20, 5))
        self.statistics_box = ctk.CTkTextbox(self.left_frame, height=200)
        self.statistics_box.pack(fill="both", expand=True, padx=10, pady=10)

        self.plot_frame = ctk.CTkFrame(self)
        self.plot_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)

        self.fig = Figure(facecolor='#2b2b2b') 
        self.fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95)
        
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_facecolor('#2b2b2b')
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.canvas.draw_idle()
        self.update_idletasks()

        self.bottom_frame = ctk.CTkFrame(self, height=80, corner_radius=0)
        self.bottom_frame.grid(row=1, column=1, sticky="ew", padx=(0, 10), pady=10)
        
        self.bottom_frame.grid_columnconfigure(0, weight=1)
        self.bottom_frame.grid_columnconfigure(1, weight=0)

        self.slider_frame = ctk.CTkFrame(self.bottom_frame, fg_color="transparent")
        self.slider_frame.grid(row=0, column=0, sticky="ew", padx=20)
        
        self.label_generation = ctk.CTkLabel(self.slider_frame, text="Placement Evolution (Inactive):")
        self.label_generation.pack(side="left", padx=10)
        
        self.slider_generation = ctk.CTkSlider(self.slider_frame, from_=0, to=100, state="disabled", command=self.update_slider)
        self.slider_generation.pack(side="left", fill="x", expand=True, padx=10)

        self.btn_start = ctk.CTkButton(self.bottom_frame, text="START", 
                                       font=ctk.CTkFont(size=16, weight="bold"), 
                                       height=50, fg_color="#2FA572", hover_color="#106A43",
                                       command=self.run_optimization)
        self.btn_start.grid(row=0, column=1, padx=20, pady=15)

        self.init_empty_plot()

    def get_inputs(self):
        dimensions_string = self.entry_dimensions.get().split(',')
        try:
            width = float(dimensions_string[0])
            height = float(dimensions_string[1])
            depth = float(dimensions_string[2])
            return width, height, depth
        except (IndexError, ValueError):
            return 10.0, 10.0, 10.0

    def init_empty_plot(self, width=None, height=None, depth=None):
        try:
            if width is None or height is None or depth is None:
                width, height, depth = self.get_inputs()
                
            self.ax.clear()
            
            self.ax.set_xlim([0, width])
            self.ax.set_ylim([0, height])
            self.ax.set_zlim([0, depth])
            self.ax.set_box_aspect((width, height, depth))
            self.ax.set_xlabel('X Axis')
            self.ax.set_ylabel('Y Axis')
            self.ax.set_zlabel('Z Axis')
            
            self.ax.xaxis.label.set_color('white')
            self.ax.yaxis.label.set_color('white')
            self.ax.zaxis.label.set_color('white')
            self.ax.tick_params(axis='x', colors='white')
            self.ax.tick_params(axis='y', colors='white')
            self.ax.tick_params(axis='z', colors='white')

            self.ax.set_title(f"Empty Container ({width}x{height}x{depth})", color="white")
            self.canvas.draw_idle()
            
        except Exception as error:
            if hasattr(self, 'statistics_box'):
                self.statistics_box.insert("end", f"[!] Display Error: {error}\n")

    def draw_packed_container(self, container, extreme_points_list=None):
        self.init_empty_plot(container.width, container.height, container.depth)
        self.ax.set_title("Solution Visualization", color="white")

        base_colors = ['#e63946', '#457b9d', '#2a9d8f', '#f4a261', '#e76f51', '#8338ec']
        transparency = 0.4 

        # 1. Draw the boxes
        for index, box in enumerate(container.placed_boxes):
            hex_color = base_colors[index % len(base_colors)]
            x, y, z = box.x, box.y, box.z
            width, height, depth = box.width, box.height, box.depth

            vertices = np.array([
                [x, y, z], [x+width, y, z], [x+width, y+height, z], [x, y+height, z],    
                [x, y, z+depth], [x+width, y, z+depth], [x+width, y+height, z+depth], [x, y+height, z+depth] 
            ])

            faces_indices = [
                [0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4], 
                [2, 3, 7, 6], [1, 2, 6, 5], [0, 3, 7, 4]  
            ]
            
            faces_coordinates = [[vertices[j] for j in face] for face in faces_indices]

            poly3d = Poly3DCollection(faces_coordinates, 
                                      facecolors=hex_color, 
                                      edgecolors='black', 
                                      linewidths=1,
                                      alpha=transparency)
            self.ax.add_collection3d(poly3d)

        if extreme_points_list:
            extreme_point_x = [point[0] for point in extreme_points_list]
            extreme_point_y = [point[1] for point in extreme_points_list]
            extreme_point_z = [point[2] for point in extreme_points_list]
            
            self.ax.scatter(extreme_point_x, extreme_point_y, extreme_point_z, 
                            color='#00e5ff', 
                            edgecolors='white', 
                            s=80, 
                            depthshade=False, 
                            zorder=10, 
                            label="Extreme Points")

        self.canvas.draw_idle()

    def update_slider(self, value):
        step = int(float(value))
        self.label_generation.configure(text=f"Placed Boxes: {step} / {len(self.best_order)}")
        
        temp_container = Container(self.current_width, self.current_height, self.current_depth)
        heuristic = ExtremePointsHeuristic(temp_container)
        
        if not self.best_boxes_reference or step == 0:
            self.draw_packed_container(temp_container, extreme_points_list=[(0.0, 0.0, 0.0)])
            return

        for index in range(step):
            box_index = self.best_order[index]
            orientation_index = self.best_orientations[index]
            
            original_box = self.best_boxes_reference[box_index]
            box = Box(original_box.id, original_box.width, original_box.height, original_box.depth)
            
            allowed_orientations = box.get_allowed_orientations()
            box.rotate(*allowed_orientations[orientation_index % len(allowed_orientations)])
            heuristic.pack_box(box)
            
        self.draw_packed_container(temp_container, extreme_points_list=heuristic.extreme_points_list)

    def run_optimization(self):
        self.statistics_box.delete("0.0", "end")
        self.btn_start.configure(state="disabled", text="RUNNING...")
        self.slider_generation.configure(state="disabled") 

        try:
            width, height, depth = self.get_inputs()
            number_of_boxes = int(self.entry_boxes.get())
            population_size = int(self.entry_population.get())
            number_of_generations = int(self.entry_generations.get())

            test_boxes = []
            for i in range(1, number_of_boxes + 1):
                if i % 3 == 0:
                    box_obj = Box(i, random.randint(2, 3), random.randint(4, 5), random.randint(2, 3))
                elif i % 3 == 1:
                    box_obj = Box(i, random.randint(4, 5), random.randint(2, 3), random.randint(2, 3))
                else:
                    dimension = random.randint(2, 4)
                    box_obj = Box(i, dimension, dimension, dimension) 
                test_boxes.append(box_obj)
                
            selected_method = self.method_variable.get()
            
            self.statistics_box.insert("end", f"[*] Test data generated.\n[*] Starting {selected_method} in background...\n")
            
            self.thread_queue = queue.Queue()
            
            thread = threading.Thread(target=self._optimization_thread, 
                                      args=(width, height, depth, test_boxes, population_size, number_of_generations, number_of_boxes, selected_method))
            thread.daemon = True 
            thread.start()
            
            self.check_queue()

        except Exception as error:
            self.statistics_box.insert("end", f"\n[!] INITIALIZATION ERROR: {error}")
            self.btn_start.configure(state="normal", text="START")

    def _optimization_thread(self, width, height, depth, test_boxes, population_size, number_of_generations, number_of_boxes, selected_method):
        try:
            if selected_method == "Genetic Algorithm":
                best_individual, logbook = run_genetic_algorithm(
                    width, height, depth, test_boxes, 
                    population_size=population_size, 
                    maximum_generations=number_of_generations
                )
            else:
                best_individual, logbook = run_simulated_annealing(
                    width, height, depth, test_boxes,
                    initial_temperature=1000.0,
                    cooling_rate=0.95,
                    iterations_per_temperature=number_of_generations
                )
                
            self.thread_queue.put(("success", (width, height, depth, test_boxes, best_individual, number_of_boxes)))
        except Exception as error:
            self.thread_queue.put(("error", str(error)))
            

    def check_queue(self):
        try:
            message_type, data = self.thread_queue.get_nowait()
            
            if message_type == "success":
                width, height, depth, test_boxes, best_individual, number_of_boxes = data
                self._optimization_complete(width, height, depth, test_boxes, best_individual, number_of_boxes)
            elif message_type == "error":
                self._optimization_error(data)
                
        except queue.Empty:
            self.after(100, self.check_queue)
            

    def _optimization_complete(self, width, height, depth, test_boxes, best_individual, number_of_boxes):
        self.current_width = width
        self.current_height = height
        self.current_depth = depth
        self.best_order = best_individual[0]
        self.best_orientations = best_individual[1]
        self.best_boxes_reference = test_boxes

        self.slider_generation.configure(state="normal", from_=0, to=number_of_boxes, number_of_steps=number_of_boxes)
        self.slider_generation.set(number_of_boxes)
        
        self.update_slider(number_of_boxes)

        temp_container = Container(width, height, depth)
        heuristic = ExtremePointsHeuristic(temp_container)
        
        for index in range(len(self.best_order)):
            original_box = test_boxes[self.best_order[index]]
            box = Box(original_box.id, original_box.width, original_box.height, original_box.depth)
            allowed_orientations = box.get_allowed_orientations()
            box.rotate(*allowed_orientations[self.best_orientations[index] % len(allowed_orientations)])
            heuristic.pack_box(box)

        actual_occupied_volume = sum(box.width * box.height * box.depth for box in temp_container.placed_boxes)
        container_volume = width * height * depth
        fill_percentage = (actual_occupied_volume / container_volume) * 100

        self.statistics_box.insert("end", "\n--- FINAL RESULTS ---\n")
        self.statistics_box.insert("end", f"Fill Percentage: {fill_percentage:.2f}%\n")
        self.statistics_box.insert("end", f"Placed Boxes: {len(temp_container.placed_boxes)} / {number_of_boxes}\n")
        
        self.btn_start.configure(state="normal", text="START OPTIMIZATION")

    def _optimization_error(self, error_message):
        self.statistics_box.insert("end", f"\n[!] EXECUTION ERROR: {error_message}")
        self.btn_start.configure(state="normal", text="START OPTIMIZATION")

if __name__ == "__main__":
    app = App()
    app.mainloop()