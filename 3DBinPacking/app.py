import customtkinter as ctk
import traceback
import numpy as np 
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d.art3d import Poly3DCollection 
from geometry import Container, Box
from constructive import ExtremePointsHeuristic
from genetic_algorithm import run_genetic_algorithm
from tabu_search import run_tabu_search
from comparison_window import ComparisonWindow
from test_generators import (
    generate_unit_cubes,
    generate_perfect_cubes,
    generate_mixed_fit,
    generate_realistic,
    save_test_data,
    load_test_data,
    test_data_exists,
)
import threading
import queue

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

TEST_TYPES = {
    "Test 1: Unit Cubes": "unit_cubes",
    "Test 2: Perfect Cubes": "perfect_cubes",
    "Test 3: Mixed Fit": "mixed_fit",
    "Realistic": "realistic",
}


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

        self.last_ga_result = None
        self.last_ts_result = None

        self.left_frame = ctk.CTkFrame(self, width=280, corner_radius=0)
        self.left_frame.grid(row=0, column=0, rowspan=2, sticky="nsew", padx=10, pady=10)
        self.left_frame.grid_propagate(False)

        self.tabview = ctk.CTkTabview(self.left_frame, width=260)
        self.tabview.pack(fill="x", padx=5, pady=5)

        self.tabview.add("Main Menu")
        self.tabview.add("Genetic Algorithm")
        self.tabview.add("Tabu Search")

        basic_tab = self.tabview.tab("Main Menu")

        ctk.CTkLabel(basic_tab, text="Optimization Method:").pack(anchor="w", padx=5, pady=(5, 0))
        self.method_variable = ctk.StringVar(value="Genetic Algorithm")
        self.dropdown_method = ctk.CTkOptionMenu(
            basic_tab,
            values=["Genetic Algorithm", "Tabu Search"],
            variable=self.method_variable
        )
        self.dropdown_method.pack(fill="x", padx=5, pady=(0, 10))

        ctk.CTkLabel(basic_tab, text="Test Type:").pack(anchor="w", padx=5, pady=(5, 0))
        self.test_type_variable = ctk.StringVar(value="Realistic")
        self.dropdown_test_type = ctk.CTkOptionMenu(
            basic_tab,
            values=list(TEST_TYPES.keys()),
            variable=self.test_type_variable,
            command=self._on_test_type_changed
        )
        self.dropdown_test_type.pack(fill="x", padx=5, pady=(0, 10))

        ctk.CTkLabel(basic_tab, text="Dimensions (W, H, D):").pack(anchor="w", padx=5, pady=(5, 0))
        self.entry_dimensions = ctk.CTkEntry(basic_tab)
        self.entry_dimensions.pack(fill="x", padx=5, pady=(0, 10))
        self.entry_dimensions.insert(0, "10, 10, 10")

        ctk.CTkLabel(basic_tab, text="Number of Test Boxes:").pack(anchor="w", padx=5, pady=(5, 0))
        self.entry_boxes = ctk.CTkEntry(basic_tab)
        self.entry_boxes.pack(fill="x", padx=5, pady=(0, 10))
        self.entry_boxes.insert(0, "15")

        self.btn_update_environment = ctk.CTkButton(
            basic_tab, text="Apply Dimensions", command=self.init_empty_plot
        )
        self.btn_update_environment.pack(fill="x", padx=5, pady=(10, 5))

        self.btn_generate_data = ctk.CTkButton(
            basic_tab, text="Generate Test Data",
            fg_color="#4a9eff", hover_color="#2d7ed0",
            command=self.generate_test_data
        )
        self.btn_generate_data.pack(fill="x", padx=5, pady=5)

        self.label_data_status = ctk.CTkLabel(
            basic_tab, text="No data generated", text_color="gray",
            font=ctk.CTkFont(size=11, slant="italic")
        )
        self.label_data_status.pack(anchor="w", padx=5, pady=(0, 5))

        self.btn_compare = ctk.CTkButton(
            basic_tab, text="Show Comparison",
            fg_color="#8338ec", hover_color="#5a25a3",
            command=self.show_comparison,
            state="disabled"
        )
        self.btn_compare.pack(fill="x", padx=5, pady=(10, 5))

        self.label_compare_status = ctk.CTkLabel(
            basic_tab, text="GA: not run | TS: not run",
            text_color="gray",
            font=ctk.CTkFont(size=10, slant="italic")
        )
        self.label_compare_status.pack(anchor="w", padx=5, pady=(0, 5))

        ga_tab = self.tabview.tab("Genetic Algorithm")

        ctk.CTkLabel(ga_tab, text="Population Size:").pack(anchor="w", padx=5, pady=(5, 0))
        self.entry_population = ctk.CTkEntry(ga_tab)
        self.entry_population.pack(fill="x", padx=5, pady=(0, 10))
        self.entry_population.insert(0, "50")

        ctk.CTkLabel(ga_tab, text="Generations:").pack(anchor="w", padx=5, pady=(5, 0))
        self.entry_generations = ctk.CTkEntry(ga_tab)
        self.entry_generations.pack(fill="x", padx=5, pady=(0, 10))
        self.entry_generations.insert(0, "40")

        self.label_crossover = ctk.CTkLabel(ga_tab, text="Crossover Probability: 0.50")
        self.label_crossover.pack(anchor="w", padx=5, pady=(10, 0))
        self.slider_crossover = ctk.CTkSlider(
            ga_tab, from_=0.0, to=1.0, number_of_steps=100,
            command=self._update_crossover_label
        )
        self.slider_crossover.set(0.5)
        self.slider_crossover.pack(fill="x", padx=5, pady=(0, 10))

        self.label_mutation = ctk.CTkLabel(ga_tab, text="Mutation Probability: 0.0")
        self.label_mutation.pack(anchor="w", padx=5, pady=(5, 0))
        self.slider_mutation = ctk.CTkSlider(
            ga_tab, from_=0.0, to=0.1, number_of_steps=10,
            command=self._update_mutation_label
        )
        self.slider_mutation.set(0.0)
        self.slider_mutation.pack(fill="x", padx=5, pady=(0, 10))

        self.label_elitism = ctk.CTkLabel(ga_tab, text="Elitism Ratio: 0.00")
        self.label_elitism.pack(anchor="w", padx=5, pady=(5, 0))
        self.slider_elitism = ctk.CTkSlider(
            ga_tab, from_=0.0, to=0.15, number_of_steps=30,
            command=self._update_elitism_label
        )
        self.slider_elitism.set(0.0)
        self.slider_elitism.pack(fill="x", padx=5, pady=(0, 10))

        ts_tab = self.tabview.tab("Tabu Search")

        ctk.CTkLabel(ts_tab, text="Max Iterations:").pack(anchor="w", padx=5, pady=(5, 0))
        self.entry_ts_iterations = ctk.CTkEntry(ts_tab)
        self.entry_ts_iterations.pack(fill="x", padx=5, pady=(0, 10))
        self.entry_ts_iterations.insert(0, "100")

        ctk.CTkLabel(ts_tab, text="Max Iterations w/o Improvement:").pack(anchor="w", padx=5, pady=(5, 0))
        self.entry_ts_no_improve = ctk.CTkEntry(ts_tab)
        self.entry_ts_no_improve.pack(fill="x", padx=5, pady=(0, 10))
        self.entry_ts_no_improve.insert(0, "15")

        ctk.CTkLabel(ts_tab, text="Tabu List Size:").pack(anchor="w", padx=5, pady=(5, 0))
        self.entry_ts_tabu_size = ctk.CTkEntry(ts_tab)
        self.entry_ts_tabu_size.pack(fill="x", padx=5, pady=(0, 10))
        self.entry_ts_tabu_size.insert(0, "20")

        ctk.CTkLabel(
            self.left_frame, text="Execution Statistics:",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=(15, 5))
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
        
        self.slider_generation = ctk.CTkSlider(
            self.slider_frame, from_=0, to=100, state="disabled",
            command=self.update_slider
        )
        self.slider_generation.pack(side="left", fill="x", expand=True, padx=10)

        self.btn_start = ctk.CTkButton(
            self.bottom_frame, text="START", 
            font=ctk.CTkFont(size=16, weight="bold"), 
            height=50, fg_color="#2FA572", hover_color="#106A43",
            command=self.run_optimization
        )
        self.btn_start.grid(row=0, column=1, padx=20, pady=15)

        if test_data_exists():
            self.label_data_status.configure(
                text="Data loaded from previous session", text_color="#2FA572"
            )

        self.init_empty_plot()

    def _update_crossover_label(self, value):
        self.label_crossover.configure(text=f"Crossover Probability: {float(value):.2f}")

    def _update_mutation_label(self, value):
        self.label_mutation.configure(text=f"Mutation Probability: {float(value):.2f}")

    def _update_elitism_label(self, value):
        self.label_elitism.configure(text=f"Elitism Ratio: {float(value):.2f}")

    def _update_compare_button_state(self):
        ga_status = "OK" if self.last_ga_result is not None else "not run"
        ts_status = "OK" if self.last_ts_result is not None else "not run"
        self.label_compare_status.configure(text=f"GA: {ga_status} | TS: {ts_status}")

        if self.last_ga_result is not None and self.last_ts_result is not None:
            self.btn_compare.configure(state="normal")
        else:
            self.btn_compare.configure(state="disabled")

    def _reset_comparison_results(self):
        self.last_ga_result = None
        self.last_ts_result = None
        self._update_compare_button_state()

    def show_comparison(self):
        if self.last_ga_result is None or self.last_ts_result is None:
            return
        ComparisonWindow(self, self.last_ga_result, self.last_ts_result)

    def _on_test_type_changed(self, choice):
        test_id = TEST_TYPES[choice]
        if test_id == "unit_cubes":
            self.entry_boxes.configure(state="disabled")
            self.entry_dimensions.configure(state="normal")
        else:
            self.entry_boxes.configure(state="normal")
            self.entry_dimensions.configure(state="normal")

    def generate_test_data(self):
        self.statistics_box.delete("0.0", "end")

        self._reset_comparison_results()

        try:
            test_choice = self.test_type_variable.get()
            test_id = TEST_TYPES[test_choice]

            width, height, depth = self.get_inputs()
            number_of_boxes = int(self.entry_boxes.get())

            if test_id == "unit_cubes":
                w, h, d, boxes = generate_unit_cubes(width, height, depth)

            elif test_id == "perfect_cubes":
                if not (width == height == depth):
                    self.statistics_box.insert(
                        "end",
                        "[!] Warning: Container forced to cube using W. "
                        f"Was {width}x{height}x{depth}, now {width}x{width}x{width}\n"
                    )
                w, h, d, boxes = generate_perfect_cubes(width, number_of_boxes)

            elif test_id == "mixed_fit":
                if not (width == height == depth):
                    self.statistics_box.insert(
                        "end",
                        "[!] Warning: Container forced to cube using W. "
                        f"Was {width}x{height}x{depth}, now {width}x{width}x{width}\n"
                    )
                w, h, d, boxes = generate_mixed_fit(width, number_of_boxes)

            elif test_id == "realistic":
                w, h, d, boxes = generate_realistic(width, height, depth, number_of_boxes)

            save_test_data(w, h, d, boxes, test_id)

            self.statistics_box.insert(
                "end",
                f"[OK] Generated {len(boxes)} boxes for '{test_choice}'\n"
                f"     Container: {w}x{h}x{d}\n"
                f"     Saved to last_test_data.json\n"
            )
            self.label_data_status.configure(
                text=f"Data ready: {test_choice} ({len(boxes)} boxes)",
                text_color="#2FA572"
            )

            self.entry_dimensions.delete(0, "end")
            self.entry_dimensions.insert(0, f"{w}, {h}, {d}")
            self.init_empty_plot()

        except ValueError as ve:
            self.statistics_box.insert("end", f"[!] VALIDATION ERROR: {ve}\n")
            self.label_data_status.configure(text="Generation failed", text_color="#e63946")
        except Exception:
            self.statistics_box.insert("end", f"[!] ERROR:\n{traceback.format_exc()}\n")
            self.label_data_status.configure(text="Generation failed", text_color="#e63946")

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
            try:
                current_elev = self.ax.elev
                current_azim = self.ax.azim
            except AttributeError:
                current_elev = 20
                current_azim = -60

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
            self.ax.view_init(elev=current_elev, azim=current_azim)
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

            poly3d = Poly3DCollection(
                faces_coordinates, facecolors=hex_color, 
                edgecolors='black', linewidths=1, alpha=transparency
            )
            self.ax.add_collection3d(poly3d)

        if extreme_points_list:
            extreme_point_x = [point[0] for point in extreme_points_list]
            extreme_point_y = [point[1] for point in extreme_points_list]
            extreme_point_z = [point[2] for point in extreme_points_list]
            
            self.ax.scatter(
                extreme_point_x, extreme_point_y, extreme_point_z, 
                color='#00e5ff', edgecolors='white', s=80, 
                depthshade=False, zorder=10, label="Extreme Points"
            )

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

        if not test_data_exists():
            self.statistics_box.insert(
                "end",
                "[!] No test data generated. Click 'Generate Test Data' first.\n"
            )
            return

        self.btn_start.configure(state="disabled", text="RUNNING...")
        self.slider_generation.configure(state="disabled") 

        try:
            width, height, depth, test_boxes, test_type = load_test_data()
            number_of_boxes = len(test_boxes)

            selected_method = self.method_variable.get()

            if selected_method == "Genetic Algorithm":
                method_params = {
                    "population_size": int(self.entry_population.get()),
                    "maximum_generations": int(self.entry_generations.get()),
                    "crossover_probability": float(self.slider_crossover.get()),
                    "mutation_probability": float(self.slider_mutation.get()),
                    "elitism_ratio": float(self.slider_elitism.get()),
                }
            else:
                method_params = {
                    "max_iterations": int(self.entry_ts_iterations.get()),
                    "max_iterations_without_improvement": int(self.entry_ts_no_improve.get()),
                    "tabu_list_size": int(self.entry_ts_tabu_size.get()),
                }

            self.statistics_box.insert(
                "end",
                f"[*] Loaded {number_of_boxes} boxes (test: {test_type})\n"
                f"[*] Container: {width}x{height}x{depth}\n"
                f"[*] Starting {selected_method}...\n"
            )
            if selected_method == "Genetic Algorithm":
                p = method_params
                self.statistics_box.insert(
                    "end",
                    f"    Config: pop={p['population_size']}, gens={p['maximum_generations']}, "
                    f"cx={p['crossover_probability']:.2f}, mut={p['mutation_probability']:.2f}, "
                    f"elit={p['elitism_ratio']:.2f}\n"
                )
            else:
                p = method_params
                self.statistics_box.insert(
                    "end",
                    f"    Config: max_iter={p['max_iterations']}, "
                    f"no_improve={p['max_iterations_without_improvement']}, "
                    f"tabu_size={p['tabu_list_size']}\n"
                )
            
            self.thread_queue = queue.Queue()
            thread = threading.Thread(
                target=self._optimization_thread,
                args=(
                    width, height, depth, test_boxes,
                    number_of_boxes, selected_method, method_params,
                )
            )
            thread.daemon = True 
            thread.start()
            self.check_queue()

        except Exception as error:
            self.statistics_box.insert("end", f"\n[!] INITIALIZATION ERROR: {error}")
            self.btn_start.configure(state="normal", text="START")

    def _optimization_thread(self, width, height, depth, test_boxes,
                             number_of_boxes, selected_method, method_params):
        try:
            extra_info = {}
            if selected_method == "Genetic Algorithm":
                result = run_genetic_algorithm(
                    width, height, depth, test_boxes,
                    use_parallel=False,
                    **method_params
                )
                best_individual = result["best_individual"]
                extra_info = {
                    "execution_time": result["execution_time"],
                    "iterations_run": result["generations_run"],
                    "stopped_early": result["stopped_early"],
                    "max_iterations": method_params["maximum_generations"],
                    "iter_label": "Generations",
                    "total_evaluations": result["total_evaluations"],
                    "full_result": result,
                    "method": "GA",
                }
            else:
                result = run_tabu_search(
                    width, height, depth, test_boxes,
                    **method_params
                )
                best_individual = result["best_individual"]
                extra_info = {
                    "execution_time": result["execution_time"],
                    "iterations_run": result["iterations_run"],
                    "stopped_early": result["stopped_early"],
                    "max_iterations": method_params["max_iterations"],
                    "iter_label": "Iterations",
                    "total_evaluations": result["total_evaluations"],
                    "full_result": result,
                    "method": "TS",
                }
                
            self.thread_queue.put(("success", (width, height, depth, test_boxes, best_individual, number_of_boxes, extra_info)))
        except Exception:
            self.thread_queue.put(("error", traceback.format_exc()))

    def check_queue(self):
        try:
            message_type, data = self.thread_queue.get_nowait()
            if message_type == "success":
                width, height, depth, test_boxes, best_individual, number_of_boxes, extra_info = data
                self._optimization_complete(width, height, depth, test_boxes, best_individual, number_of_boxes, extra_info)
            elif message_type == "error":
                self._optimization_error(data)
        except queue.Empty:
            self.after(100, self.check_queue)

    def _optimization_complete(self, width, height, depth, test_boxes, best_individual, number_of_boxes, extra_info):
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

        if extra_info:
            self.statistics_box.insert("end", f"Execution Time: {extra_info['execution_time']:.2f}s\n")
            self.statistics_box.insert("end", f"Total Evaluations: {extra_info['total_evaluations']}\n")
            iter_label = extra_info.get("iter_label", "Iterations")
            if extra_info.get("stopped_early"):
                self.statistics_box.insert(
                    "end",
                    f"[OK] Early stopping at {iter_label.lower()[:-1]} {extra_info['iterations_run']} "
                    f"(reached 100% fill before max {extra_info['max_iterations']})\n"
                )
            else:
                self.statistics_box.insert(
                    "end",
                    f"{iter_label}: {extra_info['iterations_run']} / {extra_info['max_iterations']}\n"
                )

        if extra_info.get("method") == "GA":
            self.last_ga_result = extra_info["full_result"]
        elif extra_info.get("method") == "TS":
            self.last_ts_result = extra_info["full_result"]

        self._update_compare_button_state()
        
        self.btn_start.configure(state="normal", text="START")

    def _optimization_error(self, error_message):
        self.statistics_box.insert("end", f"\n[!] EXECUTION ERROR: {error_message}")
        self.btn_start.configure(state="normal", text="START")

if __name__ == "__main__":
    app = App()
    app.mainloop()