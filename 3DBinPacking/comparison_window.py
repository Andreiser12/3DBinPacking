import customtkinter as ctk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class ComparisonWindow(ctk.CTkToplevel):
    def __init__(self, parent, ga_result, ts_result):
        super().__init__(parent)

        self.title("GA vs TS - Comparison")
        self.geometry("1100x750")

        self.ga_result = ga_result
        self.ts_result = ts_result

        self.grid_columnconfigure(0, weight=2)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self._build_chart()
        self._build_table()

    def _build_chart(self):
        chart_frame = ctk.CTkFrame(self)
        chart_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        fig = Figure(figsize=(7, 5), facecolor='#2b2b2b')
        ax = fig.add_subplot(111)
        ax.set_facecolor('#2b2b2b')

        ga_history = self.ga_result.get("fitness_history", [])
        ga_config = self.ga_result.get("config_used", {})
        pop_size = ga_config.get("population_size", 1)

        ga_x = []
        ga_y = []
        cumulative_evals = 0
        for gen, best in ga_history:
            if gen == 0:
                cumulative_evals = pop_size
            else:
                cumulative_evals += pop_size
            ga_x.append(cumulative_evals)
            ga_y.append(best)

        ts_history = self.ts_result.get("fitness_history", [])
        ts_config = self.ts_result.get("config_used", {})
        neighborhood = ts_config.get("neighborhood_size", 1)

        ts_x = []
        ts_y = []
        cumulative_evals = 1
        for iteration, best in ts_history:
            if iteration == 0:
                cumulative_evals = 1
            else:
                cumulative_evals += neighborhood
            ts_x.append(cumulative_evals)
            ts_y.append(best)

        ax.plot(ga_x, ga_y, color='#e63946', linewidth=2, label='GA (best fitness)', marker='o', markersize=3)
        ax.plot(ts_x, ts_y, color='#457b9d', linewidth=2, label='TS (best fitness)', marker='s', markersize=3)

        ax.set_xlabel('Total Fitness Evaluations', color='white')
        ax.set_ylabel('Best Fitness (Volume)', color='white')
        ax.set_title('Convergence: GA vs TS', color='white')
        ax.tick_params(colors='white')
        ax.legend(facecolor='#2b2b2b', edgecolor='white', labelcolor='white')
        ax.grid(True, alpha=0.3)

        for spine in ax.spines.values():
            spine.set_color('white')

        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    def _build_table(self):
        table_frame = ctk.CTkScrollableFrame(self)
        table_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)

        ctk.CTkLabel(
            table_frame, text="Comparison Metrics",
            font=ctk.CTkFont(size=18, weight="bold")
        ).pack(pady=(10, 15))

        # extragem metrici
        ga = self.ga_result
        ts = self.ts_result

        ga_fitness = ga.get("best_fitness", 0)
        ts_fitness = ts.get("best_fitness", 0)

        ga_fill = ga.get("fill_percentage", 0)
        ts_fill = ts.get("fill_percentage", 0)

        ga_placed = ga.get("placed_boxes", 0)
        ts_placed = ts.get("placed_boxes", 0)
        ga_total = ga.get("total_boxes", 0)
        ts_total = ts.get("total_boxes", 0)

        ga_evals = ga.get("total_evaluations", 0)
        ts_evals = ts.get("total_evaluations", 0)

        ga_time = ga.get("execution_time", 0)
        ts_time = ts.get("execution_time", 0)

        ga_iters = ga.get("generations_run", 0)
        ts_iters = ts.get("iterations_run", 0)

        rows = [
            ("Metric", "GA", "TS", True),
            ("Best Fitness", f"{ga_fitness:.2f}", f"{ts_fitness:.2f}", False),
            ("Fill Percentage", f"{ga_fill:.2f}%", f"{ts_fill:.2f}%", False),
            ("Placed Boxes", f"{ga_placed}/{ga_total}", f"{ts_placed}/{ts_total}", False),
            ("Total Evaluations", f"{ga_evals}", f"{ts_evals}", False),
            ("Execution Time", f"{ga_time:.2f}s", f"{ts_time:.2f}s", False),
            ("Iterations / Generations", f"{ga_iters}", f"{ts_iters}", False),
        ]

        table = ctk.CTkFrame(table_frame, fg_color="transparent")
        table.pack(fill="x", padx=10, pady=5)

        for row_idx, (metric, ga_val, ts_val, is_header) in enumerate(rows):
            font_weight = "bold" if is_header else "normal"
            font_size = 13 if is_header else 12

            label_metric = ctk.CTkLabel(
                table, text=metric, anchor="w",
                font=ctk.CTkFont(size=font_size, weight=font_weight)
            )
            label_ga = ctk.CTkLabel(
                table, text=ga_val, anchor="center",
                font=ctk.CTkFont(size=font_size, weight=font_weight),
                text_color="#e63946" if not is_header else None
            )
            label_ts = ctk.CTkLabel(
                table, text=ts_val, anchor="center",
                font=ctk.CTkFont(size=font_size, weight=font_weight),
                text_color="#457b9d" if not is_header else None
            )

            label_metric.grid(row=row_idx, column=0, sticky="ew", padx=5, pady=4)
            label_ga.grid(row=row_idx, column=1, sticky="ew", padx=5, pady=4)
            label_ts.grid(row=row_idx, column=2, sticky="ew", padx=5, pady=4)

        table.grid_columnconfigure(0, weight=2)
        table.grid_columnconfigure(1, weight=1)
        table.grid_columnconfigure(2, weight=1)

        # summary
        ctk.CTkLabel(
            table_frame, text="Summary",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=(20, 5))

        if ga_fitness > ts_fitness:
            winner_text = f"GA achieved better fitness\n({ga_fitness:.2f} vs {ts_fitness:.2f})"
            winner_color = "#e63946"
        elif ts_fitness > ga_fitness:
            winner_text = f"TS achieved better fitness\n({ts_fitness:.2f} vs {ga_fitness:.2f})"
            winner_color = "#457b9d"
        else:
            winner_text = "Both achieved equal fitness"
            winner_color = "#2FA572"

        ctk.CTkLabel(
            table_frame, text=winner_text,
            font=ctk.CTkFont(size=13),
            text_color=winner_color
        ).pack(pady=5)

        if ga_evals > 0 and ts_evals > 0:
            ga_efficiency = ga_fitness / ga_evals
            ts_efficiency = ts_fitness / ts_evals
            eff_text = (f"Fitness per evaluation:\n"
                        f"GA: {ga_efficiency:.2f}  |  TS: {ts_efficiency:.2f}")
            ctk.CTkLabel(
                table_frame, text=eff_text,
                font=ctk.CTkFont(size=12),
                text_color="#2a9d8f"
            ).pack(pady=5)

        if ga_evals > 0 and ts_evals > 0:
            ratio = ga_evals / ts_evals if ts_evals < ga_evals else ts_evals / ga_evals
            if ratio > 1.5:
                bigger = "GA" if ga_evals > ts_evals else "TS"
                note = (f"[!] Note: {bigger} used {ratio:.1f}x more evaluations.\n"
                        "Comparison is not on equal budget.")
                ctk.CTkLabel(
                    table_frame, text=note,
                    font=ctk.CTkFont(size=11, slant="italic"),
                    text_color="#f4a261", wraplength=300, justify="center"
                ).pack(pady=10)