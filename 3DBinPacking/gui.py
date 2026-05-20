import customtkinter as ctk
import random
from geometry import Container, Box
from constructive import ExtremePointsHeuristic
from main import vizualizare_3d

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("3D Bin Packing Optimizer")
        self.geometry("700x450")

        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.sidebar_frame = ctk.CTkFrame(self, width=200, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(6, weight=1)

        self.logo_label = ctk.CTkLabel(self.sidebar_frame, text="Panou Control", font=ctk.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))

        self.label_container = ctk.CTkLabel(self.sidebar_frame, text="Dimensiuni Container (W, H, D):")
        self.label_container.grid(row=1, column=0, padx=20, pady=5, sticky="w")
        
        self.entry_container = ctk.CTkEntry(self.sidebar_frame, placeholder_text="ex: 10, 10, 10")
        self.entry_container.grid(row=2, column=0, padx=20, pady=5)
        self.entry_container.insert(0, "10, 10, 10")

        self.label_cutii = ctk.CTkLabel(self.sidebar_frame, text="Număr cutii aleatoare:")
        self.label_cutii.grid(row=3, column=0, padx=20, pady=5, sticky="w")
        
        self.entry_cutii = ctk.CTkEntry(self.sidebar_frame, placeholder_text="ex: 15")
        self.entry_cutii.grid(row=4, column=0, padx=20, pady=5)
        self.entry_cutii.insert(0, "15")

        self.btn_run_euristic = ctk.CTkButton(self.sidebar_frame, text="Rulează Extreme Points", command=self.run_euristic)
        self.btn_run_euristic.grid(row=5, column=0, padx=20, pady=(20, 10))

        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        
        self.label_rezultat = ctk.CTkLabel(self.main_frame, text="Rezultate Execuție:", font=ctk.CTkFont(size=16, weight="bold"))
        self.label_rezultat.pack(pady=(10, 5), anchor="w", padx=10)

        self.textbox_log = ctk.CTkTextbox(self.main_frame, width=400, height=300)
        self.textbox_log.pack(pady=10, padx=10, fill="both", expand=True)

        self.current_container = None

    def run_euristic(self):
        """Funcția care se execută când apeși butonul"""
        self.textbox_log.delete("0.0", "end")
        
        try:
            dim_str = self.entry_container.get().split(',')
            w, h, d = float(dim_str[0]), float(dim_str[1]), float(dim_str[2])
            nr_cutii = int(self.entry_cutii.get())
            
            self.textbox_log.insert("end", f"[!] Inițializare container {w}x{h}x{d}...\n")
            self.current_container = Container(w, h, d)
            euristica = ExtremePointsHeuristic(self.current_container)

            self.textbox_log.insert("end", f"[!] Generăm {nr_cutii} cutii aleatoare...\n")
            cutii = []
            for i in range(1, nr_cutii + 1):
                cutii.append(Box(i, random.randint(2, 5), random.randint(2, 5), random.randint(2, 5)))

            plasate = 0
            for cutie in cutii:
                if euristica.pack_box(cutie):
                    plasate += 1
            
            vol_ocupat = sum(b.width * b.height * b.depth for b in self.current_container.placed_boxes)
            vol_total = w * h * d
            procent = (vol_ocupat / vol_total) * 100

            self.textbox_log.insert("end", "\n--- RAPORT FINAL ---\n")
            self.textbox_log.insert("end", f"Cutii plasate: {plasate} din {nr_cutii}\n")
            self.textbox_log.insert("end", f"Grad de umplere spațiu: {procent:.2f}%\n")
            self.textbox_log.insert("end", "\n[!] Se deschide vizualizarea 3D...\n")
            self.update() 
            
            vizualizare_3d(self.current_container)

        except Exception as e:
            self.textbox_log.insert("end", f" EROARE: Verifică dacă datele introduse sunt corecte.\nDetalii: {e}")

if __name__ == "__main__":
    app = App()
    app.mainloop()

