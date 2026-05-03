import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import random
import time

# ==========================================
# ЯДРО АЛГОРИТМІВ
# ==========================================
class GeneticAlgorithm:
    def __init__(self, num_cities, pop_size, mutation_rate):
        self.num_cities = num_cities
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.population = [random.sample(range(num_cities), num_cities) for _ in range(pop_size)]

    def crossover(self, parent1, parent2):
        start, end = sorted(random.sample(range(self.num_cities), 2))
        child = [-1] * self.num_cities
        child[start:end] = parent1[start:end]
        
        ptr = 0
        for gene in parent2:
            if gene not in child:
                while child[ptr] != -1:
                    ptr += 1
                child[ptr] = gene
        return child

    def mutate(self, route):
        if random.random() < self.mutation_rate:
            i, j = random.sample(range(self.num_cities), 2)
            route[i], route[j] = route[j], route[i]
        return route

    def evolve(self, distance_matrix):
        self.population.sort(key=lambda route: self.route_distance(route, distance_matrix))
        next_generation = self.population[:int(self.pop_size * 0.2)] 
        
        while len(next_generation) < self.pop_size:
            parent1, parent2 = random.sample(self.population[:int(self.pop_size * 0.5)], 2)
            child = self.crossover(parent1, parent2)
            child = self.mutate(child)
            next_generation.append(child)
            
        self.population = next_generation
        return self.population[0], self.route_distance(self.population[0], distance_matrix)

    def route_distance(self, route, dist_matrix):
        return sum(dist_matrix[route[i], route[i-1]] for i in range(self.num_cities))


class HopfieldOptimizer:
    @staticmethod
    def optimize(route, dist_matrix):
        improved = True
        best_route = route.copy()
        num_cities = len(route)
        
        def calc_dist(r):
            return sum(dist_matrix[r[i], r[i-1]] for i in range(num_cities))
            
        best_dist = calc_dist(best_route)
        
        while improved:
            improved = False
            for i in range(1, num_cities - 2):
                for j in range(i + 1, num_cities):
                    if j - i == 1: continue
                    
                    new_route = best_route.copy()
                    new_route[i:j] = best_route[j-1:i-1:-1]
                    new_dist = calc_dist(new_route)
                    
                    if new_dist < best_dist:
                        best_dist = new_dist
                        best_route = new_route
                        improved = True
        return best_route, best_dist


# ==========================================
# ГРАФІЧНИЙ ІНТЕРФЕЙС (GUI З ДИЗАЙНОМ)
# ==========================================
class TSPApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Оптимізація маршрутів: ГА + Мережа Хопфілда")
        self.root.geometry("1200x750")
        self.root.configure(bg="#EAECEE") # Світло-сірий фон
        
        self.apply_styles()
        
        self.cities_coords = None
        self.dist_matrix = None
        
        self.setup_ui()
        self.generate_cities()

    def apply_styles(self):
        # Налаштування сучасної теми
        style = ttk.Style()
        style.theme_use('clam')
        
        style.configure("TFrame", background="#EAECEE")
        style.configure("Panel.TFrame", background="#FFFFFF", relief="flat")
        
        style.configure("TLabel", background="#FFFFFF", font=("Segoe UI", 10), foreground="#2C3E50")
        style.configure("Title.TLabel", font=("Segoe UI", 12, "bold"), foreground="#2980B9")
        
        style.configure("TButton", font=("Segoe UI", 10, "bold"), padding=6)
        style.map("TButton",
                  background=[('active', '#3498DB'), ('!disabled', '#2980B9')],
                  foreground=[('!disabled', 'white')])
        
        style.configure("Accent.TButton", background="#27AE60")
        style.map("Accent.TButton", background=[('active', '#2ECC71')])

        style.configure("TNotebook", background="#EAECEE", borderwidth=0)
        style.configure("TNotebook.Tab", font=("Segoe UI", 10, "bold"), padding=[10, 5], background="#D5D8DC")
        style.map("TNotebook.Tab", background=[('selected', '#FFFFFF')], foreground=[('selected', '#2980B9')])

    def setup_ui(self):
        # --- Ліва панель (Біла картка) ---
        left_panel = ttk.Frame(self.root, style="Panel.TFrame")
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=15, pady=15)
        
        # Контейнер з відступами всередині картки
        control_frame = ttk.Frame(left_panel, style="Panel.TFrame")
        control_frame.pack(padx=20, pady=20, fill=tk.BOTH, expand=True)

        ttk.Label(control_frame, text="⚙ НАЛАШТУВАННЯ", style="Title.TLabel").pack(pady=(0, 15))

        ttk.Label(control_frame, text="Кількість міст:").pack(anchor=tk.W)
        self.var_cities = tk.IntVar(value=30)
        ttk.Spinbox(control_frame, from_=5, to=150, textvariable=self.var_cities, width=20, font=("Segoe UI", 10)).pack(pady=(0, 15))
        
        ttk.Button(control_frame, text="🔄 Згенерувати карту", command=self.generate_cities).pack(fill=tk.X, pady=(0, 25))

        ttk.Label(control_frame, text="Розмір популяції (ГА):").pack(anchor=tk.W)
        self.var_pop = tk.IntVar(value=50)
        ttk.Spinbox(control_frame, from_=10, to=500, textvariable=self.var_pop, width=20).pack(pady=(0, 10))

        ttk.Label(control_frame, text="Кількість поколінь (ГА):").pack(anchor=tk.W)
        self.var_gen = tk.IntVar(value=100)
        ttk.Spinbox(control_frame, from_=10, to=1000, textvariable=self.var_gen, width=20).pack(pady=(0, 10))

        ttk.Label(control_frame, text="Ймовірність мутації (0-1):").pack(anchor=tk.W)
        self.var_mut = tk.DoubleVar(value=0.15)
        ttk.Entry(control_frame, textvariable=self.var_mut, width=22).pack(pady=(0, 25))

        self.btn_run = ttk.Button(control_frame, text="▶ ЗАПУСТИТИ", style="Accent.TButton", command=self.run_optimization)
        self.btn_run.pack(fill=tk.X, pady=(0, 15))

        # Індикатор прогресу
        self.progress = ttk.Progressbar(control_frame, orient=tk.HORIZONTAL, length=100, mode='determinate')
        self.progress.pack(fill=tk.X, pady=5)
        
        self.lbl_result = tk.Label(control_frame, text="Очікування дій...", font=("Segoe UI", 10, "bold"), fg="#7F8C8D", bg="#FFFFFF")
        self.lbl_result.pack(pady=10)

        # --- Права панель з графіками ---
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH, padx=(0, 15), pady=15)
        
        self.tab_map = ttk.Frame(self.notebook, style="Panel.TFrame")
        self.tab_plot = ttk.Frame(self.notebook, style="Panel.TFrame")
        self.tab_stats = ttk.Frame(self.notebook, style="Panel.TFrame") # НОВА ВКЛАДКА
        
        self.notebook.add(self.tab_map, text="🗺 Карта маршруту")
        self.notebook.add(self.tab_plot, text="📈 Збіжність алгоритму")
        self.notebook.add(self.tab_stats, text="📊 Журнал та Статистика")
        
        # Візуалізація 1: Карта
        self.fig_map, (self.ax_map_ga, self.ax_map_hybrid) = plt.subplots(1, 2, figsize=(8, 5))
        self.fig_map.patch.set_facecolor('#FFFFFF')
        self.canvas_map = FigureCanvasTkAgg(self.fig_map, master=self.tab_map)
        self.canvas_map.get_tk_widget().pack(expand=True, fill=tk.BOTH, padx=10, pady=10)
        
        # Візуалізація 2: Збіжність
        self.fig_plot, self.ax_plot = plt.subplots(figsize=(8, 5))
        self.fig_plot.patch.set_facecolor('#FFFFFF')
        self.canvas_plot = FigureCanvasTkAgg(self.fig_plot, master=self.tab_plot)
        self.canvas_plot.get_tk_widget().pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

        # Візуалізація 3: Статистика (Новий модуль)
        self.setup_stats_tab()

    def setup_stats_tab(self):
        # Розділяємо вкладку на 2 частини: текст і графік
        top_frame = ttk.Frame(self.tab_stats, style="Panel.TFrame")
        top_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Текстовий лог
        log_frame = ttk.LabelFrame(top_frame, text=" Журнал подій ", style="Panel.TFrame")
        log_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        self.text_log = tk.Text(log_frame, width=40, font=("Consolas", 9), bg="#F4F6F7", fg="#2C3E50", relief="flat")
        self.text_log.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Гістограма
        self.fig_bar, self.ax_bar = plt.subplots(figsize=(4, 4))
        self.fig_bar.patch.set_facecolor('#FFFFFF')
        self.canvas_bar = FigureCanvasTkAgg(self.fig_bar, master=top_frame)
        self.canvas_bar.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))

    def log(self, message):
        self.text_log.insert(tk.END, message + "\n")
        self.text_log.see(tk.END)
        self.root.update()

    def generate_cities(self):
        try:
            n = self.var_cities.get()
            if n < 3: raise ValueError
        except:
            messagebox.showerror("Помилка", "Введіть коректну кількість міст (>= 3)")
            return

        self.cities_coords = np.random.rand(n, 2) * 100
        
        self.dist_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                self.dist_matrix[i, j] = np.linalg.norm(self.cities_coords[i] - self.cities_coords[j])
                
        self.ax_map_ga.clear()
        self.ax_map_hybrid.clear()
        
        self.ax_map_ga.scatter(self.cities_coords[:, 0], self.cities_coords[:, 1], c='#E74C3C', s=50, zorder=5)
        self.ax_map_hybrid.scatter(self.cities_coords[:, 0], self.cities_coords[:, 1], c='#E74C3C', s=50, zorder=5)
        
        self.ax_map_ga.set_title("Карта міст", color="#34495E")
        self.ax_map_hybrid.set_title("Очікування оптимізації...", color="#34495E")
        self.ax_map_ga.axis('off')
        self.ax_map_hybrid.axis('off')
        self.canvas_map.draw()
        
        self.lbl_result.config(text="Карта готова.", fg="#2980B9")
        self.text_log.delete(1.0, tk.END)
        self.log(f"[*] Згенеровано {n} міст.")
        
        # Очищення гістограми
        self.ax_bar.clear()
        self.canvas_bar.draw()

    def run_optimization(self):
        try:
            pop_size = self.var_pop.get()
            generations = self.var_gen.get()
            mut_rate = self.var_mut.get()
        except:
            messagebox.showerror("Помилка", "Перевірте параметри!")
            return

        self.btn_run.config(state=tk.DISABLED)
        self.lbl_result.config(text="Обчислення...", fg="#E67E22")
        self.progress['maximum'] = generations
        self.text_log.delete(1.0, tk.END)
        self.log(f"[*] Запуск оптимізації...\nМіст: {len(self.cities_coords)} | Популяція: {pop_size}")
        
        start_time = time.time()
        ga = GeneticAlgorithm(len(self.cities_coords), pop_size, mut_rate)
        
        # Оцінка початкового (випадкового) стану для порівняння
        initial_route = list(range(len(self.cities_coords)))
        initial_dist = sum(self.dist_matrix[initial_route[i], initial_route[i-1]] for i in range(len(initial_route)))
        self.log(f"[>] Відстань до оптимізації: {initial_dist:.2f}")

        history_dist = []
        best_ga_route = None
        best_ga_dist = float('inf')
        
        # --- ЕТАП 1: Генетичний алгоритм ---
        for gen in range(generations):
            route, dist = ga.evolve(self.dist_matrix)
            history_dist.append(dist)
            if dist < best_ga_dist:
                best_ga_dist = dist
                best_ga_route = route
                
            if gen % max(1, generations // 10) == 0:
                self.progress['value'] = gen
                self.log(f"   Покоління {gen}: {dist:.2f}")
                self.root.update()

        self.log(f"[+] ГА завершено. Краща відстань: {best_ga_dist:.2f}")

        # --- ЕТАП 2: Мережа Хопфілда ---
        self.lbl_result.config(text="Локальна оптимізація (Хопфілд)...")
        self.root.update()
        
        hybrid_route, hybrid_dist = HopfieldOptimizer.optimize(best_ga_route, self.dist_matrix)
        
        end_time = time.time()
        self.progress['value'] = generations
        self.log(f"[+] Хопфілд завершено. Відстань: {hybrid_dist:.2f}")
        self.log(f"[*] Витрачено часу: {end_time - start_time:.2f} сек.")
        self.log(f"\n[!] Загальне покращення: {initial_dist - hybrid_dist:.2f}")

        # --- ВІЗУАЛІЗАЦІЯ ---
        self.ax_map_ga.clear()
        self.ax_map_hybrid.clear()
        
        self.ax_map_ga.scatter(self.cities_coords[:, 0], self.cities_coords[:, 1], c='#E74C3C', s=40, zorder=5)
        self.ax_map_hybrid.scatter(self.cities_coords[:, 0], self.cities_coords[:, 1], c='#E74C3C', s=40, zorder=5)

        route_ga_coords = self.cities_coords[best_ga_route + [best_ga_route[0]]]
        self.ax_map_ga.plot(route_ga_coords[:, 0], route_ga_coords[:, 1], c='#3498DB', linewidth=1.5, alpha=0.8)
        self.ax_map_ga.set_title(f"Генетичний алгоритм\n({best_ga_dist:.1f})", color="#2980B9", fontsize=10)
        self.ax_map_ga.axis('off')
        
        route_hyb_coords = self.cities_coords[hybrid_route + [hybrid_route[0]]]
        self.ax_map_hybrid.plot(route_hyb_coords[:, 0], route_hyb_coords[:, 1], c='#27AE60', linewidth=2)
        self.ax_map_hybrid.set_title(f"ГА + Мережа Хопфілда\n({hybrid_dist:.1f})", color="#27AE60", fontsize=10)
        self.ax_map_hybrid.axis('off')
        
        self.canvas_map.draw()
        
        # Графік збіжності
        self.ax_plot.clear()
        self.ax_plot.plot(range(generations), history_dist, label='ГА (Глобальний пошук)', color='#3498DB', linewidth=2)
        self.ax_plot.axhline(y=hybrid_dist, color='#27AE60', linestyle='--', linewidth=2, label='Хопфілд (Локальний мінімум)')
        self.ax_plot.set_title("Графік зменшення довжини маршруту", color="#34495E")
        self.ax_plot.set_xlabel("Покоління")
        self.ax_plot.set_ylabel("Довжина маршруту")
        self.ax_plot.legend()
        self.ax_plot.grid(True, linestyle=':', alpha=0.7)
        self.canvas_plot.draw()

        # Гістограма у вкладці "Статистика"
        self.ax_bar.clear()
        labels = ['Випадково', 'ГА', 'ГА+Хопфілд']
        values = [initial_dist, best_ga_dist, hybrid_dist]
        colors = ['#95A5A6', '#3498DB', '#27AE60']
        
        bars = self.ax_bar.bar(labels, values, color=colors)
        self.ax_bar.set_title("Ефективність методів", color="#34495E")
        self.ax_bar.set_ylabel("Довжина маршруту")
        
        # Додаємо цифри над колонками
        for bar in bars:
            yval = bar.get_height()
            self.ax_bar.text(bar.get_x() + bar.get_width()/2, yval + (max(values)*0.02), f'{yval:.0f}', ha='center', va='bottom', fontsize=9)
            
        self.fig_bar.tight_layout()
        self.canvas_bar.draw()

        self.btn_run.config(state=tk.NORMAL)
        self.lbl_result.config(text=f"✅ Готово! Зменшено на {initial_dist - hybrid_dist:.0f}", fg="#27AE60")

if __name__ == "__main__":
    root = tk.Tk()
    app = TSPApp(root)
    root.mainloop()
