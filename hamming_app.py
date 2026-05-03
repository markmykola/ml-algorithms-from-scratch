import numpy as np
import tkinter as tk
from tkinter import ttk
import random
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score, f1_score
import warnings

warnings.filterwarnings("ignore")

# =====================================================================
# ЯДРО: Нейронна мережа Хеммінга
# =====================================================================
class HammingNetwork:
    def __init__(self, patterns, labels):
        self.patterns = np.array(patterns)
        self.labels = labels
        self.M = self.patterns.shape[0]  
        self.n = self.patterns.shape[1]  
        
        self.weights = self.patterns / 2.0
        self.bias = self.n / 2.0
        self.epsilon = 1.0 / self.M - 0.01

    def recognize(self, input_vector):
        y = np.dot(self.weights, input_vector) + self.bias
        iteration = 0
        while np.count_nonzero(y > 0) > 1:
            new_y = np.zeros(self.M)
            for j in range(self.M):
                sum_others = np.sum(y) - y[j]
                new_y[j] = max(0, y[j] - self.epsilon * sum_others)
            y = new_y
            iteration += 1
            if iteration > 1000:
                break
        winner_index = np.argmax(y)
        return self.labels[winner_index], iteration

# =====================================================================
# ДОПОМІЖНІ ФУНКЦІЇ: Робота з шумом та Фільтрація
# =====================================================================
class Preprocessor:
    @staticmethod
    def add_noise(pattern, noise_level):
        noisy_pattern = pattern.copy()
        num_noisy_pixels = int(len(pattern) * noise_level)
        indices = random.sample(range(len(pattern)), num_noisy_pixels)
        for idx in indices:
            noisy_pattern[idx] = -noisy_pattern[idx]
        return noisy_pattern

    @staticmethod
    def apply_majority_filter(vector, shape=(8, 8)):
        img = vector.reshape(shape)
        padded = np.pad(img, pad_width=1, mode='constant', constant_values=-1)
        filtered_img = np.copy(img)
        
        for i in range(shape[0]):
            for j in range(shape[1]):
                window = padded[i:i+3, j:j+3]
                center = img[i, j]
                
                white_neighbors = np.sum(window == -1)
                black_neighbors = np.sum(window == 1)
                
                if center == 1 and white_neighbors >= 7:
                    filtered_img[i, j] = -1
                elif center == -1 and black_neighbors >= 7:
                    filtered_img[i, j] = 1
                    
        return filtered_img.flatten()

# =====================================================================
# ДАНІ: Завантаження та підготовка
# =====================================================================
class DataManager:
    @staticmethod
    def get_alphabet_patterns():
        A = [-1, 1, 1, 1, -1,   1, -1, -1, -1, 1,   1, 1, 1, 1, 1,   1, -1, -1, -1, 1,   1, -1, -1, -1, 1]
        C = [-1, 1, 1, 1, 1,    1, -1, -1, -1, -1,  1, -1, -1, -1, -1, 1, -1, -1, -1, -1,  -1, 1, 1, 1, 1]
        I = [1, 1, 1, 1, 1,     -1, -1, 1, -1, -1,  -1, -1, 1, -1, -1, -1, -1, 1, -1, -1,  1, 1, 1, 1, 1]
        O = [-1, 1, 1, 1, -1,   1, -1, -1, -1, 1,   1, -1, -1, -1, 1,  1, -1, -1, -1, 1,   -1, 1, 1, 1, -1]
        X = [1, -1, -1, -1, 1,  -1, 1, -1, 1, -1,   -1, -1, 1, -1, -1, -1, 1, -1, 1, -1,   1, -1, -1, -1, 1]
        return np.array([A, C, I, O, X]), ["A", "C", "I", "O", "X"]

    @staticmethod
    def load_real_dataset():
        digits = load_digits()
        X = digits.data
        y = digits.target
        
        # --- НОВЕ: ВИВІД ІНФОРМАЦІЇ У КОНСОЛЬ ---
        print("\n" + "="*50)
        print("📊 ІНФОРМАЦІЯ ПРО ЗАВАНТАЖЕНИЙ ДАТАСЕТ")
        print("="*50)
        print(f"🔹 Загальна кількість зображень : {len(X)}")
        print(f"🔹 Кількість класів             : {len(np.unique(y))} (цифри від 0 до 9)")
        print(f"🔹 Розмірність одного вектора   : 8x8 пікселів ({X.shape[1]} ознак)")
        print("="*50 + "\n")
        # ----------------------------------------
        
        patterns = []
        labels = []
        OPTIMAL_THRESHOLD = 6 
        
        for i in range(10):
            mean_image = np.mean(X[y == i], axis=0)
            ideal_pattern = np.where(mean_image > OPTIMAL_THRESHOLD, 1, -1)
            patterns.append(ideal_pattern)
            labels.append(str(i))
                
        X_binary = np.where(X > OPTIMAL_THRESHOLD, 1, -1)
        return np.array(patterns), labels, X_binary, y

# =====================================================================
# ГРАФІЧНИЙ ІНТЕРФЕЙС (GUI) - Оптимізовано для Mac
# =====================================================================
class HammingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Нейронна Мережа Хеммінга")
        # Збільшено вікно для нормального відображення на Mac
        self.root.geometry("950x850") 

        alpha_p, alpha_l = DataManager.get_alphabet_patterns()
        self.net_alpha = HammingNetwork(alpha_p, alpha_l)
        
        digit_p, digit_l, self.test_X, self.test_y = DataManager.load_real_dataset()
        self.net_digits = HammingNetwork(digit_p, digit_l)

        self.notebook = ttk.Notebook(root)
        self.notebook.pack(expand=True, fill='both')

        self.tab_draw = ttk.Frame(self.notebook)
        self.tab_dataset = ttk.Frame(self.notebook)
        self.tab_metrics = ttk.Frame(self.notebook)
        
        self.notebook.add(self.tab_draw, text='Модуль 1: Малювання (5x5)')
        self.notebook.add(self.tab_dataset, text='Модуль 2: Реальнi цифри (8x8)')
        self.notebook.add(self.tab_metrics, text='Модуль 3: Графіки похибок')

        self.setup_drawing_tab()
        self.setup_dataset_tab()
        self.setup_metrics_tab()

    # --- МОДУЛЬ 1: Малювання ---
    def setup_drawing_tab(self):
        self.grid_size = 5
        self.state = [-1] * 25
        self.buttons = []

        tk.Label(self.tab_draw, text="Намалюйте одну з літер: A, C, I, O, X", font=("Arial", 14)).pack(pady=15)
        grid_frame = tk.Frame(self.tab_draw)
        grid_frame.pack()

        for i in range(self.grid_size):
            row_btns = []
            for j in range(self.grid_size):
                idx = i * self.grid_size + j
                # Fix for Mac: Використовуємо Label замість Button
                lbl = tk.Label(grid_frame, width=4, height=2, bg="white", relief="raised", borderwidth=2)
                lbl.bind("<Button-1>", lambda event, idx=idx: self.toggle_pixel(idx))
                lbl.grid(row=i, column=j, padx=2, pady=2)
                row_btns.append(lbl)
            self.buttons.append(row_btns)

        btn_frame = tk.Frame(self.tab_draw)
        btn_frame.pack(pady=20)
        tk.Button(btn_frame, text="Розпізнати", font=("Arial", 12, "bold"), command=self.recognize_drawing).grid(row=0, column=0, padx=10)
        tk.Button(btn_frame, text="Очистити", font=("Arial", 12), command=self.clear_grid).grid(row=0, column=1, padx=10)

        self.lbl_result_draw = tk.Label(self.tab_draw, text="Результат: ...", font=("Arial", 16, "bold"), fg="blue")
        self.lbl_result_draw.pack(pady=20)

    def toggle_pixel(self, idx):
        row, col = divmod(idx, self.grid_size)
        self.state[idx] = 1 if self.state[idx] == -1 else -1
        new_color = "black" if self.state[idx] == 1 else "white"
        self.buttons[row][col].config(bg=new_color)

    def clear_grid(self):
        self.state = [-1] * 25
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                self.buttons[i][j].config(bg="white")
        self.lbl_result_draw.config(text="Результат: ...")

    def recognize_drawing(self):
        label, iters = self.net_alpha.recognize(np.array(self.state))
        self.lbl_result_draw.config(text=f"Розпізнано: {label} (за {iters} ітерацій)")

    # --- МОДУЛЬ 2: Робота з датасетом ---
    def setup_dataset_tab(self):
        tk.Label(self.tab_dataset, text="Етапи: Оригінал -> Зашумлене -> Відфільтроване", font=("Arial", 14, "bold")).pack(pady=10)
        
        control_frame = tk.Frame(self.tab_dataset)
        control_frame.pack(pady=10)
        
        tk.Button(control_frame, text="Випадкова цифра", font=("Arial", 12), command=self.load_random_digit).grid(row=0, column=0, padx=10)
        
        tk.Label(control_frame, text="Додати шум:", font=("Arial", 12)).grid(row=0, column=1, padx=5)
        self.noise_slider = tk.Scale(control_frame, from_=0, to=40, orient=tk.HORIZONTAL, length=200)
        self.noise_slider.grid(row=0, column=2, padx=10)
        
        tk.Button(self.tab_dataset, text="Розпізнати", font=("Arial", 14, "bold"), bg="lightblue", command=self.recognize_dataset_image).pack(pady=5)

        # Fix for Mac: Перенесли текст результату НАГОРУ
        self.lbl_result_dataset = tk.Label(self.tab_dataset, text="Очікування...", font=("Arial", 18, "bold"), fg="green")
        self.lbl_result_dataset.pack(pady=15)

        # Трохи зменшили висоту графіка
        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(1, 3, figsize=(9, 2.8))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.tab_dataset)
        self.canvas.get_tk_widget().pack(expand=True, fill='both')
        
        self.current_img_vector = None
        self.current_true_label = None
        self.load_random_digit()

    def load_random_digit(self):
        idx = random.randint(0, len(self.test_X) - 1)
        self.current_img_vector = self.test_X[idx]
        self.current_true_label = self.test_y[idx]
        
        self.ax1.clear()
        self.ax1.imshow(self.current_img_vector.reshape(8, 8), cmap='gray')
        self.ax1.set_title(f"Оригінал: {self.current_true_label}")
        self.ax1.axis('off')
        
        self.ax2.clear(), self.ax2.axis('off')
        self.ax3.clear(), self.ax3.axis('off')
        self.canvas.draw()
        self.lbl_result_dataset.config(text="Зображення завантажено.", fg="black")

    def recognize_dataset_image(self):
        if self.current_img_vector is None: return
        
        noise_level = self.noise_slider.get() / 100.0
        noisy_vector = Preprocessor.add_noise(self.current_img_vector, noise_level)
        cleaned_vector = Preprocessor.apply_majority_filter(noisy_vector)
        
        self.ax2.clear()
        self.ax2.imshow(noisy_vector.reshape(8, 8), cmap='gray')
        self.ax2.set_title(f"Шум {int(noise_level*100)}%")
        self.ax2.axis('off')
        
        self.ax3.clear()
        self.ax3.imshow(cleaned_vector.reshape(8, 8), cmap='gray')
        self.ax3.set_title("Після фільтрації")
        self.ax3.axis('off')
        
        self.canvas.draw()
        
        predicted, _ = self.net_digits.recognize(cleaned_vector)
        
        color = "green" if predicted == str(self.current_true_label) else "red"
        self.lbl_result_dataset.config(text=f"Розпізнано: {predicted} (Справжня: {self.current_true_label})", fg=color)

    # --- МОДУЛЬ 3: Метрики похибки ---
    def setup_metrics_tab(self):
        tk.Label(self.tab_metrics, text="Метрики оцінки алгоритму (з урахуванням фільтрації)", font=("Arial", 14, "bold")).pack(pady=15)
        tk.Button(self.tab_metrics, text="Побудувати графік точності", font=("Arial", 12, "bold"), command=self.run_metrics).pack(pady=5)
        
        self.plot_frame = tk.Frame(self.tab_metrics)
        self.plot_frame.pack(fill="both", expand=True)

    def run_metrics(self):
        for widget in self.plot_frame.winfo_children(): widget.destroy()
        
        noise_levels = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
        accs, f1s, errs = [], [], []
        
        indices = random.sample(range(len(self.test_X)), 200)
        
        for noise in noise_levels:
            y_true, y_pred = [], []
            for idx in indices:
                noisy_in = Preprocessor.add_noise(self.test_X[idx], noise)
                cleaned_in = Preprocessor.apply_majority_filter(noisy_in)
                
                pred, _ = self.net_digits.recognize(cleaned_in)
                y_true.append(str(self.test_y[idx]))
                y_pred.append(pred)
                
            acc = accuracy_score(y_true, y_pred)
            f1s.append(f1_score(y_true, y_pred, average='macro', zero_division=0))
            accs.append(acc)
            errs.append(1.0 - acc)
            
        fig, ax = plt.subplots(figsize=(8, 5))
        n_perc = [int(n * 100) for n in noise_levels]
        ax.plot(n_perc, [a * 100 for a in accs], marker='o', label="Accuracy (%)")
        ax.plot(n_perc, [f * 100 for f in f1s], marker='^', label="F1-Score (%)")
        ax.plot(n_perc, [e * 100 for e in errs], marker='s', label="Похибка (%)", color='red')
        ax.set_title("Вплив шуму на розпізнавання (з медіанним фільтром)")
        ax.set_xlabel("Рівень шуму (%)")
        ax.set_ylabel("Відсотки (%)")
        ax.legend()
        ax.grid(True)

        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(expand=True, fill="both")

if __name__ == "__main__":
    root = tk.Tk()
    app = HammingApp(root)
    root.mainloop()
