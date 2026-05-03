# 🧠 Machine Learning Algorithms From Scratch

Цей репозиторій містить власні імплементації класичних алгоритмів машинного навчання та нейронних мереж, написані "з нуля" (from scratch) з використанням лінійної алгебри на базі `NumPy`. 

Проєкти демонструють глибоке розуміння архітектури алгоритмів без використання високорівневих фреймворків (таких як TensorFlow чи PyTorch), а також навички створення інтерактивних візуалізацій та графічних інтерфейсів (GUI).

---

## 📂 Проєкт 1: Нейронна мережа Хеммінга (Optical Character Recognition)
**Файл:** `hamming_app.py`

Десктопний застосунок для розпізнавання образів та зашумлених рукописних цифр на базі архітектури мережі Хеммінга. 

### 🚀 Ключові особливості:
* **Custom Neural Network:** Повністю власна реалізація алгоритму Хеммінга (ініціалізація ваг, розрахунок відстаней, ітеративний пошук переможця).
* **Data Preprocessing:** Вбудований генератор шуму та алгоритми попереднього очищення (мажоритарні/медіанні фільтри).
* **Interactive GUI:** Інтерфейс для малювання символів (5x5) та тестування на реальному датасеті цифр (8x8).
* **Metrics & Analytics:** Модуль оцінки якості моделі, який будує графіки залежності `Accuracy` та `F1-Score` від рівня зашумленості даних.

### 📸 Демонстрація роботи
<p align="center">
  <img width="800" alt="Hamming Draw" src="https://github.com/user-attachments/assets/ea70585d-5fbe-410a-bfd6-d9fe525ca59e" />
  <br><br>
  <img width="800" alt="Hamming Noise" src="https://github.com/user-attachments/assets/afe2b6e7-0818-4434-a55a-ed698b39c580" />
  <br><br>
  <img width="800" alt="Hamming Analytics" src="https://github.com/user-attachments/assets/18c2e2a8-3c77-49e1-801d-749e94b3fcb0" />
</p>

---

## 📂 Проєкт 2: Гібридний оптимізатор маршрутів (TSP)
**Файл:** `HopfieldOptimizer.py`

Система для вирішення задачі комівояжера (Traveling Salesman Problem) у логістиці, яка використовує гібридний підхід для знаходження найкоротшого маршруту.

### 🚀 Ключові особливості:
* **Глобальний пошук (Genetic Algorithm):** Реалізація еволюційного підходу з кросовером, мутаціями та селекцією популяцій.
* **Локальна оптимізація (Hopfield Network):** Використання принципів мережі Хопфілда для фінального "полірування" та знаходження локальних мінімумів маршруту.
* **Real-time візуалізація:** Динамічна побудова маршрутів на карті за допомогою `Matplotlib` та `Tkinter`.
* **Статистика:** Побудова графіків збіжності алгоритму та гістограм ефективності методів.

### 📸 Демонстрація роботи
<p align="center">
  <img width="800" alt="TSP Optimization" src="https://github.com/user-attachments/assets/388f905a-83f8-441a-baa7-4afe7da6d2da" />
  <br><br>
  <img width="800" alt="TSP Map" src="https://github.com/user-attachments/assets/ab5e150b-733d-4722-a3c6-ce0c79e4681d" />
  <br><br>
  <img width="800" alt="TSP Analytics" src="https://github.com/user-attachments/assets/f72627ee-e03a-4020-b69b-a8efc30dcc8f" />
</p>

---

## 🛠️ Технологічний стек
* **Core:** Python 3
* **Math & Logic:** NumPy, SciPy
* **Data & Machine Learning:** scikit-learn (лише для завантаження датасету та метрик)
* **GUI & Visualization:** Tkinter, Matplotlib, Seaborn

---

## ⚙️ Як запустити локально

1. Клонуйте репозиторій:
   ```bash
   git clone [https://github.com/markmykola/ml-algorithms-from-scratch.git](https://github.com/markmykola/ml-algorithms-from-scratch.git)
   cd ml-algorithms-from-scratch
