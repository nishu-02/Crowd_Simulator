import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from physics_calculations import particle_in_box

class QuantumVisualizer:
    def __init__(self, master):
        self.master = master
        self.n = 1  # Initial quantum number
        self.L = 1  # Initial box length
        
        # Create figure
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(6,6))
        
        # GUI controls
        self.controls = tk.Frame(master)
        self.controls.pack()
        
        # Quantum number slider
        tk.Label(self.controls, text="Quantum Number (n)").pack()
        self.n_slider = tk.Scale(self.controls, from_=1, to=5, orient='horizontal', command=self.update)
        self.n_slider.pack()
        
        # Box length slider
        tk.Label(self.controls, text="Box Length (L)").pack()
        self.L_slider = tk.Scale(self.controls, from_=1, to=3, resolution=0.1, orient='horizontal', command=self.update)
        self.L_slider.pack()
        
        # Matplotlib canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=master)
        self.canvas.get_tk_widget().pack()
        self.update()

    def update(self, event=None):
        x = np.linspace(0, self.L_slider.get(), 1000)
        psi, prob = particle_in_box(self.n_slider.get(), self.L_slider.get(), x)
        
        self.ax1.clear()
        self.ax1.plot(x, psi, 'b')
        self.ax1.set_title("Wavefunction ψ(x)")
        self.ax1.set_ylabel("Amplitude")
        
        self.ax2.clear()
        self.ax2.plot(x, prob, 'r')
        self.ax2.set_title("Probability Density |ψ(x)|²")
        self.ax2.set_xlabel("Position (x)")
        self.ax2.set_ylabel("Probability")
        
        self.fig.tight_layout()
        self.canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = QuantumVisualizer(root)
    root.mainloop()
