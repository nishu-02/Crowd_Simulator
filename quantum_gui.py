import tkinter as tk
from tkinter import ttk  
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from quantum_systems import *
import plotly.graph_objects as go
from ipywidgets import interact

class QuantumVisualizer:
    def __init__(self, master):
        self.master = master
        master.title("Quantum Physics Explorer")
        
        # Notebook-style tabs
        self.tabs = tk.ttk.Notebook(master)
        self.tab1 = tk.Frame(self.tabs)
        self.tab2 = tk.Frame(self.tabs)
        self.tabs.add(self.tab1, text='Particle in Box')
        self.tabs.add(self.tab2, text='Harmonic Oscillator')
        self.tabs.pack(expand=1, fill="both")
        
        # Tab 1: 3D Particle in Box
        self.fig1 = plt.figure(figsize=(8,6))
        self.ax1 = self.fig1.add_subplot(111, projection='3d')
        self.canvas1 = FigureCanvasTkAgg(self.fig1, self.tab1)
        self.canvas1.get_tk_widget().pack()
        
        # Tab 2: Interactive Potential Wells
        self.fig2 = go.FigureWidget()
        self.plotly_canvas = go.FigureWidget(self.fig2)
        # Add plotly components here
        
        # Unified controls
        self.controls = tk.Frame(master)
        self.controls.pack()
        
        # Quantum system selector
        self.system_var = tk.StringVar(value='box')
        tk.Radiobutton(self.controls, text="Particle in Box", variable=self.system_var, value='box').pack(side='left')
        tk.Radiobutton(self.controls, text="Harmonic Oscillator", variable=self.system_var, value='ho').pack(side='left')
        
        # Dynamic parameter controls
        self.param_frame = tk.Frame(self.controls)
        self.param_frame.pack()
        self.create_sliders()
        
        # Real-time equations
        self.eq_label = tk.Label(master, text="", font=('Cambria Math', 14))
        self.eq_label.pack()
        
        self.update_plots()

    def create_sliders(self):
        # Particle in Box controls
        self.n_slider = tk.Scale(self.param_frame, from_=1, to=5, orient='horizontal', label="Quantum Number (n)")
        self.L_slider = tk.Scale(self.param_frame, from_=1, to=3, resolution=0.1, orient='horizontal', label="Box Length (Å)")
        
        # Harmonic Oscillator controls
        self.omega_slider = tk.Scale(self.param_frame, from_=1e12, to=1e15, orient='horizontal', label="Angular Frequency (rad/s)")
        
        self.n_slider.pack(side='left')
        self.L_slider.pack(side='left')
        self.omega_slider.pack(side='left')

    def update_plots(self):
        system = self.system_var.get()
        
        if system == 'box':
            n = self.n_slider.get()
            L = self.L_slider.get()
            x = np.linspace(0, L, 1000)
            psi, prob = particle_in_box(n, L, x)
            
            # 3D visualization
            self.ax1.clear()
            X, Y = np.meshgrid(x, x)
            Z = np.outer(psi, psi)
            self.ax1.plot_surface(X, Y, Z, cmap='viridis')
            self.ax1.set_title(f"3D Probability Density (n={n})")
            
            # Update equation display
            self.eq_label.config(text=r"$\psi_n(x) = \sqrt{\frac{2}{L}} \sin\left(\frac{n\pi x}{L}\right)$")
            
        self.canvas1.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = QuantumVisualizer(root)
    root.mainloop()
