import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle, Circle, Wedge
import matplotlib.colors as mcolors
from matplotlib.widgets import Slider, Button, RadioButtons, CheckButtons
import argparse
import time
from crowd_flow_simulator import CrowdFlowSimulator

class CrowdSimulationGUI:
    def __init__(self, simulator):
        """Initialize the GUI with a simulator instance"""
        self.simulator = simulator
        self.paused = False
        self.step_count = 0
        self.show_velocities = True
        self.show_density = False
        self.show_personal_space = False
        self.density_data = None
        self.colormap = plt.cm.viridis
        self.agent_colors = self._generate_agent_colors()
        
        # Set up the figure and axes
        self.fig = plt.figure(figsize=(14, 9))
        self.fig.canvas.manager.set_window_title('Advanced Crowd Flow Simulator')
        
        # Main plotting area
        self.ax_main = self.fig.add_axes([0.1, 0.25, 0.8, 0.65])
        
        # Set up control panel area
        self.setup_controls()
        
        # Initialize the stats area
        self.stats_text = self.ax_main.text(0.02, 0.98, '', 
                                          transform=self.ax_main.transAxes,
                                          verticalalignment='top',
                                          bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        # Create heat map for density visualization
        self.grid_size = 50
        self.x_grid = np.linspace(0, simulator.arena_width, self.grid_size)
        self.y_grid = np.linspace(0, simulator.arena_height, self.grid_size)
        self.X, self.Y = np.meshgrid(self.x_grid, self.y_grid)
        self.density_plot = None
        
        # Set up the animation
        self.setup_animation()
        
    def _generate_agent_colors(self):
        """Generate consistent colors for agents"""
        # Use a colormap to assign colors based on agent index
        cmap = plt.cm.tab20
        colors = [cmap(i % 20) for i in range(self.simulator.num_agents)]
        return colors
        
    def setup_controls(self):
        """Set up the control panel"""
        # Buttons for simulation control
        ax_pause = self.fig.add_axes([0.1, 0.1, 0.15, 0.05])
        self.btn_pause = Button(ax_pause, 'Pause/Resume')
        self.btn_pause.on_clicked(self.toggle_pause)
        
        ax_reset = self.fig.add_axes([0.3, 0.1, 0.15, 0.05])
        self.btn_reset = Button(ax_reset, 'Reset Simulation')
        self.btn_reset.on_clicked(self.reset_simulation)
        
        # Slider for simulation speed
        ax_speed = self.fig.add_axes([0.55, 0.1, 0.35, 0.03])
        self.slider_speed = Slider(ax_speed, 'Speed', 10, 200, valinit=50)
        
        # Checkboxes for display options
        ax_display = self.fig.add_axes([0.1, 0.02, 0.3, 0.05])
        self.check_display = CheckButtons(ax_display, ['Velocities', 'Density', 'Personal Space'], 
                                         [self.show_velocities, self.show_density, self.show_personal_space])
        self.check_display.on_clicked(self.update_display_options)
        
        # Exit width slider (if exit is present)
        if self.simulator.has_exit:
            ax_exit = self.fig.add_axes([0.55, 0.05, 0.35, 0.03])
            self.slider_exit = Slider(ax_exit, 'Exit Width', 1.0, 10.0, valinit=self.simulator.exit_width)
            self.slider_exit.on_changed(self.update_exit_width)
    
    def update_display_options(self, label):
        """Update display options based on checkbox selection"""
        if label == 'Velocities':
            self.show_velocities = not self.show_velocities
        elif label == 'Density':
            self.show_density = not self.show_density
        elif label == 'Personal Space':
            self.show_personal_space = not self.show_personal_space
    
    def update_exit_width(self, val):
        """Update exit width based on slider"""
        if hasattr(self.simulator, 'exit_width'):
            self.simulator.exit_width = val
    
    def toggle_pause(self, event):
        """Toggle pause/resume of the simulation"""
        self.paused = not self.paused
    
    def reset_simulation(self, event):
        """Reset the simulation"""
        # Recreate the simulator with the same parameters
        self.simulator = CrowdFlowSimulator(
            num_agents=self.simulator.num_agents,
            arena_width=self.simulator.arena_width,
            arena_height=self.simulator.arena_height,
            time_steps=self.simulator.time_steps,
            dt=self.simulator.dt,
            has_exit=self.simulator.has_exit,
            has_obstacles=self.simulator.has_obstacles
        )
        self.step_count = 0
        # Generate new agent colors
        self.agent_colors = self._generate_agent_colors()
    
    def calculate_density_field(self):
        """Calculate density field from agent positions"""
        active_positions = self.simulator.positions[self.simulator.active_agents]
        
        # Create empty density field
        density = np.zeros((self.grid_size, self.grid_size))
        
        if len(active_positions) > 0:
            # Create a 2D histogram of agent positions
            hist, _, _ = np.histogram2d(
                active_positions[:, 0], active_positions[:, 1],
                bins=[self.grid_size, self.grid_size],
                range=[[0, self.simulator.arena_width], [0, self.simulator.arena_height]]
            )
            
            # Smooth the density field
            from scipy.ndimage import gaussian_filter
            density = gaussian_filter(hist, sigma=1.0)
        
        return density
    
    def setup_animation(self):
        """Set up the animation elements"""
        # Initialize the plot
        self.ax_main.set_xlim(0, self.simulator.arena_width)
        self.ax_main.set_ylim(0, self.simulator.arena_height)
        self.ax_main.set_title('Crowd Flow Simulation')
        self.ax_main.set_xlabel('X Position')
        self.ax_main.set_ylabel('Y Position')
        self.ax_main.grid(True, alpha=0.3)
        
        # Create collections for various graphical elements
        self.agent_circles = []
        self.velocity_arrows = []
        self.personal_space_circles = []
        
        # Draw obstacles
        for obstacle in self.simulator.obstacles:
            pos = obstacle["position"]
            width = obstacle["width"]
            height = obstacle["height"]
            rect = Rectangle((pos[0]-width/2, pos[1]-height/2), width, height,
                           color='gray', alpha=0.7)
            self.ax_main.add_patch(rect)
        
        # Draw exit if present
        if self.simulator.has_exit:
            exit_pos = self.simulator.exit_position
            exit_width = self.simulator.exit_width
            exit_y = exit_pos[1] - exit_width/2
            exit_rect = Rectangle((self.simulator.arena_width-0.2, exit_y), 0.2, exit_width,
                                color='green', alpha=0.7)
            self.ax_main.add_patch(exit_rect)
            self.ax_main.text(self.simulator.arena_width-2, exit_pos[1], 'EXIT',
                            ha='right', va='center', color='green', fontsize=12)
        
        # Create animation
        self.anim = FuncAnimation(self.fig, self.update_frame, 
                                 frames=self.simulator.time_steps,
                                 interval=20, blit=False)
    
    def update_frame(self, frame):
        """Update the animation frame"""
        # Clear the plot
        self.ax_main.cla()
        self.ax_main.set_xlim(0, self.simulator.arena_width)
        self.ax_main.set_ylim(0, self.simulator.arena_height)
        self.ax_main.set_title(f'Crowd Flow Simulation - Step {self.step_count}')
        self.ax_main.set_xlabel('X Position')
        self.ax_main.set_ylabel('Y Position')
        self.ax_main.grid(True, alpha=0.3)
        
        # Run simulation step if not paused
        if not self.paused:
            self.simulator.step()
            self.step_count += 1
        
        # Get simulation data
        active_indices = np.where(self.simulator.active_agents)[0]
        active_positions = self.simulator.positions[self.simulator.active_agents]
        active_velocities = self.simulator.velocities[self.simulator.active_agents]
        
        # Show density field if selected
        if self.show_density:
            density = self.calculate_density_field()
            self.density_plot = self.ax_main.contourf(
                self.X, self.Y, density, cmap='viridis', alpha=0.5, levels=15
            )
        
        # Draw agents
        for i, pos in enumerate(active_positions):
            agent_idx = active_indices[i]
            color = self.agent_colors[agent_idx]
            
            # Draw agent circle
            circle = Circle(pos, self.simulator.agent_radius, color=color, alpha=0.7)
            self.ax_main.add_patch(circle)
            
            # Draw velocity vector if enabled
            if self.show_velocities:
                vel = active_velocities[i]
                speed = np.linalg.norm(vel)
                
                # Scale arrow length based on speed
                arrow_length = min(speed, 3.0)
                if speed > 0:
                    direction = vel / speed
                    self.ax_main.arrow(
                        pos[0], pos[1],
                        direction[0] * arrow_length, direction[1] * arrow_length,
                        head_width=0.3, head_length=0.3, fc='red', ec='red', alpha=0.7
                    )
            
            # Draw personal space if enabled
            if self.show_personal_space:
                personal_space = Circle(pos, 1.5, color=color, fill=False, 
                                      linestyle='--', alpha=0.3)
                self.ax_main.add_patch(personal_space)
        
        # Draw obstacles
        for obstacle in self.simulator.obstacles:
            pos = obstacle["position"]
            width = obstacle["width"]
            height = obstacle["height"]
            rect = Rectangle((pos[0]-width/2, pos[1]-height/2), width, height,
                           color='gray', alpha=0.7)
            self.ax_main.add_patch(rect)
        
        # Draw exit if present
        if self.simulator.has_exit:
            exit_pos = self.simulator.exit_position
            exit_width = self.simulator.exit_width  # Use current slider value
            exit_y = exit_pos[1] - exit_width/2
            exit_rect = Rectangle((self.simulator.arena_width-0.2, exit_y), 0.2, exit_width,
                                color='green', alpha=0.7)
            self.ax_main.add_patch(exit_rect)
            self.ax_main.text(self.simulator.arena_width-2, exit_pos[1], 'EXIT',
                            ha='right', va='center', color='green', fontsize=12)
        
        # Update statistics
        active_count = np.sum(self.simulator.active_agents)
        avg_speed = np.mean(np.linalg.norm(active_velocities, axis=1)) if len(active_velocities) > 0 else 0
        evacuation_ratio = 1 - (active_count / self.simulator.num_agents)
        
        stats_text = (
            f'Step: {self.step_count}/{self.simulator.time_steps}\n'
            f'Active Agents: {active_count}/{self.simulator.num_agents}\n'
            f'Evacuation: {evacuation_ratio:.1%}\n'
            f'Avg Speed: {avg_speed:.2f}'
        )
        self.stats_text = self.ax_main.text(0.02, 0.98, stats_text, 
                                         transform=self.ax_main.transAxes,
                                         verticalalignment='top',
                                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        # Check if simulation should end
        if active_count == 0 and self.simulator.has_exit:
            self.paused = True
            self.ax_main.text(0.5, 0.5, 'Evacuation Complete!', 
                           transform=self.ax_main.transAxes, 
                           ha='center', va='center', fontsize=24,
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        
        # Adjust animation speed based on slider
        self.anim.event_source.interval = 200 - self.slider_speed.val
        
        return []
    
    def run(self):
        """Run the GUI"""
        plt.tight_layout()
        plt.show()

def run_crowd_simulation_with_gui():
    """Run crowd simulation with enhanced GUI"""
    parser = argparse.ArgumentParser(description='Crowd Flow Simulator with Enhanced GUI')
    parser.add_argument('--num_agents', type=int, default=100, help='Number of agents')
    parser.add_argument('--arena_width', type=float, default=50, help='Width of arena')
    parser.add_argument('--arena_height', type=float, default=50, help='Height of arena')
    parser.add_argument('--time_steps', type=int, default=1000, help='Maximum simulation steps')
    parser.add_argument('--dt', type=float, default=0.1, help='Time step size')
    parser.add_argument('--no_exit', action='store_false', dest='has_exit', help='Disable exit')
    parser.add_argument('--obstacles', action='store_true', dest='has_obstacles', help='Add obstacles')
    args = parser.parse_args()
    
    # Create simulator
    simulator = CrowdFlowSimulator(
        num_agents=args.num_agents,
        arena_width=args.arena_width,
        arena_height=args.arena_height,
        time_steps=args.time_steps,
        dt=args.dt,
        has_exit=args.has_exit,
        has_obstacles=args.has_obstacles
    )
    
    # Create and run GUI
    gui = CrowdSimulationGUI(simulator)
    gui.run()

if __name__ == "__main__":
    run_crowd_simulation_with_gui()