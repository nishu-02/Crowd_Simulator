import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle, Circle, Wedge
import matplotlib.colors as mcolors
from matplotlib.widgets import Slider, Button, RadioButtons, CheckButtons
import argparse
import time
from crowd_flow_simulator import CrowdFlowSimulator
from crowd_flow_analyzer import CrowdFlowAnalyzer

class CrowdSimulationGUI:
    def __init__(self, simulator, analyzer=None):
        """Initialize the GUI with a simulator and optional analyzer instance"""
        self.simulator = simulator
        self.analyzer = analyzer or CrowdFlowAnalyzer(simulator)
        self.paused = False
        self.step_count = 0
        self.show_velocities = True
        self.show_density = False
        self.show_personal_space = False
        self.density_data = None
        self.colormap = plt.cm.viridis
        self.agent_colors = self._generate_agent_colors()
        self.exit_width = self._get_default_exit_width()
        
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
        
        # Add new attributes for analysis
        self.show_analysis = False
        self.analysis_type = 'density'  # Default analysis view
        self.recording = True  # Whether to record frames for analysis
        
    def _generate_agent_colors(self):
        """Generate consistent colors for agents"""
        # Use a colormap to assign colors based on agent index
        cmap = plt.cm.tab20
        colors = [cmap(i % 20) for i in range(self.simulator.num_agents)]
        return colors

    def _get_default_exit_width(self):
        """Get the width of the first exit or return default value"""
        if hasattr(self.simulator, 'exits') and self.simulator.exits:
            return self.simulator.exits[0]['width']
        return 3.0  # Default exit width
        
    def setup_controls(self):
        """Set up the control panel with additional analysis controls"""
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
        
        # Exit width slider (if exits exist)
        if hasattr(self.simulator, 'exits') and self.simulator.exits:
            ax_exit = self.fig.add_axes([0.55, 0.05, 0.35, 0.03])
            self.slider_exit = Slider(ax_exit, 'Exit Width', 1.0, 10.0, 
                                    valinit=self.exit_width)
            self.slider_exit.on_changed(self.update_exit_width)
        
        # Add analysis controls
        ax_analysis = self.fig.add_axes([0.85, 0.25, 0.1, 0.3])
        self.radio_analysis = RadioButtons(ax_analysis, 
            ('None', 'Density', 'Velocity', 'Voronoi'),
            active=0)
        self.radio_analysis.on_clicked(self.update_analysis_view)
        
        # Add analysis buttons
        ax_record = self.fig.add_axes([0.85, 0.2, 0.1, 0.04])
        self.btn_record = Button(ax_record, 'Toggle Recording')
        self.btn_record.on_clicked(self.toggle_recording)
        
        ax_export = self.fig.add_axes([0.85, 0.15, 0.1, 0.04])
        self.btn_export = Button(ax_export, 'Export Analysis')
        self.btn_export.on_clicked(self.export_analysis)
    
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
        if hasattr(self.simulator, 'exits') and self.simulator.exits:
            self.exit_width = val
            # Update all exits with new width
            for exit_info in self.simulator.exits:
                exit_info['width'] = val
    
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
            custom_layout={
                "exits": self.simulator.exits,
                "obstacles": self.simulator.obstacles
            }
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
        
        # Draw exits if present
        if hasattr(self.simulator, 'exits') and self.simulator.exits:
            for exit_info in self.simulator.exits:
                exit_pos = exit_info["position"]
                exit_width = exit_info["width"]
                
                # Different visualization based on exit position
                if exit_pos[0] == 0:  # Left wall
                    rect = Rectangle((-0.2, exit_pos[1] - exit_width/2), 0.2, exit_width,
                                   color='green', alpha=0.7)
                    self.ax_main.add_patch(rect)
                    self.ax_main.text(2, exit_pos[1], 'EXIT',
                                    ha='left', va='center', color='green', fontsize=12)
                elif exit_pos[0] == self.simulator.arena_width:  # Right wall
                    rect = Rectangle((self.simulator.arena_width, exit_pos[1] - exit_width/2),
                                   0.2, exit_width, color='green', alpha=0.7)
                    self.ax_main.add_patch(rect)
                    self.ax_main.text(self.simulator.arena_width-2, exit_pos[1], 'EXIT',
                                    ha='right', va='center', color='green', fontsize=12)
                elif exit_pos[1] == 0:  # Bottom wall
                    rect = Rectangle((exit_pos[0] - exit_width/2, -0.2), exit_width, 0.2,
                                   color='green', alpha=0.7)
                    self.ax_main.add_patch(rect)
                    self.ax_main.text(exit_pos[0], 2, 'EXIT',
                                    ha='center', va='bottom', color='green', fontsize=12)
                elif exit_pos[1] == self.simulator.arena_height:  # Top wall
                    rect = Rectangle((exit_pos[0] - exit_width/2, self.simulator.arena_height),
                                   exit_width, 0.2, color='green', alpha=0.7)
                    self.ax_main.add_patch(rect)
                    self.ax_main.text(exit_pos[0], self.simulator.arena_height-2, 'EXIT',
                                    ha='center', va='top', color='green', fontsize=12)
        
        # Create animation
        self.anim = FuncAnimation(self.fig, self.update_frame, 
                                 frames=self.simulator.time_steps,
                                 interval=20, blit=False)
    
    def update_frame(self, frame):
        """Update the animation frame with analysis integration"""
        # Clear the plot
        self.ax_main.cla()
        self.ax_main.set_xlim(0, self.simulator.arena_width)
        self.ax_main.set_ylim(0, self.simulator.arena_height)
        
        # Run simulation step if not paused
        if not self.paused:
            self.simulator.step()
            self.step_count += 1
            if self.recording:
                self.analyzer.record_frame()
        
        # Show analysis visualizations if enabled
        if self.show_analysis:
            if self.analysis_type == 'density':
                # Save current axes
                current_ax = plt.gca()
                plt.sca(self.ax_main)
                self.analyzer.visualize_density()
                plt.sca(current_ax)
            elif self.analysis_type == 'velocity':
                current_ax = plt.gca()
                plt.sca(self.ax_main)
                self.analyzer.visualize_velocity_field()
                plt.sca(current_ax)
            elif self.analysis_type == 'voronoi':
                current_ax = plt.gca()
                plt.sca(self.ax_main)
                self.analyzer.visualize_voronoi_diagram()
                plt.sca(current_ax)
        else:
            # Regular simulation visualization
            self._draw_simulation_state()
        
        # Update statistics with analysis data
        self._update_statistics()
        
        return []
    
    def _update_statistics(self):
        """Update statistics display with analysis data"""
        active_count = np.sum(self.simulator.active_agents)
        stats = self.analyzer.analyze_evacuation_efficiency()
        emergent = self.analyzer.summarize_emergent_behaviors()
        
        stats_text = (
            f'Step: {self.step_count}/{self.simulator.time_steps}\n'
            f'Active Agents: {active_count}/{self.simulator.num_agents}\n'
            f'Evacuation: {stats["evacuation_percentage"]:.1f}%\n'
            f'Avg Evac Rate: {stats["avg_evacuation_rate"]:.2f}\n'
            f'Bottlenecks: {emergent["bottlenecks"]["average"]:.1f}'
        )
        
        self.stats_text.set_text(stats_text)
        
    def _draw_simulation_state(self):
        """Draw the current simulation state"""
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
        
        # Draw exits if present
        if hasattr(self.simulator, 'exits') and self.simulator.exits:
            for exit_info in self.simulator.exits:
                exit_pos = exit_info["position"]
                exit_width = exit_info["width"]
                
                # Different visualization based on exit position
                if exit_pos[0] == 0:  # Left wall
                    rect = Rectangle((-0.2, exit_pos[1] - exit_width/2), 0.2, exit_width,
                                   color='green', alpha=0.7)
                    self.ax_main.add_patch(rect)
                    self.ax_main.text(2, exit_pos[1], 'EXIT',
                                    ha='left', va='center', color='green', fontsize=12)
                elif exit_pos[0] == self.simulator.arena_width:  # Right wall
                    rect = Rectangle((self.simulator.arena_width, exit_pos[1] - exit_width/2),
                                   0.2, exit_width, color='green', alpha=0.7)
                    self.ax_main.add_patch(rect)
                    self.ax_main.text(self.simulator.arena_width-2, exit_pos[1], 'EXIT',
                                    ha='right', va='center', color='green', fontsize=12)
                elif exit_pos[1] == 0:  # Bottom wall
                    rect = Rectangle((exit_pos[0] - exit_width/2, -0.2), exit_width, 0.2,
                                   color='green', alpha=0.7)
                    self.ax_main.add_patch(rect)
                    self.ax_main.text(exit_pos[0], 2, 'EXIT',
                                    ha='center', va='bottom', color='green', fontsize=12)
                elif exit_pos[1] == self.simulator.arena_height:  # Top wall
                    rect = Rectangle((exit_pos[0] - exit_width/2, self.simulator.arena_height),
                                   exit_width, 0.2, color='green', alpha=0.7)
                    self.ax_main.add_patch(rect)
                    self.ax_main.text(exit_pos[0], self.simulator.arena_height-2, 'EXIT',
                                    ha='center', va='top', color='green', fontsize=12)
        
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
    
    def update_analysis_view(self, label):
        """Update the analysis visualization type"""
        self.analysis_type = label.lower()
        self.show_analysis = label != 'None'
        
    def toggle_recording(self, event):
        """Toggle recording of simulation data for analysis"""
        self.recording = not self.recording
        self.btn_record.label.set_text('Recording: ' + ('On' if self.recording else 'Off'))
        
    def export_analysis(self, event):
        """Export current analysis results"""
        filename = 'simulation_analysis.json'
        self.analyzer.export_analysis(filename)
        print(f"Analysis results exported to {filename}")

def run_crowd_simulation_with_gui():
    """Run crowd simulation with enhanced GUI"""
    parser = argparse.ArgumentParser(description='Crowd Flow Simulator with Enhanced GUI')
    parser.add_argument('--num_agents', type=int, default=100, help='Number of agents')
    parser.add_argument('--arena_width', type=float, default=50, help='Width of arena')
    parser.add_argument('--arena_height', type=float, default=50, help='Height of arena')
    parser.add_argument('--time_steps', type=int, default=1000, help='Maximum simulation steps')
    parser.add_argument('--dt', type=float, default=0.1, help='Time step size')
    parser.add_argument('--custom_layout', type=str, help='Path to custom layout JSON file')
    args = parser.parse_args()
    
    # Create default layout if no custom layout provided
    default_layout = {
        "exits": [
            {
                "position": [args.arena_width, args.arena_height/2],
                "width": 3.0
            }
        ],
        "obstacles": []
    }
    
    # Create simulator
    simulator = CrowdFlowSimulator(
        num_agents=args.num_agents,
        arena_width=args.arena_width,
        arena_height=args.arena_height,
        time_steps=args.time_steps,
        dt=args.dt,
        custom_layout=args.custom_layout or default_layout
    )
    
    # Create and run GUI
    gui = CrowdSimulationGUI(simulator)
    gui.run()

if __name__ == "__main__":
    run_crowd_simulation_with_gui()