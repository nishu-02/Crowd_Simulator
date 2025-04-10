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

# Add these imports at the top
from matplotlib.patches import Circle, Rectangle, Arrow, FancyArrowPatch
from matplotlib.collections import PatchCollection
import matplotlib.patheffects as path_effects

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
        
        # Set up the figure and axes with adjusted layout
        self.fig = plt.figure(figsize=(14, 9))
        self.fig.canvas.manager.set_window_title('Advanced Crowd Flow Simulator')
        
        # Main plotting area with adjusted position
        self.ax_main = self.fig.add_axes([0.1, 0.25, 0.7, 0.65])
        
        # Control panel area
        self.setup_controls()
        
        # Add this line to avoid tight_layout warning
        self.fig.set_tight_layout(False)
        
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
        
        # Add new visualization settings
        self.show_human_shapes = True  # Toggle between circles and human shapes
        self.show_paths = False        # Show agent paths
        self.show_heatmap = False     # Show real-time heatmap
        self.path_history = []        # Store agent paths
        self.max_path_length = 50     # Maximum length of path history
        
        # Color schemes
        self.color_schemes = {
            'speed': plt.cm.RdYlBu,      # Red to Blue for speed
            'density': plt.cm.viridis,    # Default density colormap
            'stress': plt.cm.RdYlGn_r     # Red for high stress
        }
        
        # Add visualization controls
        self.setup_visualization_controls()
        
        # Add obstacle visualization settings
        self.show_obstacles = True
    
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
        
        # Add obstacle controls
        ax_obstacles = self.fig.add_axes([0.85, 0.35, 0.1, 0.1])
        self.check_obstacles = CheckButtons(ax_obstacles, ['Show Obstacles'], [True])
        self.check_obstacles.on_clicked(self.toggle_obstacles)

        # Add obstacle editor button
        ax_edit = self.fig.add_axes([0.85, 0.3, 0.1, 0.04])
        self.btn_edit = Button(ax_edit, 'Edit Obstacles')
        self.btn_edit.on_clicked(self.edit_obstacles)
    
    def setup_visualization_controls(self):
        """Add controls for advanced visualization options"""
        # Visualization mode buttons
        ax_viz = self.fig.add_axes([0.85, 0.6, 0.1, 0.2])
        self.viz_radio = RadioButtons(ax_viz, 
            ('Basic', 'Human', 'Heatmap', 'Stress'),
            active=0)
        self.viz_radio.on_clicked(self.update_visualization_mode)
        
        # Path tracking toggle
        ax_path = self.fig.add_axes([0.85, 0.55, 0.1, 0.04])
        self.btn_path = Button(ax_path, 'Show Paths')
        self.btn_path.on_clicked(self.toggle_paths)
        
        # Color scheme selector
        ax_color = self.fig.add_axes([0.85, 0.5, 0.1, 0.04])
        self.btn_color = Button(ax_color, 'Color Scheme')
        self.btn_color.on_clicked(self.cycle_color_scheme)
    
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
        self.ax_main.set_xlim(0, self.simulator.arena_width)
        self.ax_main.set_ylim(0, self.simulator.arena_height)
        self.ax_main.set_title('Crowd Flow Simulation')
        self.ax_main.grid(True, alpha=0.3)
        
        # Create animation with improved settings
        self.anim = FuncAnimation(
            self.fig, 
            self.update_frame,
            frames=None,
            interval=50,  # 50ms between frames
            blit=False,
            cache_frame_data=False,  # Disable frame caching
            repeat=False
        )
    
    def update_frame(self, frame):
        """Update the animation frame"""
        # Clear the plot for redrawing
        self.ax_main.cla()
        self.ax_main.set_xlim(0, self.simulator.arena_width)
        self.ax_main.set_ylim(0, self.simulator.arena_height)
        self.ax_main.set_title('Crowd Flow Simulation')
        self.ax_main.grid(True, alpha=0.3)
        
        # Run simulation step if not paused
        if not self.paused:
            self.simulator.step()
            self.step_count += 1
            if self.recording:
                self.analyzer.record_frame()
        
        # Draw current simulation state
        self._draw_simulation_state()
        
        # Return empty list as required by FuncAnimation
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
        """Enhanced drawing of simulation state"""
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
        
        # Update path history if enabled
        if self.show_paths:
            if len(self.path_history) > self.max_path_length:
                self.path_history.pop(0)
            self.path_history.append(active_positions.copy())
        
        # Draw density heatmap if enabled
        if self.show_heatmap:
            density = self.calculate_density_field()
            self.ax_main.contourf(self.X, self.Y, density, 
                                levels=15, cmap=self.color_schemes['density'],
                                alpha=0.3)
        
        # Draw agents and their velocities
        for i, pos in enumerate(active_positions):
            agent_idx = active_indices[i]
            color = self.agent_colors[agent_idx]
            vel = active_velocities[i]
            speed = np.linalg.norm(vel)
            heading = np.arctan2(vel[1], vel[0]) if speed > 0 else 0
            
            # Calculate stress level based on local density and speed
            local_density = self._calculate_local_density(pos, active_positions)
            stress_level = min(1.0, local_density * (1 - speed/3.0))
            
            if self.show_human_shapes:
                self._draw_human_shape(pos, heading, color, stress_level)
            else:
                circle = Circle(pos, self.simulator.agent_radius, 
                              color=color, alpha=0.7)
                self.ax_main.add_patch(circle)
            
            # Draw velocity vector if enabled
            if self.show_velocities and speed > 0.1:  # Only show for moving agents
                # Scale velocity vector for visualization
                scale = 1.0  # Adjust this value to change arrow length
                vel_scaled = vel * scale
                
                # Draw arrow
                arrow = FancyArrowPatch(
                    pos, pos + vel_scaled,
                    color='red',
                    alpha=0.7,
                    arrowstyle='-|>',
                    mutation_scale=10,
                    shrinkA=0.5,
                    shrinkB=0.5
                )
                self.ax_main.add_patch(arrow)
            
            # Draw paths if enabled
            if self.show_paths and len(self.path_history) > 1:
                path_positions = [frame[i] for frame in self.path_history 
                                if i < len(frame)]
                self._draw_agent_path(path_positions, color)
        
        # Draw environment
        self._draw_environment()
        
        # Update status display
        self._update_status_display()
    
    def _calculate_local_density(self, pos, all_positions):
        """Calculate local density around a position"""
        distances = np.linalg.norm(all_positions - pos, axis=1)
        nearby = distances < 2.0  # 2 meter radius
        return np.sum(nearby) / (np.pi * 4.0)  # agents per square meter
    
    def _draw_environment(self):
        """Draw environmental elements with enhanced visuals"""
        # Draw obstacles if enabled
        if hasattr(self, 'show_obstacles') and self.show_obstacles:
            for obstacle in self.simulator.obstacles:
                pos = obstacle["position"]
                width = obstacle["width"]
                height = obstacle["height"]
                is_wall = obstacle.get("is_wall", False)
                
                # Different styling for walls vs obstacles
                color = 'black' if is_wall else 'gray'
                alpha = 0.9 if is_wall else 0.7
                
                rect = Rectangle((pos[0]-width/2, pos[1]-height/2), width, height,
                               color=color, alpha=alpha)
                self.ax_main.add_patch(rect)
                
                # Label significant obstacles
                if width * height > 9 and not is_wall:  # Larger than 3x3
                    self.ax_main.text(pos[0], pos[1], 'Obstacle',
                                    ha='center', va='center',
                                    color='white', fontsize=8)
        
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
    
    def _draw_human_shape(self, pos, heading, color, stress_level=0):
        """Draw a more human-like shape instead of a circle"""
        # Body
        body_radius = self.simulator.agent_radius
        body = Circle(pos, body_radius, color=color, alpha=0.7)
        self.ax_main.add_patch(body)
        
        # Head
        head_pos = (pos[0] + 0.3*body_radius*np.cos(heading),
                   pos[1] + 0.3*body_radius*np.sin(heading))
        head = Circle(head_pos, body_radius*0.3, color=color, alpha=0.7)
        self.ax_main.add_patch(head)
        
        # Stress indicator (optional)
        if stress_level > 0:
            stress_color = plt.cm.RdYlGn_r(stress_level)
            stress_ring = Circle(pos, body_radius*1.2, 
                               color=stress_color, fill=False, 
                               linestyle='--', alpha=0.5)
            self.ax_main.add_patch(stress_ring)
    
    def _draw_agent_path(self, positions, color):
        """Draw the recent path of an agent"""
        if len(positions) < 2:
            return
            
        # Create smooth path
        from scipy.interpolate import interp1d
        points = np.array(positions)
        x = points[:, 0]
        y = points[:, 1]
        
        # Interpolate path
        t = np.arange(len(positions))
        t_smooth = np.linspace(0, len(positions)-1, 100)
        
        try:
            x_smooth = interp1d(t, x, kind='quadratic')(t_smooth)
            y_smooth = interp1d(t, y, kind='quadratic')(t_smooth)
            
            # Draw smooth path with fade effect
            for i in range(len(x_smooth)-1):
                alpha = 0.1 + 0.4 * (i / len(x_smooth))
                self.ax_main.plot(x_smooth[i:i+2], y_smooth[i:i+2], 
                                color=color, alpha=alpha, linewidth=1)
        except ValueError:
            # Fallback to simple line if interpolation fails
            self.ax_main.plot(x, y, color=color, alpha=0.3, linewidth=1)
    
    def _draw_enhanced_exit(self, exit_info):
        """Draw exit with enhanced visual effects"""
        exit_pos = exit_info["position"]
        exit_width = exit_info["width"]
        
        # Base exit rectangle
        rect = self._create_exit_rect(exit_pos, exit_width)
        self.ax_main.add_patch(rect)
        
        # Add glow effect
        glow = self._create_exit_rect(exit_pos, exit_width*1.1)
        glow.set_alpha(0.3)
        glow.set_facecolor('yellow')
        self.ax_main.add_patch(glow)
        
        # Add 'EXIT' text with outline
        text = self._add_exit_text(exit_pos, exit_width)
        text.set_path_effects([
            path_effects.Stroke(linewidth=3, foreground='white'),
            path_effects.Normal()
        ])
    
    def update_visualization_mode(self, label):
        """Update visualization mode based on radio button selection"""
        self.show_human_shapes = (label == 'Human')
        self.show_heatmap = (label == 'Heatmap')
        self.show_stress = (label == 'Stress')
    
    def toggle_paths(self, event):
        """Toggle path visualization"""
        self.show_paths = not self.show_paths
        if not self.show_paths:
            self.path_history.clear()
        self.btn_path.label.set_text('Paths: ' + ('On' if self.show_paths else 'Off'))
    
    def cycle_color_scheme(self, event):
        """Cycle through available color schemes"""
        schemes = list(self.color_schemes.keys())
        current = getattr(self, '_current_scheme_index', 0)
        next_index = (current + 1) % len(schemes)
        self._current_scheme_index = next_index
        self.colormap = self.color_schemes[schemes[next_index]]
    
    def run(self):
        """Run the simulation GUI"""
        # Make sure the window is in the foreground
        self.fig.canvas.manager.window.attributes('-topmost', 1)
        self.fig.canvas.manager.window.attributes('-topmost', 0)
        
        # Show the plot and start the animation
        plt.show()
        
        return self.anim
    
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
    
    def _update_status_display(self):
        """Update status display with simulation metrics"""
        # Get statistics
        active_count = np.sum(self.simulator.active_agents)
        stats = self.analyzer.analyze_evacuation_efficiency()
        emergent = self.analyzer.summarize_emergent_behaviors()
        
        # Create modern status box in top-right corner
        status_box = dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.7)
        
        # Format status text with emoji indicators
        status_text = (
            f"🕒 Step: {self.step_count}/{self.simulator.time_steps}\n"
            f"👥 Active: {active_count}/{self.simulator.num_agents}\n"
            f"🚪 Evacuated: {stats['evacuation_percentage']:.1f}%\n"
            f"⚡ Avg Speed: {stats['avg_evacuation_rate']:.2f} m/s\n"
            f"⚠️ Bottlenecks: {emergent['bottlenecks']['average']:.1f}\n"
            f"🔄 {'Recording' if self.recording else 'Paused'}"
        )
        
        # Update or create status text with styling
        if not hasattr(self, 'status_display'):
            self.status_display = self.ax_main.text(
                0.98, 0.98, status_text,
                transform=self.ax_main.transAxes,
                color='white',
                fontfamily='monospace',
                fontsize=10,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=status_box
            )
        else:
            self.status_display.set_text(status_text)
    
    def toggle_obstacles(self, label):
        """Toggle obstacle visibility"""
        self.show_obstacles = not self.show_obstacles

    def edit_obstacles(self, event):
        """Open obstacle editor dialog"""
        from matplotlib.widgets import TextBox
        
        # Create a new figure for obstacle editing
        fig_edit = plt.figure(figsize=(8, 6))
        fig_edit.canvas.manager.set_window_title('Obstacle Editor')
        
        ax = fig_edit.add_subplot(111)
        ax.set_xlim(0, self.simulator.arena_width)
        ax.set_ylim(0, self.simulator.arena_height)
        
        # Draw current obstacles
        for obstacle in self.simulator.obstacles:
            pos = obstacle["position"]
            width = obstacle["width"]
            height = obstacle["height"]
            rect = Rectangle((pos[0]-width/2, pos[1]-height/2), width, height,
                            color='gray' if not obstacle.get("is_wall", False) else 'black',
                            alpha=0.7)
            ax.add_patch(rect)
        
        # Add obstacle on click
        def onclick(event):
            if event.inaxes != ax:
                return
            
            # Create obstacle at click position
            new_obstacle = {
                "position": [event.xdata, event.ydata],
                "width": 2.0,  # Default width
                "height": 2.0  # Default height
            }
            self.simulator.obstacles.append(new_obstacle)
            
            # Redraw
            ax.clear()
            ax.set_xlim(0, self.simulator.arena_width)
            ax.set_ylim(0, self.simulator.arena_height)
            for obstacle in self.simulator.obstacles:
                pos = obstacle["position"]
                width = obstacle["width"]
                height = obstacle["height"]
                rect = Rectangle((pos[0]-width/2, pos[1]-height/2), width, height,
                               color='gray', alpha=0.7)
                ax.add_patch(rect)
            plt.draw()
        
        # Connect click event
        fig_edit.canvas.mpl_connect('button_press_event', onclick)
        
        # Add instructions
        ax.set_title('Click to add obstacles\nClose window when done')
        plt.show()

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