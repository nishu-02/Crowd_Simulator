import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle, Circle, Wedge, FancyArrowPatch
from matplotlib.collections import PatchCollection
import matplotlib.colors as mcolors
import matplotlib.patheffects as path_effects
from matplotlib.widgets import Slider, Button, RadioButtons, CheckButtons
import argparse
import time
from scipy.ndimage import gaussian_filter
from scipy.interpolate import interp1d
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
        
        # Set up the figure and axes with improved visual style
        plt.style.use('dark_background')  # Use dark theme for modern look
        self.fig = plt.figure(figsize=(14, 9), facecolor='#1c1c1c')
        self.fig.canvas.manager.set_window_title('Advanced Crowd Flow Simulator')
        
        # Main plotting area with adjusted position
        self.ax_main = self.fig.add_axes([0.1, 0.25, 0.7, 0.65], facecolor='#2c2c2c')
        self.ax_main.tick_params(colors='white')
        for spine in self.ax_main.spines.values():
            spine.set_edgecolor('#555555')
        
        # Disable tight layout to avoid warnings and better control layout
        self.fig.set_tight_layout(False)
        
        # Initialize the stats area with modern styling
        self.stats_text = self.ax_main.text(0.02, 0.98, '', 
                                        transform=self.ax_main.transAxes,
                                        verticalalignment='top',
                                        color='white',
                                        fontfamily='monospace',
                                        fontsize=10,
                                        bbox=dict(boxstyle='round', facecolor='#333333', alpha=0.8))
        
        # Create heat map for density visualization with higher resolution
        self.grid_size = 75  # Increased from 50 for smoother heatmaps
        self.x_grid = np.linspace(0, simulator.arena_width, self.grid_size)
        self.y_grid = np.linspace(0, simulator.arena_height, self.grid_size)
        self.X, self.Y = np.meshgrid(self.x_grid, self.y_grid)
        self.density_plot = None
        
        # Add new attributes for analysis
        self.show_analysis = False
        self.analysis_type = 'density'  # Default analysis view
        self.recording = True  # Whether to record frames for analysis
        
        # Add new visualization settings
        self.show_human_shapes = True  # Toggle between circles and human shapes
        self.show_paths = False        # Show agent paths
        self.show_heatmap = False      # Show real-time heatmap
        self.path_history = []         # Store agent paths
        self.max_path_length = 50      # Maximum length of path history
        
        # Color schemes with improved aesthetics
        self.color_schemes = {
            'speed': plt.cm.coolwarm,      # Better color contrast for speed
            'density': plt.cm.plasma,       # More vibrant density colormap
            'stress': plt.cm.RdYlGn_r       # Red for high stress
        }
        self._current_scheme_index = 0      # Track current color scheme
        
        # Setup smooth transitions
        self.transition_speed = 0.2         # Animation smoothness
        self.last_positions = None          # For movement interpolation
        
        # Add obstacle visualization settings
        self.show_obstacles = True
        
        # Setup all controls
        self.setup_controls()
        self.setup_visualization_controls()
        
        # Set up the animation with optimized parameters
        self.setup_animation()
        
        # Add frame timing for performance monitoring
        self.last_frame_time = time.time()
        self.frame_times = []
        
        # For smooth animation
        self.prev_positions = None
        self.prev_velocities = None
    
    def _generate_agent_colors(self):
        """Generate visually appealing colors for agents"""
        # Use a carefully selected colormap to assign colors based on agent index
        cmap = plt.cm.viridis
        colors = [cmap(0.1 + 0.8 * (i / self.simulator.num_agents)) for i in range(self.simulator.num_agents)]
        return colors

    def _get_default_exit_width(self):
        """Get the width of the first exit or return default value"""
        if hasattr(self.simulator, 'exits') and self.simulator.exits:
            return self.simulator.exits[0]['width']
        return 3.0  # Default exit width
        
    def setup_controls(self):
        """Set up the control panel with improved styling"""
        # Define common button style
        button_color = '#444444'
        button_text_color = 'white'
        
        # Buttons for simulation control with improved styling
        ax_pause = self.fig.add_axes([0.1, 0.1, 0.15, 0.05])
        self.btn_pause = Button(ax_pause, 'Pause/Resume', color=button_color, hovercolor='#555555')
        self.btn_pause.label.set_color(button_text_color)
        self.btn_pause.on_clicked(self.toggle_pause)
        
        ax_reset = self.fig.add_axes([0.3, 0.1, 0.15, 0.05])
        self.btn_reset = Button(ax_reset, 'Reset Simulation', color=button_color, hovercolor='#555555')
        self.btn_reset.label.set_color(button_text_color)
        self.btn_reset.on_clicked(self.reset_simulation)
        
        # Slider for simulation speed with improved styling
        ax_speed = self.fig.add_axes([0.55, 0.1, 0.35, 0.03])
        # For even higher speeds:
        self.slider_speed = Slider(
            ax_speed, 'Speed', 10, 300, valinit=150,  # Increased max to 300 and default to 150
            color='#007bff', initcolor='#00a8ff'
        )
        self.slider_speed.label.set_color('white')
        self.slider_speed.valtext.set_color('white')
        
        # Checkboxes for display options with improved styling
        ax_display = self.fig.add_axes([0.1, 0.02, 0.3, 0.05])
        self.check_display = CheckButtons(
            ax_display, 
            ['Velocities', 'Density', 'Personal Space'], 
            [self.show_velocities, self.show_density, self.show_personal_space]
        )
        
        # Style the checkboxes safely
        self._safely_style_widget(self.check_display, '#00a8ff', 'white')
        for text in self.check_display.labels:
            text.set_color('white')
        self.check_display.on_clicked(self.update_display_options)
        
        # Exit width slider if exits exist
        if hasattr(self.simulator, 'exits') and self.simulator.exits:
            ax_exit = self.fig.add_axes([0.55, 0.05, 0.35, 0.03])
            self.slider_exit = Slider(
                ax_exit, 'Exit Width', 1.0, 10.0, 
                valinit=self.exit_width,
                color='#28a745', initcolor='#34ce57'
            )
            self.slider_exit.label.set_color('white')
            self.slider_exit.valtext.set_color('white')
            self.slider_exit.on_changed(self.update_exit_width)
        
        # Add analysis controls with improved styling
        ax_analysis = self.fig.add_axes([0.85, 0.4, 0.1, 0.2])
        self.radio_analysis = RadioButtons(
            ax_analysis, 
            ('None', 'Density', 'Velocity', 'Voronoi'),
            active=0
        )
        # Style the radio buttons safely
        self._safely_style_radio(self.radio_analysis, '#00a8ff', 'white')
        for text in self.radio_analysis.labels:
            text.set_color('white')
        self.radio_analysis.on_clicked(self.update_analysis_view)
        
        # Analysis section title
        self.fig.text(0.9, 0.61, 'ANALYSIS', 
                    color='white', fontweight='bold', 
                    fontsize=10, ha='center')
        
        # Add analysis buttons with improved styling
        ax_record = self.fig.add_axes([0.85, 0.35, 0.1, 0.04])
        self.btn_record = Button(ax_record, 'Toggle Recording', color=button_color, hovercolor='#555555')
        self.btn_record.label.set_color(button_text_color)
        self.btn_record.on_clicked(self.toggle_recording)
        
        ax_export = self.fig.add_axes([0.85, 0.3, 0.1, 0.04])
        self.btn_export = Button(ax_export, 'Export Analysis', color=button_color, hovercolor='#555555')
        self.btn_export.label.set_color(button_text_color)
        self.btn_export.on_clicked(self.export_analysis)
        
        # Add obstacle controls with improved styling
        ax_obstacles = self.fig.add_axes([0.85, 0.24, 0.1, 0.05])
        self.check_obstacles = CheckButtons(
            ax_obstacles, ['Show Obstacles'], [True],
        )
        # Style the obstacle checkbox safely
        self._safely_style_widget(self.check_obstacles, '#00a8ff', 'white')
        for text in self.check_obstacles.labels:
            text.set_color('white')
        self.check_obstacles.on_clicked(self.toggle_obstacles)

        # Add obstacle editor button
        ax_edit = self.fig.add_axes([0.85, 0.19, 0.1, 0.04])
        self.btn_edit = Button(ax_edit, 'Edit Obstacles', color=button_color, hovercolor='#555555')
        self.btn_edit.label.set_color(button_text_color)
        self.btn_edit.on_clicked(self.edit_obstacles)
        
        # Add simulation info with title
        self.fig.text(0.9, 0.85, 'SIMULATION INFO', 
                    color='white', fontweight='bold', 
                    fontsize=10, ha='center')
        
        # Add performance monitor text
        self.perf_text = self.fig.text(
            0.9, 0.8, 'FPS: --', 
            color='white', fontsize=9, ha='center',
            fontfamily='monospace'
        )
        
    def setup_visualization_controls(self):
        """Add controls for advanced visualization options with improved styling"""
        # Header for visualization section
        self.fig.text(0.9, 0.75, 'VISUALIZATION', 
                    color='white', fontweight='bold', 
                    fontsize=10, ha='center')
        
        # Visualization mode buttons with improved styling
        ax_viz = self.fig.add_axes([0.85, 0.65, 0.1, 0.09])
        self.viz_radio = RadioButtons(
            ax_viz, 
            ('Basic', 'Human', 'Heatmap', 'Stress'),
            active=1,  # Default to human shapes
        )
        # Style the radio buttons safely
        self._safely_style_radio(self.viz_radio, '#00a8ff', 'white')
        for text in self.viz_radio.labels:
            text.set_color('white')
        self.viz_radio.on_clicked(self.update_visualization_mode)
        
        # Button color for visualization controls
        button_color = '#444444'
        button_text_color = 'white'
        
        # Path tracking toggle with improved styling
        ax_path = self.fig.add_axes([0.85, 0.6, 0.1, 0.04])
        self.btn_path = Button(ax_path, 'Show Paths', color=button_color, hovercolor='#555555')
        self.btn_path.label.set_color(button_text_color)
        self.btn_path.on_clicked(self.toggle_paths)
        
        # Color scheme selector with improved styling
        ax_color = self.fig.add_axes([0.85, 0.55, 0.1, 0.04])
        self.btn_color = Button(ax_color, 'Color Scheme', color=button_color, hovercolor='#555555')
        self.btn_color.label.set_color(button_text_color)
        self.btn_color.on_clicked(self.cycle_color_scheme)
        
        # Add performance info status box
        self.fig.text(0.9, 0.16, 'PERFORMANCE', 
                    color='white', fontweight='bold', 
                    fontsize=10, ha='center')
        
        # Add application title and status
        self.fig.text(0.1, 0.94, 'ADVANCED CROWD FLOW SIMULATOR', 
                    color='white', fontweight='bold', 
                    fontsize=16, ha='left')
        
        self.status_label = self.fig.text(
            0.1, 0.91, 'Status: Running', 
            color='#00ff00', fontsize=10, ha='left',
            fontfamily='monospace'
        )
    
    def _safely_style_widget(self, widget, facecolor, edgecolor):
        """Safely style checkbox and radio button widgets across different matplotlib versions"""
        try:
            # Try different attribute names used in various matplotlib versions
            if hasattr(widget, 'rectangles'):
                for rect in widget.rectangles:
                    rect.set_facecolor(facecolor)
                    rect.set_edgecolor(edgecolor)
            elif hasattr(widget, 'boxes'):
                for box in widget.boxes:
                    box.set_facecolor(facecolor)
                    box.set_edgecolor(edgecolor)
            # Access container or artists as fallback
            elif hasattr(widget, 'container'):
                for artist in widget.container.get_children():
                    if hasattr(artist, 'set_facecolor'):
                        artist.set_facecolor(facecolor)
                    if hasattr(artist, 'set_edgecolor'):
                        artist.set_edgecolor(edgecolor)
            elif hasattr(widget, 'artists'):
                for artist in widget.artists:
                    if hasattr(artist, 'set_facecolor'):
                        artist.set_facecolor(facecolor)
                    if hasattr(artist, 'set_edgecolor'):
                        artist.set_edgecolor(edgecolor)
        except (AttributeError, TypeError):
            # If all else fails, don't style the widget but don't crash
            pass
    
    def _safely_style_radio(self, radio_widget, facecolor, edgecolor):
        """Safely style radio button widgets across different matplotlib versions"""
        try:
            # Try different attribute names used in various matplotlib versions
            if hasattr(radio_widget, 'circles'):
                for circle in radio_widget.circles:
                    if hasattr(circle, 'set_facecolor'):
                        circle.set_facecolor(facecolor)
                    if hasattr(circle, 'set_edgecolor'):
                        circle.set_edgecolor(edgecolor)
            # Access container or artists as fallback
            elif hasattr(radio_widget, 'container'):
                for artist in radio_widget.container.get_children():
                    if hasattr(artist, 'set_facecolor'):
                        artist.set_facecolor(facecolor)
                    if hasattr(artist, 'set_edgecolor'):
                        artist.set_edgecolor(edgecolor)
            elif hasattr(radio_widget, 'artists'):
                for artist in radio_widget.artists:
                    if hasattr(artist, 'set_facecolor'):
                        artist.set_facecolor(facecolor)
                    if hasattr(artist, 'set_edgecolor'):
                        artist.set_edgecolor(edgecolor)
        except (AttributeError, TypeError):
            # If all else fails, don't style the widget but don't crash
            pass

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
        """Toggle pause/resume of the simulation with visual feedback"""
        self.paused = not self.paused
        self.status_label.set_text(f"Status: {'Paused' if self.paused else 'Running'}")
        self.status_label.set_color('#ff9900' if self.paused else '#00ff00')
    
    def reset_simulation(self, event):
        """Reset the simulation with visual feedback"""
        # Visual feedback during reset
        self.status_label.set_text("Status: Resetting...")
        self.status_label.set_color('#ff9900')
        plt.pause(0.1)  # Brief pause for visual feedback
        
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
        # Clear path history
        self.path_history = []
        self.prev_positions = None
        self.prev_velocities = None
        
        # Reset status
        self.status_label.set_text("Status: Running")
        self.status_label.set_color('#00ff00')
    
    def calculate_density_field(self):
        """Calculate density field from agent positions with improved smoothing"""
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
            
            # Apply adaptive smoothing based on agent density
            avg_density = np.mean(hist[hist > 0]) if np.any(hist > 0) else 1
            smooth_sigma = 1.5 if avg_density < 2 else 1.0  # Less smoothing in dense areas
            density = gaussian_filter(hist, sigma=smooth_sigma)
        
        return density
    
    def setup_animation(self):
        """Set up the animation elements with improved visuals"""
        self.ax_main.set_xlim(0, self.simulator.arena_width)
        self.ax_main.set_ylim(0, self.simulator.arena_height)
        self.ax_main.set_title('Crowd Flow Simulation', color='white', fontsize=12)
        self.ax_main.grid(True, alpha=0.2, color='#555555', linestyle='--')
        
        # Create animation with optimized settings
        self.anim = FuncAnimation(
            self.fig, 
            self.update_frame,
            frames=None,
            interval=20,  # Faster refresh rate (50fps) for smoother animation
            blit=False,   # blit=True can cause issues with complex plots
            cache_frame_data=False,  # Disable frame caching
            repeat=False
        )
    
    def update_frame(self, frame):
        """Update the animation frame with performance monitoring"""
        # Start frame timing
        frame_start = time.time()
        
        # Store previous positions for interpolation
        if not self.paused and self.simulator.active_agents.any():
            self.prev_positions = self.simulator.positions[self.simulator.active_agents].copy()
            self.prev_velocities = self.simulator.velocities[self.simulator.active_agents].copy()
        
        # Clear the plot for redrawing with reduced flicker
        self.ax_main.clear()
        self.ax_main.set_xlim(0, self.simulator.arena_width)
        self.ax_main.set_ylim(0, self.simulator.arena_height)
        self.ax_main.set_title('Crowd Flow Simulation', color='white', fontsize=12)
        self.ax_main.grid(True, alpha=0.2, color='#555555', linestyle='--')
        
        # Run simulation step if not paused
        if not self.paused:
            self.simulator.step()
            self.step_count += 1
            if self.recording:
                self.analyzer.record_frame()
        
        # Draw current simulation state
        self._draw_simulation_state()
        
        # Calculate frame time and update FPS counter
        frame_time = time.time() - frame_start
        self.frame_times.append(frame_time)
        
        # Only keep recent frame times to calculate moving average
        if len(self.frame_times) > 30:
            self.frame_times.pop(0)
        
        # Calculate and display FPS
        if self.frame_times:
            avg_frame_time = sum(self.frame_times) / len(self.frame_times)
            fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
            self.perf_text.set_text(f"FPS: {fps:.1f}")
            
            # Colorize performance metrics
            if fps > 30:
                self.perf_text.set_color('#00ff00')  # Green for good performance
            elif fps > 15:
                self.perf_text.set_color('#ffff00')  # Yellow for medium performance
            else:
                self.perf_text.set_color('#ff0000')  # Red for poor performance
        
        # Return empty list as required by FuncAnimation
        return []
    
    def _draw_simulation_state(self):
        """Enhanced drawing of simulation state with smooth animations"""
        # Draw environment elements first (background)
        self._draw_environment()
        
        # Show density field if selected
        if self.show_density or self.show_heatmap:
            density = self.calculate_density_field()
            cmap = self.color_schemes['density']
            self.density_plot = self.ax_main.contourf(
                self.X, self.Y, density, cmap=cmap, 
                alpha=0.4, levels=20  # More levels for smoother gradient
            )
        
        # Get simulation data
        active_indices = np.where(self.simulator.active_agents)[0]
        active_positions = self.simulator.positions[self.simulator.active_agents]
        active_velocities = self.simulator.velocities[self.simulator.active_agents]
        
        # Update path history if enabled
        if self.show_paths:
            if len(self.path_history) >= self.max_path_length:
                self.path_history.pop(0)
            if active_positions.size > 0:  # Only append if we have positions
                self.path_history.append(active_positions.copy())
            
            # Draw paths with improved styling
            if len(self.path_history) >= 2:
                agent_count = min(len(self.path_history[0]), len(active_positions))
                for i in range(agent_count):
                    path_points = [path[i] for path in self.path_history if i < len(path)]
                    if len(path_points) > 1:
                        points = np.array(path_points)
                        # Use alpha gradient for fade effect
                        for j in range(len(points) - 1):
                            alpha = 0.1 + 0.4 * (j / len(points))
                            self.ax_main.plot(
                                points[j:j+2, 0], points[j:j+2, 1],
                                color=self.agent_colors[active_indices[i] if i < len(active_indices) else 0],
                                alpha=alpha, linewidth=1.5
                            )
        
        # Draw agents and their velocities with smooth transitions
        for i, pos in enumerate(active_positions):
            if i >= len(active_indices):
                continue  # Skip if indices don't match
                
            agent_idx = active_indices[i]
            color = self.agent_colors[agent_idx]
            vel = active_velocities[i]
            speed = np.linalg.norm(vel)
            heading = np.arctan2(vel[1], vel[0]) if speed > 0 else 0
            
            # Calculate stress level for enhanced visuals
            stress_level = min(1.0, speed / 2.0)  # Higher speed means more stress
            
            # Draw agent using the appropriate visualization
            if self.show_human_shapes:
                self._draw_human_shape(pos, heading, color, stress_level)
            else:
                glow_radius = self.simulator.agent_radius * 1.2
                # Add subtle glow effect
                glow = Circle(pos, glow_radius, color=color, alpha=0.3)
                self.ax_main.add_patch(glow)
                
                # Main agent circle
                circle = Circle(pos, self.simulator.agent_radius, color=color, alpha=0.7)
                self.ax_main.add_patch(circle)
            
            # Draw velocity vector if enabled with improved appearance
            if self.show_velocities and speed > 0.1:  # Only show for moving agents
                # Scale velocity vector for better visualization
                scale = 1.0  # Adjust this value to change arrow length
                vel_scaled = vel * scale
                
                # Create elegant arrow
                arrow = FancyArrowPatch(
                    pos, pos + vel_scaled,
                    color='#ff5555',  # Brighter red for better visibility
                    alpha=0.8,
                    arrowstyle='-|>',
                    mutation_scale=10,
                    shrinkA=0.5,
                    shrinkB=0.5,
                    linewidth=1.5
                )
                # Add subtle glow effect to arrows
                arrow.set_path_effects([
                    path_effects.Stroke(linewidth=2.5, foreground='#ff555533'),
                    path_effects.Normal()
                ])
                self.ax_main.add_patch(arrow)
        
        # Update status display
        self._update_status_display()
    
    def _draw_environment(self):
        """Draw environmental elements with enhanced visuals"""
        # Draw exits if present with enhanced visuals
        if hasattr(self.simulator, 'exits') and self.simulator.exits:
            for exit_info in self.simulator.exits:
                exit_pos = exit_info["position"]
                exit_width = exit_info["width"]
                
                # Determine exit position and orientation
                if exit_pos[0] == 0:  # Left wall
                    # Draw exit area with gradient
                    rect = Rectangle(
                        (-0.5, exit_pos[1] - exit_width/2), 0.5, exit_width,
                        color='#00cc00', alpha=0.7
                    )
                    self.ax_main.add_patch(rect)
                    
                    # Add glow effect
                    for i in range(5):
                        alpha = 0.1 - i * 0.015
                        glow = Rectangle(
                            (-0.5-i*0.2, exit_pos[1] - exit_width/2 - i*0.2),
                            0.5, exit_width + i*0.4,
                            color='#66ff66', alpha=alpha
                        )
                        self.ax_main.add_patch(glow)
                    
                    # Add EXIT text with glow
                    text = self.ax_main.text(
                        2, exit_pos[1], 'EXIT',
                        ha='left', va='center',
                        color='#00ff00', fontsize=12, fontweight='bold'
                    )
                    text.set_path_effects([
                        path_effects.Stroke(linewidth=3, foreground='#003300'),
                        path_effects.Normal()
                    ])
                
                elif exit_pos[0] == self.simulator.arena_width:  # Right wall
                    # Draw exit area with gradient
                    rect = Rectangle(
                        (self.simulator.arena_width, exit_pos[1] - exit_width/2),
                        0.5, exit_width, color='#00cc00', alpha=0.7
                    )
                    self.ax_main.add_patch(rect)
                    
                    # Add glow effect
                    for i in range(5):
                        alpha = 0.1 - i * 0.015
                        glow = Rectangle(
                            (self.simulator.arena_width+i*0.2, exit_pos[1] - exit_width/2 - i*0.2),
                            0.5, exit_width + i*0.4,
                            color='#66ff66', alpha=alpha
                        )
                        self.ax_main.add_patch(glow)
                    
                    # Add EXIT text with glow
                    text = self.ax_main.text(
                        self.simulator.arena_width-2, exit_pos[1], 'EXIT',
                        ha='right', va='center',
                        color='#00ff00', fontsize=12, fontweight='bold'
                    )
                    text.set_path_effects([
                        path_effects.Stroke(linewidth=3, foreground='#003300'),
                        path_effects.Normal()
                    ])
                
                elif exit_pos[1] == 0:  # Bottom wall
                    # Draw exit area with gradient
                    rect = Rectangle(
                        (exit_pos[0] - exit_width/2, -0.5), exit_width, 0.5,
                        color='#00cc00', alpha=0.7
                    )
                    self.ax_main.add_patch(rect)
                    
                    # Add glow effect
                    for i in range(5):
                        alpha = 0.1 - i * 0.015
                        glow = Rectangle(
                            (exit_pos[0] - exit_width/2 - i*0.2, -0.5-i*0.2),
                            exit_width + i*0.4, 0.5,
                            color='#66ff66', alpha=alpha
                        )
                        self.ax_main.add_patch(glow)
                    
                    # Add EXIT text with glow
                    text = self.ax_main.text(
                        exit_pos[0], 2, 'EXIT',
                        ha='center', va='bottom',
                        color='#00ff00', fontsize=12, fontweight='bold'
                    )
                    text.set_path_effects([
                        path_effects.Stroke(linewidth=3, foreground='#003300'),
                        path_effects.Normal()
                    ])
                
                elif exit_pos[1] == self.simulator.arena_height:  # Top wall
                    # Draw exit area with gradient
                    rect = Rectangle(
                        (exit_pos[0] - exit_width/2, self.simulator.arena_height),
                        exit_width, 0.5, color='#00cc00', alpha=0.7
                    )
                    self.ax_main.add_patch(rect)
                    
                    # Add glow effect
                    for i in range(5):
                        alpha = 0.1 - i * 0.015
                        glow = Rectangle(
                            (exit_pos[0] - exit_width/2 - i*0.2, self.simulator.arena_height+i*0.2),
                            exit_width + i*0.4, 0.5,
                            color='#66ff66', alpha=alpha
                        )
                        self.ax_main.add_patch(glow)
                    
                    # Add EXIT text with glow
                    text = self.ax_main.text(
                        exit_pos[0], self.simulator.arena_height-2, 'EXIT',
                        ha='center', va='top',
                        color='#00ff00', fontsize=12, fontweight='bold'
                    )
                    text.set_path_effects([
                        path_effects.Stroke(linewidth=3, foreground='#003300'),
                        path_effects.Normal()
                    ])
        
        # Draw obstacles with enhanced visuals
        if hasattr(self, 'show_obstacles') and self.show_obstacles:
            for obstacle in self.simulator.obstacles:
                pos = obstacle["position"]
                width = obstacle["width"]
                height = obstacle["height"]
                is_wall = obstacle.get("is_wall", False)
                
                # Different styling for walls vs obstacles
                if is_wall:
                    # Wall styling - solid, more imposing
                    color = '#333333'
                    edge_color = '#666666'
                    alpha = 0.95
                else:
                    # Regular obstacle styling - more varied
                    color = '#555555'
                    edge_color = '#888888'
                    alpha = 0.8
                
                # Create base rectangle with enhanced styling
                rect = Rectangle(
                    (pos[0]-width/2, pos[1]-height/2), width, height,
                    facecolor=color, edgecolor=edge_color, 
                    alpha=alpha, linewidth=1.5
                )
                
                # Add subtle pattern or texture
                if not is_wall and (width > 3 or height > 3):
                    rect.set_hatch('///')
                
                self.ax_main.add_patch(rect)
                
                # Label significant obstacles
                if width * height > 9 and not is_wall:  # Larger than 3x3
                    # Add text with outline effect
                    text = self.ax_main.text(
                        pos[0], pos[1], 'Obstacle',
                        ha='center', va='center',
                        color='white', fontsize=8, fontweight='bold'
                    )
                    text.set_path_effects([
                        path_effects.Stroke(linewidth=2, foreground='black'),
                        path_effects.Normal()
                    ])
    
    def _draw_human_shape(self, pos, heading, color, stress_level=0):
        """Draw a more human-like shape with enhanced visuals"""
        # Calculate radius with slight random variation for more natural appearance
        body_radius = self.simulator.agent_radius * (0.95 + 0.1 * np.random.random())
        
        # Create gradient effect based on stress level
        if stress_level > 0.5:
            # High stress - reddish glow
            stress_color = plt.cm.RdYlGn_r(stress_level)
            # Add subtle stress glow
            stress_glow = Circle(
                pos, body_radius*1.4, 
                color=stress_color, fill=False, 
                alpha=0.3 * stress_level
            )
            self.ax_main.add_patch(stress_glow)
        
        # Body with subtle gradient
        body = Circle(pos, body_radius, color=color, alpha=0.8)
        self.ax_main.add_patch(body)
        
        # Head
        head_pos = (pos[0] + 0.3*body_radius*np.cos(heading),
                pos[1] + 0.3*body_radius*np.sin(heading))
        head = Circle(head_pos, body_radius*0.35, color=color, alpha=0.9)
        self.ax_main.add_patch(head)
        
        # Arms (optional, can be simplified for performance)
        arm_length = body_radius * 0.7
        arm_angle1 = heading + np.pi/4  # Right arm
        arm_angle2 = heading - np.pi/4  # Left arm
        
        # Only draw arms for some agents to reduce visual complexity
        if np.random.random() < 0.3:  # 30% chance to draw arms
            # Right arm
            arm1_end = (pos[0] + arm_length*np.cos(arm_angle1),
                    pos[1] + arm_length*np.sin(arm_angle1))
            self.ax_main.plot([pos[0], arm1_end[0]], [pos[1], arm1_end[1]],
                            color=color, linewidth=2, alpha=0.7)
            
            # Left arm
            arm2_end = (pos[0] + arm_length*np.cos(arm_angle2),
                    pos[1] + arm_length*np.sin(arm_angle2))
            self.ax_main.plot([pos[0], arm2_end[0]], [pos[1], arm2_end[1]],
                            color=color, linewidth=2, alpha=0.7)
    
    def _update_status_display(self):
        """Update status display with simulation metrics using a modern design"""
        # Get statistics
        active_count = np.sum(self.simulator.active_agents)
        stats = self.analyzer.analyze_evacuation_efficiency()
        emergent = self.analyzer.summarize_emergent_behaviors()
        
        # Create modern status box in top-right corner with glass effect
        status_box = dict(
            boxstyle='round,pad=0.6', 
            facecolor='#1e1e1e', 
            edgecolor='#444444',
            alpha=0.85
        )
        
        # Format status text with emoji indicators and colorful text
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
            
        # Add evacuation progress bar if there are exits
        if hasattr(self.simulator, 'exits') and self.simulator.exits:
            evac_percent = stats['evacuation_percentage'] / 100.0
            # Draw background
            self.ax_main.add_patch(Rectangle(
                (5, self.simulator.arena_height - 3), 
                self.simulator.arena_width - 10, 1,
                facecolor='#333333', edgecolor='#666666', alpha=0.5
            ))
            
            # Draw progress
            width = (self.simulator.arena_width - 10) * evac_percent
            self.ax_main.add_patch(Rectangle(
                (5, self.simulator.arena_height - 3), 
                width, 1,
                facecolor='#00cc00', alpha=0.7
            ))
            
            # Add label
            self.ax_main.text(
                self.simulator.arena_width / 2, self.simulator.arena_height - 2.5,
                f"Evacuation: {stats['evacuation_percentage']:.1f}%",
                ha='center', va='center', color='white', fontsize=9
            )
    
    def update_visualization_mode(self, label):
        """Update visualization mode based on radio button selection with visual feedback"""
        self.show_human_shapes = (label == 'Human')
        self.show_heatmap = (label == 'Heatmap')
        self.show_stress = (label == 'Stress')
        
        # Update status with visual feedback
        if label == 'Human':
            self.status_label.set_text("Status: Human shapes active")
        elif label == 'Heatmap':
            self.status_label.set_text("Status: Heatmap active")
        elif label == 'Stress':
            self.status_label.set_text("Status: Stress visualization active")
        else:
            self.status_label.set_text("Status: Basic visualization")
    
    def toggle_paths(self, event):
        """Toggle path visualization with visual feedback"""
        self.show_paths = not self.show_paths
        if not self.show_paths:
            self.path_history.clear()
        
        # Update button text and status
        self.btn_path.label.set_text('Paths: ' + ('On' if self.show_paths else 'Off'))
        self.status_label.set_text(f"Status: Paths {'enabled' if self.show_paths else 'disabled'}")
    
    def cycle_color_scheme(self, event):
        """Cycle through available color schemes with visual feedback"""
        schemes = list(self.color_schemes.keys())
        current = getattr(self, '_current_scheme_index', 0)
        next_index = (current + 1) % len(schemes)
        self._current_scheme_index = next_index
        self.colormap = self.color_schemes[schemes[next_index]]
        
        # Update button text and status
        self.btn_color.label.set_text(f'Color: {schemes[next_index].title()}')
        self.status_label.set_text(f"Status: {schemes[next_index].title()} color scheme")
    
    def update_analysis_view(self, label):
        """Update the analysis visualization type with visual feedback"""
        self.analysis_type = label.lower()
        self.show_analysis = label != 'None'
        
        # Update status
        if label != 'None':
            self.status_label.set_text(f"Status: {label} analysis active")
        else:
            self.status_label.set_text("Status: Analysis disabled")
        
    def toggle_recording(self, event):
        """Toggle recording of simulation data for analysis with visual feedback"""
        self.recording = not self.recording
        
        # Update button text and status
        self.btn_record.label.set_text('Recording: ' + ('On' if self.recording else 'Off'))
        self.status_label.set_text(f"Status: Recording {'enabled' if self.recording else 'disabled'}")
        
    def export_analysis(self, event):
        """Export current analysis results with visual feedback"""
        # Update status during export
        self.status_label.set_text("Status: Exporting analysis...")
        self.status_label.set_color('#ffff00')  # Yellow during export
        plt.pause(0.1)  # Allow UI to update
        
        # Perform export
        filename = 'simulation_analysis.json'
        self.analyzer.export_analysis(filename)
        
        # Show success message in status
        self.status_label.set_text(f"Status: Exported to {filename}")
        self.status_label.set_color('#00ff00')  # Green for success
        
        # Create a temporary notification
        notification = self.fig.text(
            0.5, 0.5, f"Analysis exported to {filename}",
            ha='center', va='center',
            color='white', fontsize=14,
            bbox=dict(boxstyle='round', facecolor='#007bff', alpha=0.8)
        )
        
        # Flash notification
        plt.pause(1.0)
        notification.remove()
        
        # Reset status
        self.status_label.set_text("Status: Running")
        self.status_label.set_color('#00ff00')
    
    def toggle_obstacles(self, label):
        """Toggle obstacle visibility with visual feedback"""
        self.show_obstacles = not self.show_obstacles
        
        # Update status
        self.status_label.set_text(f"Status: Obstacles {'visible' if self.show_obstacles else 'hidden'}")

    def edit_obstacles(self, event):
        """Open obstacle editor dialog with improved visuals"""
        # Update status
        self.status_label.set_text("Status: Editing obstacles...")
        self.status_label.set_color('#ffff00')
        plt.pause(0.1)  # Allow UI to update
        
        # Create a new figure for obstacle editing with improved styling
        fig_edit = plt.figure(figsize=(10, 8), facecolor='#1c1c1c')
        fig_edit.canvas.manager.set_window_title('Obstacle Editor')
        
        ax = fig_edit.add_subplot(111, facecolor='#2c2c2c')
        ax.set_xlim(0, self.simulator.arena_width)
        ax.set_ylim(0, self.simulator.arena_height)
        ax.set_title('Obstacle Editor', color='white', fontsize=14)
        ax.grid(True, alpha=0.2, color='#555555', linestyle='--')
        
        # Add instructions with styled text
        instructions = fig_edit.text(
            0.5, 0.95, 
            "Click to add obstacles • Right-click to remove • Close window when done",
            ha='center', color='white', fontsize=12,
            transform=fig_edit.transFigure
        )
        
        # Draw current obstacles with enhanced styling
        obstacle_patches = []
        for i, obstacle in enumerate(self.simulator.obstacles):
            pos = obstacle["position"]
            width = obstacle["width"]
            height = obstacle["height"]
            is_wall = obstacle.get("is_wall", False)
            
            # Create rectangle with styling
            color = '#333333' if is_wall else '#555555'
            edge_color = '#666666' if is_wall else '#888888'
            alpha = 0.9 if is_wall else 0.8
            
            rect = Rectangle(
                (pos[0]-width/2, pos[1]-height/2), width, height,
                facecolor=color, edgecolor=edge_color, 
                alpha=alpha, linewidth=1.5
            )
            
            if not is_wall and (width > 3 or height > 3):
                rect.set_hatch('///')
                
            ax.add_patch(rect)
            obstacle_patches.append(rect)
            
            # Add label for significant obstacles
            if width * height > 9 and not is_wall:
                ax.text(
                    pos[0], pos[1], f'Obstacle {i+1}',
                    ha='center', va='center',
                    color='white', fontsize=8, fontweight='bold'
                )
        
        # Status text for the editor
        status_text = fig_edit.text(
            0.5, 0.02, "Ready to edit obstacles",
            ha='center', color='white', fontsize=10,
            transform=fig_edit.transFigure
        )
        
        # Add obstacle on click
        def onclick(event):
            if event.inaxes != ax:
                return
                
            if event.button == 1:  # Left click to add
                # Create obstacle at click position
                new_obstacle = {
                    "position": [event.xdata, event.ydata],
                    "width": 2.0,  # Default width
                    "height": 2.0  # Default height
                }
                self.simulator.obstacles.append(new_obstacle)
                status_text.set_text(f"Added obstacle at ({event.xdata:.1f}, {event.ydata:.1f})")
            
            elif event.button == 3:  # Right click to remove
                # Find closest obstacle to click
                min_dist = float('inf')
                min_idx = -1
                for i, obstacle in enumerate(self.simulator.obstacles):
                    pos = obstacle["position"]
                    dist = np.sqrt((pos[0] - event.xdata)**2 + (pos[1] - event.ydata)**2)
                    if dist < min_dist:
                        min_dist = dist
                        min_idx = i
                
                # Remove if within threshold
                if min_dist < 3.0 and min_idx >= 0:
                    removed = self.simulator.obstacles.pop(min_idx)
                    status_text.set_text(f"Removed obstacle at ({removed['position'][0]:.1f}, {removed['position'][1]:.1f})")
            
            # Redraw
            ax.clear()
            ax.set_xlim(0, self.simulator.arena_width)
            ax.set_ylim(0, self.simulator.arena_height)
            ax.set_title('Obstacle Editor', color='white', fontsize=14)
            ax.grid(True, alpha=0.2, color='#555555', linestyle='--')
            
            for i, obstacle in enumerate(self.simulator.obstacles):
                pos = obstacle["position"]
                width = obstacle["width"]
                height = obstacle["height"]
                is_wall = obstacle.get("is_wall", False)
                
                color = '#333333' if is_wall else '#555555'
                edge_color = '#666666' if is_wall else '#888888'
                alpha = 0.9 if is_wall else 0.8
                
                rect = Rectangle(
                    (pos[0]-width/2, pos[1]-height/2), width, height,
                    facecolor=color, edgecolor=edge_color, 
                    alpha=alpha, linewidth=1.5
                )
                
                if not is_wall and (width > 3 or height > 3):
                    rect.set_hatch('///')
                    
                ax.add_patch(rect)
                
                if width * height > 9 and not is_wall:
                    ax.text(
                        pos[0], pos[1], f'Obstacle {i+1}',
                        ha='center', va='center',
                        color='white', fontsize=8, fontweight='bold'
                    )
            
            plt.draw()
        
        # Connect click event
        fig_edit.canvas.mpl_connect('button_press_event', onclick)
        
        # Show the editor
        plt.show()
        
        # After editor is closed, update main GUI status
        self.status_label.set_text("Status: Obstacle editing complete")
        self.status_label.set_color('#00ff00')
    
    def run(self):
        """Run the simulation GUI with splash screen"""
        # Show splash screen
        self._show_splash_screen()
        
        # Make sure the window is in the foreground
        self.fig.canvas.manager.window.attributes('-topmost', 1)
        self.fig.canvas.manager.window.attributes('-topmost', 0)
        
        # Show the plot and start the animation
        plt.show()
        
        return self.anim
    
    def _show_splash_screen(self):
        """Show a brief splash screen during loading"""
        # Create splash figure
        splash_fig = plt.figure(figsize=(6, 4), facecolor='#1c1c1c')
        splash_fig.canvas.manager.set_window_title('Crowd Flow Simulator')
        
        # Hide axes
        ax = splash_fig.add_subplot(111)
        ax.axis('off')
        
        # Title text
        ax.text(
            0.5, 0.6, 'ADVANCED CROWD FLOW SIMULATOR',
            ha='center', va='center',
            color='white', fontsize=16, fontweight='bold'
        )
        
        # Version info
        ax.text(
            0.5, 0.5, 'Version 2.0',
            ha='center', va='center',
            color='#aaaaaa', fontsize=12
        )
        
        # Loading message
        loading_text = ax.text(
            0.5, 0.3, 'Loading simulation...',
            ha='center', va='center',
            color='#00a8ff', fontsize=10
        )
        
        # Show splash
        splash_fig.canvas.draw()
        plt.pause(0.5)
        
        # Loading animation
        dots = ''
        for i in range(5):
            dots += '.'
            loading_text.set_text(f'Loading simulation{dots}')
            splash_fig.canvas.draw()
            plt.pause(0.2)
        
        # Close splash
        plt.close(splash_fig)

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