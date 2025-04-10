import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from sklearn.cluster import DBSCAN
from scipy.spatial import Voronoi, voronoi_plot_2d
import json
import os
import matplotlib.patches as patches

class CrowdFlowAnalyzer:
    def __init__(self, simulator):
        """Initialize with a simulator instance"""
        self.simulator = simulator
        self.density_history = []
        self.velocity_field_history = []
        self.grid_size = 50  # Grid size for density calculation
        
        # Set up grid for density and velocity calculations
        self.x_grid = np.linspace(0, simulator.arena_width, self.grid_size)
        self.y_grid = np.linspace(0, simulator.arena_height, self.grid_size)
        self.X, self.Y = np.meshgrid(self.x_grid, self.y_grid)
        
        # Track statistics
        self.bottleneck_history = []
        self.lane_history = []
        self.cluster_history = []
        self.evacuation_rate_history = []
        self.active_agents_history = []
        
    def record_frame(self):
        """Record current simulation state for analysis"""
        active_positions = self.simulator.positions[self.simulator.active_agents]
        active_velocities = self.simulator.velocities[self.simulator.active_agents]
        
        # Track number of active agents
        self.active_agents_history.append(np.sum(self.simulator.active_agents))
        
        # Calculate density field
        density = self._calculate_density_field(active_positions)
        self.density_history.append(density)
        
        # Calculate velocity field
        vx_field, vy_field = self._calculate_velocity_field(active_positions, active_velocities)
        self.velocity_field_history.append((vx_field, vy_field))
        
        # Calculate evacuation rate (agents exiting per time step)
        if len(self.active_agents_history) > 1:
            evac_rate = self.active_agents_history[-2] - self.active_agents_history[-1]
            self.evacuation_rate_history.append(max(0, evac_rate))
        else:
            self.evacuation_rate_history.append(0)
        
        # Record emergent behaviors
        lanes = self._detect_lanes(active_positions, active_velocities)
        self.lane_history.append(len(lanes))
        
        clusters = self._detect_clusters(active_positions)
        self.cluster_history.append(len(clusters))
        
        bottlenecks = self._detect_bottlenecks(active_positions, active_velocities)
        self.bottleneck_history.append(bottlenecks if isinstance(bottlenecks, int) else 0)
        
    def _calculate_density_field(self, positions):
        """Calculate density field from agent positions"""
        density = np.zeros((self.grid_size, self.grid_size))
        
        # Create density field using a 2D histogram
        if len(positions) > 0:
            hist, _, _ = np.histogram2d(
                positions[:, 0], positions[:, 1], 
                bins=[self.grid_size, self.grid_size], 
                range=[[0, self.simulator.arena_width], [0, self.simulator.arena_height]]
            )
            
            # Smooth density field with Gaussian filter
            density = gaussian_filter(hist, sigma=1.0)
            
        return density
    
    def _calculate_velocity_field(self, positions, velocities):
        """Calculate average velocity field from agent positions and velocities"""
        vx_field = np.zeros((self.grid_size, self.grid_size))
        vy_field = np.zeros((self.grid_size, self.grid_size))
        count_field = np.zeros((self.grid_size, self.grid_size))
        
        if len(positions) > 0:
            # Find grid indices for each agent
            x_indices = np.clip(np.floor(positions[:, 0] / self.simulator.arena_width * (self.grid_size-1)).astype(int), 
                               0, self.grid_size-1)
            y_indices = np.clip(np.floor(positions[:, 1] / self.simulator.arena_height * (self.grid_size-1)).astype(int), 
                               0, self.grid_size-1)
            
            # Sum velocities for each grid cell
            for i in range(len(positions)):
                x_idx = x_indices[i]
                y_idx = y_indices[i]
                vx_field[y_idx, x_idx] += velocities[i, 0]
                vy_field[y_idx, x_idx] += velocities[i, 1]
                count_field[y_idx, x_idx] += 1
            
            # Average velocities
            nonzero = count_field > 0
            vx_field[nonzero] /= count_field[nonzero]
            vy_field[nonzero] /= count_field[nonzero]
        
        return vx_field, vy_field
    
    def detect_emergent_behavior(self):
        """Detect emergent behaviors in the crowd"""
        if not any(self.simulator.active_agents):
            return "No active agents to analyze"
        
        active_positions = self.simulator.positions[self.simulator.active_agents]
        active_velocities = self.simulator.velocities[self.simulator.active_agents]
        
        results = []
        
        # Detect lane formation (aligned velocities)
        lanes = self._detect_lanes(active_positions, active_velocities)
        if lanes:
            results.append(f"Lane formation detected: {len(lanes)} distinct lanes")
        
        # Detect clustering/grouping
        clusters = self._detect_clusters(active_positions)
        if clusters:
            results.append(f"Clustering detected: {len(clusters)} distinct clusters")
        
        # Detect bottlenecks
        bottlenecks = self._detect_bottlenecks(active_positions, active_velocities)
        if bottlenecks:
            results.append(f"Bottleneck detected at {bottlenecks} regions")
        
        # Analyze crowd density
        high_density_areas = self._analyze_density(active_positions)
        if high_density_areas:
            results.append(f"High density areas detected at {high_density_areas} regions")
        
        # Check for herding behavior
        if self._detect_herding_behavior(active_velocities):
            results.append("Herding behavior detected")
        
        # Check for counter-flow
        if self._detect_counter_flow(active_velocities):
            results.append("Counter-flow patterns detected")
        
        if not results:
            results.append("No significant emergent behaviors detected")
            
        return results
    
    def _detect_lanes(self, positions, velocities):
        """Detect lane formation based on velocity alignment"""
        if len(positions) < 5:
            return []
            
        # Normalize velocities
        speeds = np.linalg.norm(velocities, axis=1)
        nonzero_speeds = speeds > 0.1
        
        if not any(nonzero_speeds):
            return []
            
        normalized_velocities = np.zeros_like(velocities)
        normalized_velocities[nonzero_speeds] = velocities[nonzero_speeds] / speeds[nonzero_speeds, np.newaxis]
        
        # Cluster based on velocity direction
        angle_threshold = 0.9  # Cosine similarity threshold
        lanes = []
        
        for i in range(len(positions)):
            if not nonzero_speeds[i]:
                continue
                
            # Check if agent belongs to an existing lane
            lane_found = False
            
            for lane in lanes:
                # Calculate average direction of the lane
                avg_direction = np.mean([normalized_velocities[j] for j in lane], axis=0)
                avg_direction /= np.linalg.norm(avg_direction) if np.linalg.norm(avg_direction) > 0 else 1
                
                # Check if agent's direction aligns with lane
                similarity = np.dot(normalized_velocities[i], avg_direction)
                if similarity > angle_threshold:
                    lane.append(i)
                    lane_found = True
                    break
            
            # If not part of existing lane, create new lane
            if not lane_found:
                lanes.append([i])
        
        # Filter lanes with at least 3 agents
        lanes = [lane for lane in lanes if len(lane) >= 3]
        
        return lanes
    
    def _detect_clusters(self, positions):
        """Detect spatial clusters of agents"""
        if len(positions) < 5:
            return []
            
        # Use DBSCAN for clustering
        clustering = DBSCAN(eps=2.0, min_samples=3).fit(positions)
        labels = clustering.labels_
        
        # Count clusters (excluding noise points labeled as -1)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        
        clusters = []
        for i in range(n_clusters):
            cluster_indices = np.where(labels == i)[0]
            clusters.append(cluster_indices)
            
        return clusters
    
    def _detect_bottlenecks(self, positions, velocities):
        """Detect bottlenecks based on density and reduced velocity"""
        if len(positions) < 5:
            return 0
            
        # Calculate local density for each agent
        densities = self._calculate_local_densities(positions)
        
        # Calculate local average speed
        speeds = np.linalg.norm(velocities, axis=1)
        
        # Look for areas with high density and low speeds
        bottleneck_threshold = 0.8  # For normalized values
        
        # Normalize densities and speeds
        if np.max(densities) > 0:
            norm_densities = densities / np.max(densities)
        else:
            norm_densities = densities
            
        if np.max(speeds) > 0:
            norm_speeds = 1 - (speeds / np.max(speeds))  # Invert so high value = low speed
        else:
            norm_speeds = np.ones_like(speeds)
        
        # Identify bottleneck regions (high density, low speed)
        bottleneck_score = norm_densities * norm_speeds
        bottleneck_indices = np.where(bottleneck_score > bottleneck_threshold)[0]
        
        if len(bottleneck_indices) < 3:
            return 0
        
        # Cluster bottleneck points to find distinct regions
        if len(bottleneck_indices) > 0:
            bottleneck_positions = positions[bottleneck_indices]
            clustering = DBSCAN(eps=3.0, min_samples=2).fit(bottleneck_positions)
            labels = clustering.labels_
            
            # Count distinct bottleneck regions
            n_regions = len(set(labels)) - (1 if -1 in labels else 0)
            return n_regions
        
        return 0
    
    def _detect_herding_behavior(self, velocities):
        """Detect if agents are showing herding behavior (moving in similar directions)"""
        if len(velocities) < 5:
            return False
            
        # Calculate angular distribution of velocities
        speeds = np.linalg.norm(velocities, axis=1)
        nonzero_speeds = speeds > 0.1
        
        if not any(nonzero_speeds):
            return False
            
        normalized_velocities = np.zeros_like(velocities)
        normalized_velocities[nonzero_speeds] = velocities[nonzero_speeds] / speeds[nonzero_speeds, np.newaxis]
        
        # Convert to angles
        angles = np.arctan2(normalized_velocities[:, 1], normalized_velocities[:, 0])
        
        # Calculate circular variance (1 - mean resultant length)
        x_mean = np.mean(np.cos(angles))
        y_mean = np.mean(np.sin(angles))
        r = np.sqrt(x_mean**2 + y_mean**2)
        
        # Low variance indicates herding (agents moving in similar directions)
        herding_threshold = 0.7  # Threshold for mean resultant length
        return r > herding_threshold
    
    def _detect_counter_flow(self, velocities):
        """Detect counter-flow patterns (agents moving in opposite directions)"""
        if len(velocities) < 10:
            return False
            
        # Calculate angular distribution of velocities
        speeds = np.linalg.norm(velocities, axis=1)
        nonzero_speeds = speeds > 0.1
        
        if np.sum(nonzero_speeds) < 10:
            return False
            
        normalized_velocities = np.zeros_like(velocities)
        normalized_velocities[nonzero_speeds] = velocities[nonzero_speeds] / speeds[nonzero_speeds, np.newaxis]
        
        # Check if there are distinct groups moving in opposite directions
        from sklearn.cluster import KMeans
        
        # Use K-means to find 2 dominant directions
        kmeans = KMeans(n_clusters=2).fit(normalized_velocities[nonzero_speeds])
        centers = kmeans.cluster_centers_
        
        # Calculate dot product between centers to see if they are in opposite directions
        dot_product = np.dot(centers[0], centers[1])
        
        # If dot product is negative, directions are more than 90 degrees apart
        return dot_product < -0.7  # Threshold for counter-flow detection
    
    def _calculate_local_densities(self, positions):
        """Calculate local density around each agent"""
        densities = np.zeros(len(positions))
        
        if len(positions) < 2:
            return densities
            
        # Use pairwise distances to calculate density
        for i in range(len(positions)):
            # Count agents within radius
            diffs = positions - positions[i]
            distances = np.linalg.norm(diffs, axis=1)
            
            # Agents within 3 units contribute to density
            radius = 3.0
            count = np.sum(distances < radius) - 1  # Exclude self
            
            # Density is proportional to count divided by area
            area = np.pi * radius**2
            densities[i] = count / area
            
        return densities
    
    def _analyze_density(self, positions):
        """Analyze crowd density to find high-density areas"""
        if len(positions) < 5:
            return 0
            
        # Calculate density field
        density = self._calculate_density_field(positions)
        
        # Find high density areas (top 10% of density values)
        threshold = np.percentile(density, 90)
        high_density = density > threshold
        
        # Count connected regions of high density
        from scipy.ndimage import label
        labeled_array, num_features = label(high_density)
        
        return num_features if num_features > 0 else 0
    
    def visualize_density(self, frame=-1, save_path=None):
        """Visualize crowd density at a specific frame"""
        if not self.density_history:
            print("No density data available")
            return
            
        if frame == -1:
            density = self.density_history[-1]
        else:
            density = self.density_history[frame]
            
        fig, ax = plt.subplots(figsize=(12, 10))
        contour = ax.contourf(self.X, self.Y, density, cmap='viridis', levels=20)
        plt.colorbar(contour, label='Agent Density (agents/m²)')
        
        # Plot agent positions
        active_positions = self.simulator.positions[self.simulator.active_agents]
        ax.scatter(active_positions[:, 0], active_positions[:, 1], 
                   c='red', s=20, alpha=0.7, label='Agents')
        
        # Draw floor plan elements
        self._draw_floor_plan(ax)
        
        ax.set_title('Crowd Density Map', fontsize=14)
        ax.set_xlabel('X Position (m)', fontsize=12)
        ax.set_ylabel('Y Position (m)', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Density map saved to {save_path}")
        
        plt.show()
    
    def visualize_velocity_field(self, frame=-1, save_path=None):
        """Visualize velocity field at a specific frame"""
        if not self.velocity_field_history:
            print("No velocity field data available")
            return
            
        if frame == -1:
            vx_field, vy_field = self.velocity_field_history[-1]
        else:
            vx_field, vy_field = self.velocity_field_history[frame]
            
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Calculate magnitude for color mapping
        magnitude = np.sqrt(vx_field**2 + vy_field**2)
        
        # Plot velocity field as streamlines
        streamplot = ax.streamplot(self.X, self.Y, vx_field, vy_field, 
                       density=1.5, color=magnitude, 
                       cmap='coolwarm', linewidth=1.5*magnitude/np.max(magnitude+0.001))
        
        plt.colorbar(streamplot.lines, label='Velocity Magnitude (m/s)')
        
        # Plot agent positions
        active_positions = self.simulator.positions[self.simulator.active_agents]
        ax.scatter(active_positions[:, 0], active_positions[:, 1], 
                   c='black', s=20, alpha=0.7, label='Agents')
        
        # Draw floor plan elements
        self._draw_floor_plan(ax)
        
        ax.set_title('Crowd Flow Velocity Field', fontsize=14)
        ax.set_xlabel('X Position (m)', fontsize=12)
        ax.set_ylabel('Y Position (m)', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Velocity field map saved to {save_path}")
        
        plt.show()
    
    def _draw_floor_plan(self, ax):
        """Draw floor plan elements (exits, obstacles, walls) on the given axes"""
        # Draw exits
        for exit_info in self.simulator.exits:
            exit_pos = np.array(exit_info["position"])
            exit_width = exit_info["width"]
            
            # Different visualization based on exit position
            if exit_pos[0] == 0:  # Exit on left wall
                rect = patches.Rectangle((-0.2, exit_pos[1] - exit_width/2), 0.2, exit_width, 
                                 color='green', alpha=0.7)
                ax.add_patch(rect)
                ax.text(2, exit_pos[1], 'EXIT', ha='left', va='center', color='green', fontsize=12)
                
            elif exit_pos[0] == self.simulator.arena_width:  # Exit on right wall
                rect = patches.Rectangle((self.simulator.arena_width, exit_pos[1] - exit_width/2), 0.2, exit_width, 
                                 color='green', alpha=0.7)
                ax.add_patch(rect)
                ax.text(self.simulator.arena_width-2, exit_pos[1], 'EXIT', 
                       ha='right', va='center', color='green', fontsize=12)
                
            elif exit_pos[1] == 0:  # Exit on bottom wall
                rect = patches.Rectangle((exit_pos[0] - exit_width/2, -0.2), exit_width, 0.2, 
                                 color='green', alpha=0.7)
                ax.add_patch(rect)
                ax.text(exit_pos[0], 2, 'EXIT', ha='center', va='bottom', color='green', fontsize=12)
                
            elif exit_pos[1] == self.simulator.arena_height:  # Exit on top wall
                rect = patches.Rectangle((exit_pos[0] - exit_width/2, self.simulator.arena_height), exit_width, 0.2, 
                                 color='green', alpha=0.7)
                ax.add_patch(rect)
                ax.text(exit_pos[0], self.simulator.arena_height-2, 'EXIT', 
                       ha='center', va='top', color='green', fontsize=12)
        
        # Draw obstacles
        for obstacle in self.simulator.obstacles:
            pos = obstacle["position"]
            width = obstacle["width"]
            height = obstacle["height"]
            
            if obstacle.get("is_wall", False):
                color = 'black'
                alpha = 0.9
            else:
                color = 'gray'
                alpha = 0.7
                
            rect = patches.Rectangle((pos[0]-width/2, pos[1]-height/2), width, height, 
                           color=color, alpha=alpha)
            ax.add_patch(rect)
            
            # Label significant obstacles
            if width > 3 and height > 3 and not obstacle.get("is_wall", False):
                ax.text(pos[0], pos[1], 'Obstacle', ha='center', va='center', 
                       color='white', fontsize=10)
    
    def visualize_voronoi_diagram(self, save_path=None):
        """Visualize Voronoi diagram to analyze space usage"""
        active_positions = self.simulator.positions[self.simulator.active_agents]
        
        if len(active_positions) < 3:
            print("Not enough agents for Voronoi diagram")
            return
            
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Create Voronoi diagram
        vor = Voronoi(active_positions)
        voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_colors='blue', line_width=1, line_alpha=0.6, point_size=5)
        
        # Draw floor plan elements
        self._draw_floor_plan(ax)
        
        # Plot agent positions
        ax.scatter(active_positions[:, 0], active_positions[:, 1], 
                  c='red', s=20, alpha=0.7, label='Agents')
        
        ax.set_title('Voronoi Diagram of Agent Positions', fontsize=14)
        ax.set_xlabel('X Position (m)', fontsize=12)
        ax.set_ylabel('Y Position (m)', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Voronoi diagram saved to {save_path}")
        
        plt.show()
    
    def visualize_evacuation_timeline(self, save_path=None):
        """Visualize evacuation timeline and statistics"""
        if not self.active_agents_history:
            print("No evacuation data available")
            return
            
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})
        
        # Plot number of active agents over time
        time_steps = np.arange(len(self.active_agents_history))
        ax1.plot(time_steps, self.active_agents_history, 'b-', linewidth=2, label='Agents Remaining')
        
        # Plot evacuation rate
        if self.evacuation_rate_history:
            ax2.bar(time_steps[1:], self.evacuation_rate_history, color='green', alpha=0.7, label='Evacuation Rate')
            ax2.set_ylabel('Agents Evacuated\nper Time Step', fontsize=12)
            ax2.set_xlabel('Time Step', fontsize=12)
            ax2.grid(True, alpha=0.3)
            ax2.legend()
        
        # Plot emergent behaviors if available
        if self.bottleneck_history:
            ax1_twin = ax1.twinx()
            ax1_twin.plot(time_steps[:len(self.bottleneck_history)], self.bottleneck_history, 
                         'r--', linewidth=1.5, label='Bottlenecks')
            ax1_twin.set_ylabel('Number of Bottlenecks', color='r', fontsize=12)
            ax1_twin.tick_params(axis='y', colors='r')
            ax1_twin.legend(loc='upper right')
        
        ax1.set_ylabel('Number of Agents', fontsize=12)
        ax1.set_title('Evacuation Timeline', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper left')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Evacuation timeline saved to {save_path}")
        
        plt.show()
    
    def create_custom_layout(self, output_file):
        """Create a custom layout JSON file interactively"""
        print("=== Interactive Custom Floor Plan Creator ===")
        
        layout = {
            "exits": [],
            "obstacles": [],
            "walls": []
        }
        
        # Get arena dimensions
        width = float(input("Enter arena width (m): ") or self.simulator.arena_width)
        height = float(input("Enter arena height (m): ") or self.simulator.arena_height)
        
        # Configure exits
        print("\n--- Adding Exits ---")
        print("Exits can be placed on the boundaries of the arena")
        while True:
            add_exit = input("Add an exit? (y/n): ").lower()
            if add_exit != 'y':
                break
                
            wall = input("Which wall? (left/right/top/bottom): ").lower()
            
            if wall == 'left':
                x_pos = 0
                y_pos = float(input(f"Enter y-position (0-{height}): "))
            elif wall == 'right':
                x_pos = width
                y_pos = float(input(f"Enter y-position (0-{height}): "))
            elif wall == 'top':
                y_pos = height
                x_pos = float(input(f"Enter x-position (0-{width}): "))
            elif wall == 'bottom':
                y_pos = 0
                x_pos = float(input(f"Enter x-position (0-{width}): "))
            else:
                print("Invalid wall selection. Try again.")
                continue
                
            exit_width = float(input("Enter exit width (m): "))
            
            layout["exits"].append({
                "position": [x_pos, y_pos],
                "width": exit_width
            })
        
        # Configure obstacles (furniture, columns, etc.)
        print("\n--- Adding Obstacles ---")
        print("Obstacles represent furniture, columns, or other obstructions")
        while True:
            add_obstacle = input("Add an obstacle? (y/n): ").lower()
            if add_obstacle != 'y':
                break
                
            x_pos = float(input(f"Enter x-position (0-{width}): "))
            y_pos = float(input(f"Enter y-position (0-{height}): "))
            obs_width = float(input("Enter obstacle width (m): "))
            obs_height = float(input("Enter obstacle height (m): "))
            
            layout["obstacles"].append({
                "position": [x_pos, y_pos],
                "width": obs_width,
                "height": obs_height
            })
        
        # Configure internal walls (optional)
        print("\n--- Adding Internal Walls ---")
        print("Internal walls divide the space but are not arena boundaries")
        while True:
            add_wall = input("Add an internal wall? (y/n): ").lower()
            if add_wall != 'y':
                break
                
            x_pos = float(input(f"Enter wall center x-position (0-{width}): "))
            y_pos = float(input(f"Enter wall center y-position (0-{height}): "))
            wall_width = float(input("Enter wall width (m): "))
            wall_height = float(input("Enter wall height (m): "))
            
            layout["walls"].append({
                "position": [x_pos, y_pos],
                "width": wall_width,
                "height": wall_height,
                "is_wall": True
            })
        
        # Save layout to file
        with open(output_file, 'w') as f:
            json.dump(layout, f, indent=4)
            
        print(f"\nCustom layout saved to {output_file}")
        print("You can use this file with the simulator using the --custom_layout parameter")
        
        return layout
    
    def preview_layout(self, layout_file=None):
        """Preview a floor plan layout"""
        if layout_file:
            # Load layout from file
            with open(layout_file, 'r') as f:
                layout = json.load(f)
        else:
            # Use the current simulator's configuration
            layout = {
                "exits": self.simulator.exits,
                "obstacles": [obs for obs in self.simulator.obstacles if not obs.get("is_wall", False)],
                "walls": [wall for wall in self.simulator.obstacles if wall.get("is_wall", False)]
            }
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Determine arena dimensions
        arena_width = self.simulator.arena_width
        arena_height = self.simulator.arena_height
        
        # Draw arena boundaries
        ax.add_patch(patches.Rectangle((0, 0), arena_width, arena_height, 
                                      fill=False, edgecolor='black', linewidth=2))
        
        # Draw exits
        for exit_info in layout["exits"]:
            exit_pos = np.array(exit_info["position"])
            exit_width = exit_info["width"]
            
            # Different visualization based on exit position
            if exit_pos[0] == 0:  # Exit on left wall
                rect = patches.Rectangle((-0.2, exit_pos[1] - exit_width/2), 0.2, exit_width, 
                                 color='green', alpha=0.7)
                ax.add_patch(rect)
                ax.text(2, exit_pos[1], 'EXIT', ha='left', va='center', color='green', fontsize=12)
                
            elif exit_pos[0] == self.simulator.arena_width:  # Exit on right wall
                rect = patches.Rectangle((self.simulator.arena_width, exit_pos[1] - exit_width/2), 0.2, exit_width, 
                                 color='green', alpha=0.7)
                ax.add_patch(rect)
                ax.text(self.simulator.arena_width-2, exit_pos[1], 'EXIT', 
                       ha='right', va='center', color='green', fontsize=12)
                
            elif exit_pos[1] == 0:  # Exit on bottom wall
                rect = patches.Rectangle((exit_pos[0] - exit_width/2, -0.2), exit_width, 0.2, 
                                 color='green', alpha=0.7)
                ax.add_patch(rect)
                ax.text(exit_pos[0], 2, 'EXIT', ha='center', va='bottom', color='green', fontsize=12)
                
            elif exit_pos[1] == self.simulator.arena_height:  # Exit on top wall
                rect = patches.Rectangle((exit_pos[0] - exit_width/2, self.simulator.arena_height), exit_width, 0.2, 
                                 color='green', alpha=0.7)
                ax.add_patch(rect)
                ax.text(exit_pos[0], self.simulator.arena_height-2, 'EXIT', 
                       ha='center', va='top', color='green', fontsize=12)
        
        # Draw obstacles
        for obstacle in layout.get("obstacles", []):
            pos = obstacle["position"]
            width = obstacle["width"]
            height = obstacle["height"]
            
            rect = patches.Rectangle((pos[0]-width/2, pos[1]-height/2), width, height, 
                           color='gray', alpha=0.7)
            ax.add_patch(rect)
            
            # Label significant obstacles
            if width > 3 and height > 3:
                ax.text(pos[0], pos[1], 'Obstacle', ha='center', va='center', 
                       color='white', fontsize=10)
        
        # Draw internal walls
        for wall in layout.get("walls", []):
            pos = wall["position"]
            width = wall["width"]
            height = wall["height"]
            
            rect = patches.Rectangle((pos[0]-width/2, pos[1]-height/2), width, height, 
                           color='black', alpha=0.9)
            ax.add_patch(rect)
            
            # Label significant walls
            if width > 5 and height > 5:
                ax.text(pos[0], pos[1], 'Wall', ha='center', va='center', 
                       color='white', fontsize=10)
        
        ax.set_title('Floor Plan Layout Preview', fontsize=14)
        ax.set_xlabel('X Position (m)', fontsize=12)
        ax.set_ylabel('Y Position (m)', fontsize=12)
        ax.set_xlim(-1, arena_width + 1)
        ax.set_ylim(-1, arena_height + 1)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def analyze_evacuation_efficiency(self):
        """Analyze the efficiency of evacuation"""
        if not self.active_agents_history:
            return "No evacuation data available"
            
        # Calculate evacuation statistics
        initial_agents = self.active_agents_history[0]
        final_agents = self.active_agents_history[-1]
        evacuated = initial_agents - final_agents
        
        if initial_agents == 0:
            return "No agents to evacuate"
            
        evacuation_percentage = (evacuated / initial_agents) * 100
        
        # Calculate average evacuation rate
        if len(self.evacuation_rate_history) > 0:
            avg_evacuation_rate = np.mean(self.evacuation_rate_history)
        else:
            avg_evacuation_rate = 0
            
        # Find peak evacuation rate
        peak_evacuation_rate = max(self.evacuation_rate_history) if self.evacuation_rate_history else 0
        
        # Time to evacuate half of the agents
        half_time = None
        for i, count in enumerate(self.active_agents_history):
            if count <= initial_agents / 2:
                half_time = i
                break
                
        # Check for evacuation bottlenecks
        bottleneck_count = sum(1 for x in self.bottleneck_history if x > 0)
        bottleneck_percentage = (bottleneck_count / len(self.bottleneck_history)) * 100 if self.bottleneck_history else 0
        
        # Generate report
        report = {
            "initial_agents": initial_agents,
            "evacuated_agents": evacuated,
            "evacuation_percentage": evacuation_percentage,
            "simulation_steps": len(self.active_agents_history),
            "avg_evacuation_rate": avg_evacuation_rate,
            "peak_evacuation_rate": peak_evacuation_rate,
            "half_evacuation_time": half_time,
            "bottleneck_percentage": bottleneck_percentage
        }
        
        return report
    
    def summarize_emergent_behaviors(self):
        """Summarize emergent behaviors observed during simulation"""
        if not self.bottleneck_history:
            return "No data available for emergent behavior analysis"
            
        # Calculate average occurrences
        avg_bottlenecks = np.mean(self.bottleneck_history)
        avg_lanes = np.mean(self.lane_history) if self.lane_history else 0
        avg_clusters = np.mean(self.cluster_history) if self.cluster_history else 0
        
        # Calculate frequency of occurrence
        bottleneck_freq = sum(1 for x in self.bottleneck_history if x > 0) / len(self.bottleneck_history)
        lane_freq = sum(1 for x in self.lane_history if x > 0) / len(self.lane_history) if self.lane_history else 0
        cluster_freq = sum(1 for x in self.cluster_history if x > 0) / len(self.cluster_history) if self.cluster_history else 0
        
        # Generate summary
        summary = {
            "bottlenecks": {
                "average": avg_bottlenecks,
                "frequency": bottleneck_freq,
                "peak": max(self.bottleneck_history) if self.bottleneck_history else 0
            },
            "lanes": {
                "average": avg_lanes,
                "frequency": lane_freq,
                "peak": max(self.lane_history) if self.lane_history else 0
            },
            "clusters": {
                "average": avg_clusters,
                "frequency": cluster_freq,
                "peak": max(self.cluster_history) if self.cluster_history else 0
            }
        }
        
        # Generate descriptive interpretation
        interpretation = []
        
        if avg_bottlenecks > 1:
            interpretation.append(f"Significant bottlenecks occurred in {bottleneck_freq*100:.1f}% of the simulation time")
        elif avg_bottlenecks > 0.5:
            interpretation.append(f"Minor bottlenecks occurred in {bottleneck_freq*100:.1f}% of the simulation time")
        else:
            interpretation.append("No significant bottlenecks were observed")
            
        if avg_lanes > 2:
            interpretation.append(f"Strong lane formation observed with an average of {avg_lanes:.1f} lanes")
        elif avg_lanes > 1:
            interpretation.append(f"Moderate lane formation observed with an average of {avg_lanes:.1f} lanes")
        else:
            interpretation.append("No significant lane formation observed")
            
        if avg_clusters > 3:
            interpretation.append(f"Significant clustering behavior with an average of {avg_clusters:.1f} clusters")
        elif avg_clusters > 1:
            interpretation.append(f"Some clustering behavior with an average of {avg_clusters:.1f} clusters")
        else:
            interpretation.append("No significant clustering observed")
        
        summary["interpretation"] = interpretation
        return summary
    
    def export_analysis(self, filename):
        """Export analysis results to a JSON file"""
        results = {
            "evacuation_efficiency": self.analyze_evacuation_efficiency(),
            "emergent_behaviors": self.summarize_emergent_behaviors(),
            "simulation_parameters": {
                "num_agents": self.simulator.num_agents,
                "arena_width": self.simulator.arena_width,
                "arena_height": self.simulator.arena_height,
                "time_steps": len(self.active_agents_history)
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=4)
            
        print(f"Analysis results exported to {filename}")
        return results
    
    def run_analysis_with_simulator(self, time_steps, visualize=False):
        """Run simulation and analysis together"""
        for t in range(time_steps):
            # Run one step of simulation
            active_count = self.simulator.step()
            
            # Record frame data for analysis
            self.record_frame()
            
            # Stop if all agents have evacuated
            if active_count == 0 and self.simulator.has_exit:
                break
                
            # Visualize intermediate steps (every 10 steps)
            if visualize and t % 10 == 0:
                print(f"Step {t}: {active_count} active agents")
                
        print(f"Analysis completed after {t+1} steps")
        
        # Return analysis summary
        return {
            "evacuation_efficiency": self.analyze_evacuation_efficiency(),
            "emergent_behaviors": self.summarize_emergent_behaviors(),
            "final_active_agents": np.sum(self.simulator.active_agents)
        }

def create_analyzer(simulator):
    """Helper function to create a CrowdFlowAnalyzer for an existing simulator"""
    return CrowdFlowAnalyzer(simulator)

if __name__ == "__main__":
    # This runs if the script is executed directly
    import argparse
    from crowd_flow_simulator import CrowdFlowSimulator
    
    parser = argparse.ArgumentParser(description='Crowd Flow Analyzer')
    parser.add_argument('--num_agents', type=int, default=100, help='Number of agents')
    parser.add_argument('--arena_width', type=float, default=50, help='Width of arena')
    parser.add_argument('--arena_height', type=float, default=50, help='Height of arena')
    parser.add_argument('--time_steps', type=int, default=500, help='Maximum number of simulation steps')
    parser.add_argument('--custom_layout', type=str, default=None, help='Path to JSON file with custom layout')
    parser.add_argument('--create_layout', action='store_true', help='Create a new custom layout')
    parser.add_argument('--output', type=str, default='analysis_results.json', help='Output file for analysis results')
    
    args = parser.parse_args()
    
    # Create a layout if requested
    if args.create_layout:
        analyzer = CrowdFlowAnalyzer(None)  # Temporary analyzer just for layout creation
        layout_file = input("Enter output filename for layout (default: custom_layout.json): ") or "custom_layout.json"
        analyzer.create_custom_layout(layout_file)
        args.custom_layout = layout_file
    
    # Create simulator and analyzer
    simulator = CrowdFlowSimulator(
        num_agents=args.num_agents,
        arena_width=args.arena_width,
        arena_height=args.arena_height,
        time_steps=args.time_steps,
        custom_layout=args.custom_layout
    )
    
    analyzer = CrowdFlowAnalyzer(simulator)
    
    # Preview layout if available
    if args.custom_layout:
        analyzer.preview_layout(args.custom_layout)
    
    # Run simulation with analysis
    results = analyzer.run_analysis_with_simulator(args.time_steps, visualize=True)
    
    # Visualize results
    analyzer.visualize_density()
    analyzer.visualize_velocity_field()
    analyzer.visualize_evacuation_timeline()
    analyzer.visualize_voronoi_diagram()
    
    # Export results
    analyzer.export_analysis(args.output)
    print("Analysis complete!")