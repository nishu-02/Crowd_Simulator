import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from sklearn.cluster import DBSCAN
from scipy.spatial import Voronoi, voronoi_plot_2d

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
        
    def record_frame(self):
        """Record current simulation state for analysis"""
        active_positions = self.simulator.positions[self.simulator.active_agents]
        active_velocities = self.simulator.velocities[self.simulator.active_agents]
        
        # Calculate density field
        density = self._calculate_density_field(active_positions)
        self.density_history.append(density)
        
        # Calculate velocity field
        vx_field, vy_field = self._calculate_velocity_field(active_positions, active_velocities)
        self.velocity_field_history.append((vx_field, vy_field))
        
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
            x_indices = np.floor(positions[:, 0] / self.simulator.arena_width * (self.grid_size-1)).astype(int)
            y_indices = np.floor(positions[:, 1] / self.simulator.arena_height * (self.grid_size-1)).astype(int)
            
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
        if not self.simulator.active_agents.any():
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
                avg_direction /= np.linalg.norm(avg_direction)
                
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
            return []
            
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
            return []
        
        # Cluster bottleneck points to find distinct regions
        if len(bottleneck_indices) > 0:
            bottleneck_positions = positions[bottleneck_indices]
            clustering = DBSCAN(eps=3.0, min_samples=2).fit(bottleneck_positions)
            labels = clustering.labels_
            
            # Count distinct bottleneck regions
            n_regions = len(set(labels)) - (1 if -1 in labels else 0)
            return n_regions
        
        return []
    
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
            return []
            
        # Calculate density field
        density = self._calculate_density_field(positions)
        
        # Find high density areas (top 10% of density values)
        threshold = np.percentile(density, 90)
        high_density = density > threshold
        
        # Count connected regions of high density
        from scipy.ndimage import label
        labeled_array, num_features = label(high_density)
        
        return num_features if num_features > 0 else []
    
    def visualize_density(self, frame=-1):
        """Visualize crowd density at a specific frame"""
        if not self.density_history:
            print("No density data available")
            return
            
        if frame == -1:
            density = self.density_history[-1]
        else:
            density = self.density_history[frame]
            
        plt.figure(figsize=(10, 8))
        plt.contourf(self.X, self.Y, density, cmap='viridis', levels=20)
        plt.colorbar(label='Agent Density')
        
        # Plot agent positions
        active_positions = self.simulator.positions[self.simulator.active_agents]
        plt.scatter(active_positions[:, 0], active_positions[:, 1], 
                   c='red', s=10, alpha=0.7)
        
        # Draw exit if present
        if self.simulator.has_exit:
            exit_pos = self.simulator.exit_position
            exit_width = self.simulator.exit_width
            plt.plot([exit_pos[0], exit_pos[0]], 
                     [exit_pos[1]-exit_width/2, exit_pos[1]+exit_width/2], 
                     'g-', linewidth=3)
            plt.text(exit_pos[0]-2, exit_pos[1], 'EXIT', 
                     ha='right', va='center', color='green', fontsize=12)
        
        # Draw obstacles
        for obstacle in self.simulator.obstacles:
            pos = obstacle["position"]
            width = obstacle["width"]
            height = obstacle["height"]
            plt.gca().add_patch(plt.Rectangle(
                (pos[0]-width/2, pos[1]-height/2), width, height, 
                color='gray', alpha=0.7))
        
        plt.title('Crowd Density Map')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def visualize_velocity_field(self, frame=-1):
        """Visualize velocity field at a specific frame"""
        if not self.velocity_field_history:
            print("No velocity field data available")
            return
            
        if frame == -1:
            vx_field, vy_field = self.velocity_field_history[-1]
        else:
            vx_field, vy_field = self.velocity_field_history[frame]
            
        plt.figure(figsize=(10, 8))
        
        # Calculate magnitude for color mapping
        magnitude = np.sqrt(vx_field**2 + vy_field**2)
        
        # Create a mask for zero velocity
        mask = magnitude > 0.1
        
        # Plot velocity field as streamlines
        plt.streamplot(self.X, self.Y, vx_field, vy_field, 
                       density=1.5, color=magnitude, 
                       cmap='coolwarm', linewidth=1.5*magnitude/np.max(magnitude+0.001))
        
        plt.colorbar(label='Velocity Magnitude')
        
        # Plot agent positions
        active_positions = self.simulator.positions[self.simulator.active_agents]
        plt.scatter(active_positions[:, 0], active_positions[:, 1], 
                   c='black', s=10, alpha=0.7)
        
        # Draw exit if present
        if self.simulator.has_exit:
            exit_pos = self.simulator.exit_position
            exit_width = self.simulator.exit_width
            plt.plot([exit_pos[0], exit_pos[0]], 
                     [exit_pos[1]-exit_width/2, exit_pos[1]+exit_width/2], 
                     'g-', linewidth=3)
            plt.text(exit_pos[0]-2, exit_pos[1], 'EXIT', 
                     ha='right', va='center', color='green', fontsize=12)
        
        # Draw obstacles
        for obstacle in self.simulator.obstacles:
            pos = obstacle["position"]
            width = obstacle["width"]
            height = obstacle["height"]
            plt.gca().add_patch(plt.Rectangle(
                (pos[0]-width/2, pos[1]-height/2), width, height, 
                color='gray', alpha=0.7))
        
        plt.title('Crowd Velocity Field')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def visualize_voronoi_diagram(self):
        """Visualize Voronoi diagram to analyze space usage"""
        active_positions = self.simulator