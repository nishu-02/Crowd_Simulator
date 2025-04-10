import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle, Circle
import argparse
from scipy.spatial import distance
import time
import json
import os

class CrowdFlowSimulator:
    def __init__(self, num_agents=100, arena_width=50, arena_height=50, 
                 time_steps=1000, dt=0.1, custom_layout=None):
        """Initialize the crowd flow simulator
        
        Args:
            num_agents: Number of agents in the simulation
            arena_width: Width of the arena
            arena_height: Height of the arena
            time_steps: Maximum number of simulation steps
            dt: Time step size
            custom_layout: Path to a JSON file with custom layout or a dictionary containing layout data
        """
        self.num_agents = num_agents
        self.arena_width = arena_width
        self.arena_height = arena_height
        self.time_steps = time_steps
        self.dt = dt
        
        # Social force parameters - more realistic values
        self.A = 2.0       # Repulsion strength between agents
        self.B = 1.0       # Repulsion range
        self.A_wall = 10.0  # Wall repulsion strength
        self.B_wall = 0.2   # Wall repulsion range
        self.tau = 0.5     # Relaxation time
        self.agent_radius = 0.3  # Physical radius of agents
        
        # Agent properties - more realistic distribution
        self.positions = np.random.rand(num_agents, 2) * [arena_width, arena_height]
        self.velocities = np.zeros((num_agents, 2))
        
        # More realistic speed distribution based on human walking speeds (meters/second)
        self.desired_speed = np.random.normal(1.4, 0.26, num_agents)  # Mean 1.4 m/s, SD 0.26 m/s
        self.desired_speed = np.clip(self.desired_speed, 0.8, 2.0)  # Clip to realistic range
        
        self.desired_directions = np.zeros((num_agents, 2))
        
        # Initialize defaults before custom layout parsing
        self.exits = []
        self.obstacles = []
        self.has_exit = False
        self.has_obstacles = False
        
        # Parse custom layout if provided
        if custom_layout:
            self._parse_custom_layout(custom_layout)
        else:
            # Default simple layout with single exit
            self.exits.append({
                "position": [arena_width, arena_height/2],
                "width": 3.0
            })
            self.has_exit = True
        
        # Statistics
        self.evacuation_times = []
        self.active_agents = np.ones(num_agents, dtype=bool)  # Track which agents are still in the arena
        
        # Set initial desired directions (toward closest exit if exists)
        self._update_desired_directions()
        
        # Ensure agents don't start inside obstacles
        self._relocate_agents_from_obstacles()
    
    def _parse_custom_layout(self, layout):
        """Parse custom layout from file or dictionary"""
        # If layout is a string, assume it's a file path
        if isinstance(layout, str):
            if os.path.exists(layout):
                with open(layout, 'r') as f:
                    layout_data = json.load(f)
            else:
                raise FileNotFoundError(f"Custom layout file not found: {layout}")
        else:
            # Assume it's already a dictionary
            layout_data = layout
        
        # Parse exits
        if 'exits' in layout_data and layout_data['exits']:
            self.exits = layout_data['exits']
            self.has_exit = True
        
        # Parse obstacles
        if 'obstacles' in layout_data and layout_data['obstacles']:
            self.obstacles = layout_data['obstacles']
            self.has_obstacles = True
            
        # Parse walls (special type of obstacles that define the arena boundaries)
        if 'walls' in layout_data and layout_data['walls']:
            for wall in layout_data['walls']:
                self.obstacles.append({
                    "position": wall["position"],
                    "width": wall["width"],
                    "height": wall["height"],
                    "is_wall": True
                })
            self.has_obstacles = True
    
    def _relocate_agents_from_obstacles(self):
        """Ensure agents don't start inside obstacles"""
        for i in range(self.num_agents):
            # Check if agent is inside any obstacle
            while self._is_inside_obstacle(self.positions[i]):
                # Relocate to a random position
                self.positions[i] = np.random.rand(2) * [self.arena_width, self.arena_height]
    
    def _is_inside_obstacle(self, position):
        """Check if a position is inside any obstacle"""
        for obstacle in self.obstacles:
            pos = np.array(obstacle["position"])
            width = obstacle["width"]
            height = obstacle["height"]
            
            # Check if position is inside the rectangle
            if (abs(position[0] - pos[0]) < width/2 and 
                abs(position[1] - pos[1]) < height/2):
                return True
                
        return False
    
    def _find_closest_exit(self, position):
        """Find the closest exit to a given position"""
        if not self.has_exit or not self.exits:
            return None
            
        closest_exit = None
        min_distance = float('inf')
        
        for exit_info in self.exits:
            exit_pos = np.array(exit_info["position"])
            exit_width = exit_info["width"]
            
            # Calculate distance to exit
            # For horizontal exits (on right/left walls)
            if exit_pos[0] == 0 or exit_pos[0] == self.arena_width:
                # Find closest point on exit line segment
                y_on_exit = np.clip(position[1], 
                                    exit_pos[1] - exit_width/2, 
                                    exit_pos[1] + exit_width/2)
                exit_point = np.array([exit_pos[0], y_on_exit])
            # For vertical exits (on top/bottom walls)
            elif exit_pos[1] == 0 or exit_pos[1] == self.arena_height:
                x_on_exit = np.clip(position[0], 
                                    exit_pos[0] - exit_width/2, 
                                    exit_pos[0] + exit_width/2)
                exit_point = np.array([x_on_exit, exit_pos[1]])
            else:
                # Interior exit (not implemented)
                exit_point = exit_pos
            
            dist = np.linalg.norm(position - exit_point)
            
            if dist < min_distance:
                min_distance = dist
                closest_exit = exit_info
                
        return closest_exit
    
    def _update_desired_directions(self):
        """Update desired directions for all agents"""
        for i in range(self.num_agents):
            if not self.active_agents[i]:
                continue
                
            if self.has_exit:
                # Find closest exit
                closest_exit = self._find_closest_exit(self.positions[i])
                
                if closest_exit:
                    exit_pos = np.array(closest_exit["position"])
                    exit_width = closest_exit["width"]
                    
                    # Determine target point on exit
                    # For horizontal exits (on right/left walls)
                    if exit_pos[0] == 0 or exit_pos[0] == self.arena_width:
                        y_on_exit = np.clip(self.positions[i, 1], 
                                           exit_pos[1] - exit_width/2, 
                                           exit_pos[1] + exit_width/2)
                        target_point = np.array([exit_pos[0], y_on_exit])
                    # For vertical exits (on top/bottom walls)
                    elif exit_pos[1] == 0 or exit_pos[1] == self.arena_height:
                        x_on_exit = np.clip(self.positions[i, 0], 
                                           exit_pos[0] - exit_width/2, 
                                           exit_pos[0] + exit_width/2)
                        target_point = np.array([x_on_exit, exit_pos[1]])
                    else:
                        # Interior exit (not implemented)
                        target_point = exit_pos
                    
                    # Direction toward exit
                    diff = target_point - self.positions[i]
                    distance = np.linalg.norm(diff)
                    if distance > 0:
                        self.desired_directions[i] = diff / distance
                else:
                    # Random direction if no exit found
                    angle = 2 * np.pi * np.random.rand()
                    self.desired_directions[i] = np.array([np.cos(angle), np.sin(angle)])
            else:
                # Random direction if no exit
                angle = 2 * np.pi * np.random.rand()
                self.desired_directions[i] = np.array([np.cos(angle), np.sin(angle)])
    
    def _compute_wall_force(self, position):
        """Compute force from arena walls"""
        force = np.zeros(2)
        
        # Distance to arena walls
        d_left = position[0]
        d_right = self.arena_width - position[0]
        d_bottom = position[1]
        d_top = self.arena_height - position[1]
        
        # Apply forces from walls
        if d_left < 2:
            force[0] += self.A_wall * np.exp(-d_left / self.B_wall)
        if d_right < 2:
            force[0] -= self.A_wall * np.exp(-d_right / self.B_wall)
        if d_bottom < 2:
            force[1] += self.A_wall * np.exp(-d_bottom / self.B_wall)
        if d_top < 2:
            force[1] -= self.A_wall * np.exp(-d_top / self.B_wall)
            
        return force
    
    def _compute_obstacle_force(self, position):
        """Compute force from obstacles"""
        force = np.zeros(2)
        
        for obstacle in self.obstacles:
            # Find closest point on rectangle to agent
            obs_pos = np.array(obstacle["position"])
            half_width = obstacle["width"] / 2
            half_height = obstacle["height"] / 2
            
            # Compute distance to obstacle
            dx = max(abs(position[0] - obs_pos[0]) - half_width, 0)
            dy = max(abs(position[1] - obs_pos[1]) - half_height, 0)
            distance = np.sqrt(dx**2 + dy**2)
            
            if distance < 2:  # Only apply force if close enough
                # If the agent is inside the obstacle, push it out strongly
                if distance == 0:
                    # Find which side is closest
                    sides_dist = [
                        half_width - abs(position[0] - obs_pos[0]),  # distance to vertical sides
                        half_height - abs(position[1] - obs_pos[1])  # distance to horizontal sides
                    ]
                    if sides_dist[0] < sides_dist[1]:
                        # Push horizontally
                        direction = np.array([1, 0]) if position[0] < obs_pos[0] else np.array([-1, 0])
                    else:
                        # Push vertically
                        direction = np.array([0, 1]) if position[1] < obs_pos[1] else np.array([0, -1])
                    
                    force += direction * self.A_wall * 5.0  # Stronger force to push out
                else:
                    # Direction away from obstacle
                    dx_sign = 1 if position[0] > obs_pos[0] else -1
                    dy_sign = 1 if position[1] > obs_pos[1] else -1
                    
                    if dx == 0:  # Agent is aligned vertically with obstacle
                        direction = np.array([0, dy_sign])
                    elif dy == 0:  # Agent is aligned horizontally with obstacle
                        direction = np.array([dx_sign, 0])
                    else:
                        direction = np.array([dx_sign * dx, dy_sign * dy])
                        direction = direction / np.linalg.norm(direction)
                    
                    # Apply force - stronger for walls
                    force_multiplier = 2.0 if obstacle.get("is_wall", False) else 1.0
                    force += direction * self.A_wall * np.exp(-distance / self.B_wall) * force_multiplier
                
        return force
    
    def _check_exit(self, position):
        """Check if agent has reached any exit"""
        if not self.has_exit:
            return False
            
        for exit_info in self.exits:
            exit_pos = np.array(exit_info["position"])
            exit_width = exit_info["width"]
            
            # Check different types of exits based on position
            # Horizontal exits (on right/left walls)
            if exit_pos[0] == 0 or exit_pos[0] == self.arena_width:
                x_at_boundary = (exit_pos[0] == 0 and position[0] < 0.5) or \
                               (exit_pos[0] == self.arena_width and position[0] > self.arena_width - 0.5)
                y_in_exit = abs(position[1] - exit_pos[1]) < exit_width / 2
                if x_at_boundary and y_in_exit:
                    return True
            
            # Vertical exits (on top/bottom walls)
            elif exit_pos[1] == 0 or exit_pos[1] == self.arena_height:
                y_at_boundary = (exit_pos[1] == 0 and position[1] < 0.5) or \
                               (exit_pos[1] == self.arena_height and position[1] > self.arena_height - 0.5)
                x_in_exit = abs(position[0] - exit_pos[0]) < exit_width / 2
                if y_at_boundary and x_in_exit:
                    return True
        
        return False
    
    def step(self):
        """Perform one simulation step"""
        forces = np.zeros((self.num_agents, 2))
        
        # Compute social forces between agents
        for i in range(self.num_agents):
            if not self.active_agents[i]:
                continue
                
            # Desired force (tendency to move toward goal)
            desired_velocity = self.desired_directions[i] * self.desired_speed[i]
            forces[i] += (desired_velocity - self.velocities[i]) / self.tau
            
            # Wall forces (from arena boundaries)
            forces[i] += self._compute_wall_force(self.positions[i])
            
            # Obstacle forces
            if self.has_obstacles:
                forces[i] += self._compute_obstacle_force(self.positions[i])
            
            # Agent-agent interactions
            for j in range(self.num_agents):
                if i != j and self.active_agents[j]:
                    diff = self.positions[j] - self.positions[i]
                    distance = np.linalg.norm(diff)
                    
                    # Skip if too far away
                    if distance > 5:
                        continue
                        
                    if distance > 0:
                        direction = diff / distance
                        # Repulsive force
                        forces[i] -= self.A * np.exp((2*self.agent_radius - distance) / self.B) * direction
                        
                        # Add contact force if very close (for collision)
                        if distance < 2 * self.agent_radius:
                            contact_force = max(0, 2*self.agent_radius - distance) * direction * 20
                            forces[i] -= contact_force
        
        # Update velocities and positions
        self.velocities = self.velocities + forces * self.dt
        
        # Limit velocities to a maximum speed (based on real human running speeds)
        speeds = np.linalg.norm(self.velocities, axis=1)
        max_speed = 3.0  # Maximum allowed speed (3 m/s is a fast walking / slow running pace)
        for i in range(self.num_agents):
            if speeds[i] > max_speed:
                self.velocities[i] = self.velocities[i] / speeds[i] * max_speed
        
        # Update positions
        self.positions = self.positions + self.velocities * self.dt
        
        # Boundary conditions
        self.positions = np.clip(self.positions, 0, [self.arena_width, self.arena_height])
        
        # Check for agents reaching the exit
        for i in range(self.num_agents):
            if self.active_agents[i] and self._check_exit(self.positions[i]):
                self.active_agents[i] = False
                self.evacuation_times.append(time.time())  # Record evacuation time
                
        # Periodically update desired directions to handle dynamic changes
        if np.random.rand() < 0.05:  # 5% chance per step to update directions
            self._update_desired_directions()
            
        # Return the number of active agents
        return np.sum(self.active_agents)
    
    def run_simulation(self, visualize=True):
        """Run the full simulation"""
        if visualize:
            return self._run_with_visualization()
        else:
            return self._run_without_visualization()
            
    def _run_without_visualization(self):
        """Run simulation without visualization"""
        start_time = time.time()
        active_count_history = []
        
        for t in range(self.time_steps):
            active_count = self.step()
            active_count_history.append(active_count)
            
            # Stop if all agents have evacuated
            if active_count == 0 and self.has_exit:
                break
                
        elapsed = time.time() - start_time
        print(f"Simulation completed in {elapsed:.2f} seconds")
        print(f"Steps taken: {t+1} out of {self.time_steps}")
        
        return active_count_history
    
    def _run_with_visualization(self):
        """Run simulation with visualization"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Set up animation
        def init():
            ax.clear()
            ax.set_xlim(0, self.arena_width)
            ax.set_ylim(0, self.arena_height)
            return []
        
        def animate(frame):
            ax.clear()
            ax.set_xlim(0, self.arena_width)
            ax.set_ylim(0, self.arena_height)
            ax.set_title(f"Crowd Flow Simulation - Step {frame}")
            
            # Run one step of simulation
            self.step()
            
            # Plot agents
            active_positions = self.positions[self.active_agents]
            active_velocities = self.velocities[self.active_agents]
            
            # Draw agents as circles
            for i, pos in enumerate(active_positions):
                circle = Circle(pos, self.agent_radius, color='blue', alpha=0.7)
                ax.add_patch(circle)
                
                # Draw velocity vector
                if frame % 5 == 0:  # Only show every 5th frame to reduce clutter
                    vel = active_velocities[i]
                    ax.arrow(pos[0], pos[1], vel[0], vel[1], 
                            head_width=0.3, head_length=0.3, fc='red', ec='red', alpha=0.7)
            
            # Draw exits
            for exit_info in self.exits:
                exit_pos = np.array(exit_info["position"])
                exit_width = exit_info["width"]
                
                # Different visualization based on exit position
                if exit_pos[0] == 0:  # Exit on left wall
                    exit_rect = Rectangle((-0.2, exit_pos[1] - exit_width/2), 0.2, exit_width, 
                                       color='green', alpha=0.7)
                    ax.add_patch(exit_rect)
                    ax.text(2, exit_pos[1], 'EXIT', ha='left', va='center', color='green', fontsize=12)
                    
                elif exit_pos[0] == self.arena_width:  # Exit on right wall
                    exit_rect = Rectangle((self.arena_width, exit_pos[1] - exit_width/2), 0.2, exit_width, 
                                       color='green', alpha=0.7)
                    ax.add_patch(exit_rect)
                    ax.text(self.arena_width-2, exit_pos[1], 'EXIT', 
                           ha='right', va='center', color='green', fontsize=12)
                    
                elif exit_pos[1] == 0:  # Exit on bottom wall
                    exit_rect = Rectangle((exit_pos[0] - exit_width/2, -0.2), exit_width, 0.2, 
                                       color='green', alpha=0.7)
                    ax.add_patch(exit_rect)
                    ax.text(exit_pos[0], 2, 'EXIT', ha='center', va='bottom', color='green', fontsize=12)
                    
                elif exit_pos[1] == self.arena_height:  # Exit on top wall
                    exit_rect = Rectangle((exit_pos[0] - exit_width/2, self.arena_height), exit_width, 0.2, 
                                       color='green', alpha=0.7)
                    ax.add_patch(exit_rect)
                    ax.text(exit_pos[0], self.arena_height-2, 'EXIT', 
                           ha='center', va='top', color='green', fontsize=12)
            
            # Draw obstacles
            for obstacle in self.obstacles:
                pos = obstacle["position"]
                width = obstacle["width"]
                height = obstacle["height"]
                color = 'black' if obstacle.get("is_wall", False) else 'gray'
                rect = Rectangle((pos[0]-width/2, pos[1]-height/2), width, height, 
                               color=color, alpha=0.7)
                ax.add_patch(rect)
            
            # Display stats
            ax.text(1, self.arena_height-2, f'Active Agents: {np.sum(self.active_agents)}', 
                   fontsize=10, ha='left')
            
            return []
        
        # Create animation
        anim = animation.FuncAnimation(fig, animate, frames=self.time_steps,
                                      init_func=init, blit=True, interval=50)
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        return anim

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Crowd Flow Simulator')
    parser.add_argument('--num_agents', type=int, default=100, help='Number of agents')
    parser.add_argument('--arena_width', type=float, default=50, help='Width of arena')
    parser.add_argument('--arena_height', type=float, default=50, help='Height of arena')
    parser.add_argument('--time_steps', type=int, default=500, help='Maximum number of simulation steps')
    parser.add_argument('--dt', type=float, default=0.1, help='Time step size')
    parser.add_argument('--custom_layout', type=str, default=None, help='Path to JSON file with custom layout')
    parser.add_argument('--no_viz', action='store_false', dest='visualize', help='Disable visualization')
    return parser.parse_args()

def run_experiment(args=None):
    """Run a crowd flow simulation experiment"""
    if args is None:
        # Use default parameters
        args = parse_args()
    
    print(f"Starting simulation with {args.num_agents} agents")
    
    # Create and run simulation
    simulator = CrowdFlowSimulator(
        num_agents=args.num_agents,
        arena_width=args.arena_width,
        arena_height=args.arena_height,
        time_steps=args.time_steps,
        dt=args.dt,
        custom_layout=args.custom_layout
    )
    
    active_counts = simulator.run_simulation(visualize=args.visualize)
    
    # Plot evacuation curve if not visualizing
    if not args.visualize and simulator.has_exit:
        plt.figure(figsize=(10, 6))
        plt.plot(active_counts)
        plt.xlabel('Time Step')
        plt.ylabel('Number of Active Agents')
        plt.title('Evacuation Curve')
        plt.grid(True, alpha=0.3)
        plt.show()
    
    return simulator

if __name__ == "__main__":
    args = parse_args()
    simulator = run_experiment(args)