import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle, Circle
import argparse
from scipy.spatial import distance
import time

class CrowdFlowSimulator:
    def __init__(self, num_agents=100, arena_width=50, arena_height=50, 
                 time_steps=1000, dt=0.1, has_exit=True, has_obstacles=False):
        """Initialize the crowd flow simulator"""
        self.num_agents = num_agents
        self.arena_width = arena_width
        self.arena_height = arena_height
        self.time_steps = time_steps
        self.dt = dt
        self.has_exit = has_exit
        self.has_obstacles = has_obstacles
        
        # Social force parameters
        self.A = 2.0       # Repulsion strength between agents
        self.B = 1.0       # Repulsion range
        self.A_wall = 10.0  # Wall repulsion strength
        self.B_wall = 0.2   # Wall repulsion range
        self.tau = 0.5     # Relaxation time
        self.agent_radius = 0.3  # Physical radius of agents
        
        # Agent properties
        self.positions = np.random.rand(num_agents, 2) * [arena_width, arena_height]
        self.velocities = np.zeros((num_agents, 2))
        self.desired_speed = 1.0 + 0.3 * np.random.randn(num_agents)  # Desired speed varies by individual
        self.desired_directions = np.zeros((num_agents, 2))
        
        # Exit and obstacles
        if self.has_exit:
            self.exit_position = np.array([arena_width, arena_height/2])
            self.exit_width = 3.0
        
        if self.has_obstacles:
            self.obstacles = self._create_obstacles()
        else:
            self.obstacles = []
        
        # Statistics
        self.evacuation_times = []
        self.active_agents = np.ones(num_agents, dtype=bool)  # Track which agents are still in the arena
        
        # Set initial desired directions (toward exit if exists)
        self._update_desired_directions()
    
    def _create_obstacles(self):
        """Create obstacles in the arena"""
        obstacles = []
        
        # Add a central obstacle
        center_x, center_y = self.arena_width / 2, self.arena_height / 2
        width, height = 10, 2
        obstacles.append({"position": [center_x, center_y], "width": width, "height": height})
        
        # Add some random obstacles
        for _ in range(2):
            pos_x = self.arena_width * 0.2 + 0.6 * self.arena_width * np.random.rand()
            pos_y = self.arena_height * 0.2 + 0.6 * self.arena_height * np.random.rand()
            width = 2 + 5 * np.random.rand()
            height = 2 + 5 * np.random.rand()
            obstacles.append({"position": [pos_x, pos_y], "width": width, "height": height})
            
        return obstacles
    
    def _update_desired_directions(self):
        """Update desired directions for all agents"""
        for i in range(self.num_agents):
            if not self.active_agents[i]:
                continue
                
            if self.has_exit:
                # Direction toward exit
                diff = self.exit_position - self.positions[i]
                distance = np.linalg.norm(diff)
                if distance > 0:
                    self.desired_directions[i] = diff / distance
            else:
                # Random direction if no exit
                angle = 2 * np.pi * np.random.rand()
                self.desired_directions[i] = np.array([np.cos(angle), np.sin(angle)])
    
    def _compute_wall_force(self, position):
        """Compute force from walls"""
        force = np.zeros(2)
        
        # Distance to walls
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
                # Direction away from obstacle
                direction = position - obs_pos
                if np.linalg.norm(direction) > 0:
                    direction = direction / np.linalg.norm(direction)
                    
                # Apply force
                force += direction * self.A_wall * np.exp(-distance / self.B_wall)
                
        return force
    
    def _check_exit(self, position):
        """Check if agent has reached the exit"""
        if not self.has_exit:
            return False
            
        # Check if agent is within exit bounds
        x_in_exit = self.arena_width - position[0] < 0.5
        y_in_exit = abs(position[1] - self.exit_position[1]) < self.exit_width / 2
        
        return x_in_exit and y_in_exit
    
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
            
            # Wall forces
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
                        
                        # Add contact force if very close
                        if distance < 2 * self.agent_radius:
                            contact_force = max(0, 2*self.agent_radius - distance) * direction * 20
                            forces[i] -= contact_force
        
        # Update velocities and positions
        self.velocities = self.velocities + forces * self.dt
        
        # Limit velocities to a maximum speed
        speeds = np.linalg.norm(self.velocities, axis=1)
        max_speed = 3.0  # Maximum allowed speed
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
            
            # Draw exit if present
            if self.has_exit:
                exit_height = self.exit_width
                exit_y = self.exit_position[1] - exit_height/2
                exit_rect = Rectangle((self.arena_width-0.2, exit_y), 0.2, exit_height, 
                                    color='green', alpha=0.7)
                ax.add_patch(exit_rect)
                ax.text(self.arena_width-2, self.exit_position[1], 'EXIT', 
                        ha='right', va='center', color='green', fontsize=12)
            
            # Draw obstacles
            for obstacle in self.obstacles:
                pos = obstacle["position"]
                width = obstacle["width"]
                height = obstacle["height"]
                rect = Rectangle((pos[0]-width/2, pos[1]-height/2), width, height, 
                                color='gray', alpha=0.7)
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
    parser.add_argument('--no_exit', action='store_false', dest='has_exit', help='Disable exit')
    parser.add_argument('--obstacles', action='store_true', dest='has_obstacles', help='Add obstacles')
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
        has_exit=args.has_exit,
        has_obstacles=args.has_obstacles
    )
    
    active_counts = simulator.run_simulation(visualize=args.visualize)
    
    # Plot evacuation curve if not visualizing
    if not args.visualize and args.has_exit:
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