import sys
import matplotlib.pyplot as plt
from crowd_flow_simulator import CrowdFlowSimulator
from crowd_simulation_gui import CrowdSimulationGUI

if __name__ == "__main__":
    print("Starting Enhanced Crowd Flow Simulation GUI...")
    
    # Create simulator with default parameters
    simulator = CrowdFlowSimulator(
        num_agents=100,
        arena_width=50,
        arena_height=50,
        time_steps=1000,
        dt=0.1,
        has_exit=False,
        has_obstacles=True
    )
    
    # Create and run GUI
    gui = CrowdSimulationGUI(simulator)
    gui.run()
    
    print("Simulation closed!")