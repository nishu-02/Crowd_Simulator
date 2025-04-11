import sys
import matplotlib.pyplot as plt
from crowd_flow_simulator import CrowdFlowSimulator
from crowd_simulation_gui import CrowdSimulationGUI
from crowd_flow_analyzer import CrowdFlowAnalyzer
import argparse

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Enhanced Crowd Flow Simulation with Analysis')
    parser.add_argument('--num_agents', type=int, default=200, help='Number of agents')
    parser.add_argument('--arena_width', type=float, default=50, help='Width of arena')
    parser.add_argument('--arena_height', type=float, default=50, help='Height of arena')
    parser.add_argument('--time_steps', type=int, default=1000, help='Maximum simulation steps')
    parser.add_argument('--custom_layout', type=str, help='Path to custom layout JSON file')
    parser.add_argument('--create_layout', action='store_true', help='Create a new custom layout')
    parser.add_argument('--analyze_only', action='store_true', help='Run analysis without GUI')
    parser.add_argument('--output', type=str, default='simulation_analysis.json', help='Output file for analysis')
    # Add new arguments for animation control
    parser.add_argument('--smooth', action='store_true', help='Enable motion smoothing')
    parser.add_argument('--fps', type=int, default=30, help='Animation frames per second')
    return parser.parse_args()

def main():
    print("Starting Enhanced Crowd Flow Simulation with Analysis...")
    args = parse_arguments()

    # Create layout if requested
    if args.create_layout:
        temp_analyzer = CrowdFlowAnalyzer(None)
        layout_file = input("Enter output filename for layout (default: custom_layout.json): ") or "custom_layout.json"
        temp_analyzer.create_custom_layout(layout_file)
        args.custom_layout = layout_file

    # Calculate time step based on FPS if smooth animation is enabled
    dt = 1.0/args.fps if args.smooth else 0.1

    # Create simulator with smooth animation settings
    simulator = CrowdFlowSimulator(
        num_agents=args.num_agents,
        arena_width=args.arena_width,
        arena_height=args.arena_height,
        time_steps=args.time_steps,
        dt=dt,  # Use FPS-based time step
        custom_layout={
            "exits": [
                {
                    "position": [args.arena_width, args.arena_height/2],
                    "width": 3.0
                }
            ],
            "obstacles": [
                # Add some default obstacles
                {
                    "position": [args.arena_width/3, args.arena_height/2],
                    "width": 2.0,
                    "height": 10.0
                },
                {
                    "position": [2*args.arena_width/3, args.arena_height/2],
                    "width": 2.0,
                    "height": 10.0
                },
                # Add some columns
                {
                    "position": [args.arena_width/4, args.arena_height/4],
                    "width": 1.0,
                    "height": 1.0
                },
                {
                    "position": [3*args.arena_width/4, 3*args.arena_height/4],
                    "width": 1.0,
                    "height": 1.0
                }
            ],
            "walls": [
                # Add internal walls
                {
                    "position": [args.arena_width/2, args.arena_height*0.8],
                    "width": args.arena_width*0.3,
                    "height": 1.0,
                    "is_wall": True
                }
            ]
        }
    )

    # Create analyzer
    analyzer = CrowdFlowAnalyzer(simulator)

    if args.analyze_only:
        # Run simulation with analysis only
        print("Running simulation with analysis...")
        results = analyzer.run_analysis_with_simulator(args.time_steps, visualize=True)
        
        # Generate visualizations
        analyzer.visualize_density()
        analyzer.visualize_velocity_field()
        analyzer.visualize_evacuation_timeline()
        analyzer.visualize_voronoi_diagram()
        
        # Export results
        analyzer.export_analysis(args.output)
        print(f"Analysis results exported to {args.output}")
    else:
        # Create and run GUI with analyzer integration
        gui = CrowdSimulationGUI(simulator, analyzer)
        if args.smooth:
            # Update animation settings for smooth motion
            gui.anim.event_source.interval = 1000 / args.fps  # Convert FPS to milliseconds
        gui.run()

    print("Simulation complete!")

if __name__ == "__main__":
    main()