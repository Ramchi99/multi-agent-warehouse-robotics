import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from shapely.geometry import Polygon

class PlannerDebugger:
    def __init__(self, output_dir="out/ex14/debug_plots"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Structure: {robot_name: {'iterations': [], 'times': [], 'collisions': [], 'backtracks': []}}
        self.logs = {}
        self.current_robot = None

    def start_robot(self, robot_name):
        self.current_robot = robot_name
        self.logs[robot_name] = {
            'iterations': [],      # Iteration count
            'sim_times': [],       # Simulation time achieved (depth of search)
            'backtracks': [],      # Iteration where backtrack happened
            'collisions': [],      # (x, y) coordinates of collision
            'waits': []            # Iterations where we waited
        }

    def record_iteration(self, iter_idx, sim_time):
        if not self.current_robot: return
        self.logs[self.current_robot]['iterations'].append(iter_idx)
        self.logs[self.current_robot]['sim_times'].append(sim_time)

    def record_collision(self, x, y):
        if not self.current_robot: return
        self.logs[self.current_robot]['collisions'].append((x, y))

    def record_backtrack(self, iter_idx, sim_time_before, sim_time_after):
        if not self.current_robot: return
        # Store as (iteration, time_dropped_to)
        self.logs[self.current_robot]['backtracks'].append((iter_idx, sim_time_after))

    def plot_summary(self, static_obstacles: List[Polygon]):
        """Generates comprehensive plots."""
        print(f"Generating Debug Plots in {self.output_dir}...")
        
        # 1. Search Progress (The "Sawtooth" Graph)
        self._plot_search_progress()
        
        # 2. Conflict Heatmap
        self._plot_conflict_heatmap(static_obstacles)

    def _plot_search_progress(self):
        """Plots Simulation Time vs Iterations. 
        Rising slope = Progress. Drops = Backtracking."""
        plt.figure(figsize=(12, 6))
        
        for r_name, data in self.logs.items():
            if not data['iterations']: continue
            
            iters = data['iterations']
            times = data['sim_times']
            
            # Plot main progress line
            plt.plot(iters, times, label=f"{r_name} Progress", linewidth=1.5, alpha=0.8)
            
            # Scatter backtracks
            bt_iters = [x[0] for x in data['backtracks']]
            bt_times = [x[1] for x in data['backtracks']]
            if bt_iters:
                plt.scatter(bt_iters, bt_times, marker='x', color='red', s=20, zorder=5, label=f"{r_name} Backtrack")

        plt.xlabel("Planner Iteration (Compute Step)")
        plt.ylabel("Simulation Time Reached (s)")
        plt.title("Backtracking Search Progress\n(Up=Moving, Vertical Drop=Backtracking)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / "debug_search_progress.png")
        plt.close()

    def _plot_conflict_heatmap(self, static_obstacles):
        """Plots where collisions occurred most frequently."""
        plt.figure(figsize=(10, 10))
        
        # Plot Static Obstacles
        for poly in static_obstacles:
            if hasattr(poly, 'geoms'): geoms = poly.geoms
            else: geoms = [poly]
            for g in geoms:
                try:
                    if g.geom_type == 'Polygon':
                        x, y = g.exterior.xy
                        plt.fill(x, y, color='gray', alpha=0.5)
                    elif g.geom_type in ['LinearRing', 'LineString']:
                        x, y = g.xy
                        plt.plot(x, y, color='gray', linewidth=2, alpha=0.5)
                except AttributeError:
                    pass # Skip unsupported geometries

        # Plot Collisions per Robot
        colors = plt.cm.get_cmap('tab10', len(self.logs))
        
        for i, (r_name, data) in enumerate(self.logs.items()):
            col_points = data['collisions']
            if not col_points: continue
            
            cx, cy = zip(*col_points)
            
            # Add jitter to see overlaps
            cx = np.array(cx) + np.random.normal(0, 0.05, len(cx))
            cy = np.array(cy) + np.random.normal(0, 0.05, len(cy))
            
            plt.scatter(cx, cy, s=10, alpha=0.6, color=colors(i), label=f"{r_name} Conflicts")

        plt.title("Conflict Heatmap (Where is_safe() returned False)")
        plt.legend()
        plt.axis('equal')
        plt.grid(True)
        plt.savefig(self.output_dir / "debug_conflict_heatmap.png")
        plt.close()
