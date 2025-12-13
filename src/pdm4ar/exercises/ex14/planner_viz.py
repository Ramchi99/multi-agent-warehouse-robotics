import matplotlib.pyplot as plt
import numpy as np
import json
import datetime # [NEW]
from pathlib import Path
from typing import List, Dict, Tuple
from shapely.geometry import Polygon
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
import matplotlib.cm as cm

class PlannerDebugger:
    def __init__(self, output_dir="out/ex14/debug_plots"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") # [NEW]
        
        # Structure: {robot_name: {'iterations': [], 'times': [], 'collisions': [], 'backtracks': [], 'targets': []}}
        self.logs = {}
        self.current_robot = None

    def start_robot(self, robot_name):
        self.current_robot = robot_name
        self.logs[robot_name] = {
            'iterations': [],      
            'sim_times': [],       
            'backtracks': [],      # (iter, t_before, t_after, x, y)
            'collisions': [],      # (x, y)
            'targets': [],          # Target Index at this iteration
            'stagnations': []      # (iter, time)
        }

    def record_iteration(self, iter_idx, sim_time, target_idx=0):
        if not self.current_robot: return
        self.logs[self.current_robot]['iterations'].append(int(iter_idx))
        self.logs[self.current_robot]['sim_times'].append(float(sim_time))
        self.logs[self.current_robot]['targets'].append(int(target_idx))
        
    def record_stagnation(self, iter_idx, sim_time):
        if not self.current_robot: return
        self.logs[self.current_robot]['stagnations'].append((int(iter_idx), float(sim_time)))

    def record_collision(self, x, y):
        if not self.current_robot: return
        self.logs[self.current_robot]['collisions'].append((float(x), float(y)))

    def record_backtrack(self, iter_idx, sim_time_before, sim_time_after):
        if not self.current_robot: return
        # Note: We don't have x,y here easily, but we can infer it later if needed or add it to args
        # For now, we just track time jumps.
        self.logs[self.current_robot]['backtracks'].append((int(iter_idx), float(sim_time_before), float(sim_time_after)))

    def export_logs_to_json(self):
        """Exports raw logs to JSON for analysis."""
        filepath = self.output_dir / f"planner_logs_{self.timestamp}.json"
        try:
            with open(filepath, "w") as f:
                json.dump(self.logs, f, indent=2)
            print(f"Exported planner logs to {filepath}")
        except Exception as e:
            print(f"Failed to export JSON logs: {e}")

    def plot_summary(self, static_obstacles: List[Polygon]):
        """Generates comprehensive plots."""
        print(f"Generating Debug Plots in {self.output_dir}...")
        
        self.export_logs_to_json() 
        
        # 1. Search Progress (Time & Targets)
        self._plot_search_progress()
        
        # 2. Conflict Heatmap
        self._plot_conflict_heatmap(static_obstacles)
        
        # 3. Wait Time Statistics
        self._print_wait_statistics()

    def _print_wait_statistics(self):
        print("\n--- Wait Time Statistics ---")
        for r_name, data in self.logs.items():
            times = data['sim_times']
            if not times: continue
            
            # Rough approximation: Count how many times sim_time didn't advance
            # This is tricky with backtracks, so we look at the FINAL trajectory duration
            # vs the number of steps if we moved full speed.
            
            # Since we don't have the final plan here easily (it's returned by the planner),
            # we can infer "Struggle" from the logs.
            
            total_iters = len(times)
            final_time = times[-1] if times else 0.0
            
            print(f"Robot {r_name}:")
            print(f"  Final Time Reached: {final_time:.2f}s")
            print(f"  Total Iterations:   {total_iters}")
            print(f"  Backtracks:         {len(data['backtracks'])}")
            print(f"  Stagnation Resets:  {len(data['stagnations'])}")

    def _plot_search_progress(self):
        """Plots Simulation Time vs Iterations AND Target Index vs Iterations."""
        
        for r_name, data in self.logs.items():
            if not data['iterations']: continue
            
            fig, ax1 = plt.subplots(figsize=(14, 8))
            
            iters = data['iterations']
            times = data['sim_times']
            targets = data['targets']
            stagnations = data.get('stagnations', [])
            backtracks = data['backtracks']

            # --- Primary Axis: Simulation Time ---
            color = 'tab:blue'
            ax1.set_xlabel('Planner Iterations')
            ax1.set_ylabel('Simulation Time (s)', color=color)
            ax1.plot(iters, times, color=color, linewidth=1.0, alpha=0.8, label="Sim Time")
            ax1.tick_params(axis='y', labelcolor=color)
            
            # Highlight Stagnation Zones
            if stagnations:
                stag_iters, stag_times = zip(*stagnations)
                # Shade the area around stagnation? Or just a line.
                for i, t in stagnations:
                    ax1.axvline(x=i, color='red', linestyle='--', alpha=0.5)
                    ax1.text(i, t, " STAGNATION", rotation=90, verticalalignment='bottom', color='red', fontsize=8)

            # Highlight Backtracks (Color coded segments)
            if backtracks:
                segments = []
                magnitudes = []
                for (i, t_before, t_after) in backtracks:
                    seg = [(i, t_before), (i, t_after)]
                    mag = abs(t_before - t_after)
                    segments.append(seg)
                    magnitudes.append(mag)
                
                if segments:
                    norm = Normalize(vmin=0, vmax=2.0)
                    cmap = cm.get_cmap('autumn_r')
                    lc = LineCollection(segments, cmap=cmap, norm=norm, linewidths=1.5, alpha=0.9, zorder=5)
                    lc.set_array(np.array(magnitudes))
                    ax1.add_collection(lc)
                    # cbar = fig.colorbar(lc, ax=ax1, label='Backtrack Size (s)', pad=0.01)

            # --- Secondary Axis: Target Index ---
            ax2 = ax1.twinx()  
            color = 'tab:green'
            ax2.set_ylabel('Target Waypoint Index', color=color)  
            ax2.plot(iters, targets, color=color, linewidth=1.5, linestyle='-', alpha=0.6, label="Target Idx")
            ax2.tick_params(axis='y', labelcolor=color)

            # Final Time Box
            final_t = times[-1] if times else 0
            plt.title(f"Search Progress: {r_name} (Final T = {final_t:.2f}s)")
            
            fig.tight_layout()  
            plt.grid(True, alpha=0.2)
            plt.savefig(self.output_dir / f"progress_{r_name}_{self.timestamp}.png", dpi=150)
            plt.close()

    def _plot_conflict_heatmap(self, static_obstacles):
        """Plots where collisions occurred most frequently using Density."""
        if not self.logs: return

        # Gather ALL collisions
        all_collisions = []
        for data in self.logs.values():
            all_collisions.extend(data['collisions'])
        
        if not all_collisions:
            print("No collisions recorded to plot.")
            return

        cx_all, cy_all = zip(*all_collisions)

        plt.figure(figsize=(12, 12))
        
        # 1. Plot Density (Hexbin) - Increased gridsize for resolution, but bigger bins
        # gridsize=30 means larger hexagons than 50
        hb = plt.hexbin(cx_all, cy_all, gridsize=20, cmap='inferno_r', mincnt=1, alpha=0.7, zorder=1)
        cb = plt.colorbar(hb, label='Collision Count')

        # 2. Plot Static Obstacles (Overlaid)
        for poly in static_obstacles:
            if hasattr(poly, 'geoms'): geoms = poly.geoms
            else: geoms = [poly]
            for g in geoms:
                try:
                    if g.geom_type == 'Polygon':
                        x, y = g.exterior.xy
                        plt.fill(x, y, color='black', alpha=0.4, zorder=2) # Darker
                        plt.plot(x, y, color='white', linewidth=1, zorder=2) # White outline for contrast
                    elif g.geom_type in ['LinearRing', 'LineString']:
                        x, y = g.xy
                        plt.plot(x, y, color='black', linewidth=3, alpha=0.6, zorder=2)
                except AttributeError:
                    pass

        plt.title("Collision Density Heatmap (All Robots)")
        plt.axis('equal')
        plt.grid(True, linestyle=':', alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / f"debug_conflict_heatmap_{self.timestamp}.png", dpi=150)
        plt.close()
