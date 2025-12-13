import math
import numpy as np
import time
from decimal import Decimal
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

from shapely.geometry import Polygon

from dg_commons.sim.models.diff_drive import DiffDriveModel, DiffDriveState, DiffDriveCommands
from dg_commons.sim.models.diff_drive_structures import DiffDriveGeometry, DiffDriveParameters
from dg_commons import apply_SE2_to_shapely_geo
import numpy as np

from .planner_viz import PlannerDebugger  # [NEW] Visualization


def SE2_from_xytheta(xytheta):
    """
    Constructs an SE2 homogeneous transformation matrix from [x, y, theta].
    """
    x, y, theta = xytheta
    c = math.cos(theta)
    s = math.sin(theta)
    return np.array([[c, -s, x], [s, c, y], [0, 0, 1]])


@dataclass
class PlanPoint:
    """Represents a single step in the final trajectory."""
    x: float
    y: float
    theta: float
    t: float
    v: float
    w: float
    target_idx: int = 0  # Metadata for backtracking


class ExactSpaceTimePlanner:
    def __init__(self, static_obstacles: List[Polygon], dt: float = 0.1, margin: float = 0.1): # Added margin default
        self.static_obstacles = static_obstacles
        self.dt = dt
        self.decimal_dt = Decimal(str(dt))
        self.margin = margin
        self.reservations: Dict[int, List[Polygon]] = {}
        self.debugger = PlannerDebugger() # [NEW] Debugger

    def plan_prioritized(
        self,
        robots_sequence: List[str],
        initial_states: Dict[str, DiffDriveState],
        waypoints_dict: Dict[str, List[Tuple[float, float]]],
        geometries: Dict[str, DiffDriveGeometry],
        params: Dict[str, DiffDriveParameters],
    ) -> Dict[str, List[PlanPoint]]:
        
        final_plans = {}
        self.reservations.clear()
        
        total_start = time.time()

        # Pre-calculate start footprints 
        start_footprints = {}
        for r_name in robots_sequence:
            s = initial_states[r_name]
            vg = geometries[r_name]
            footprint_poly = vg.outline_as_polygon.buffer(self.margin) # Consistent margin
            tform = SE2_from_xytheta([s.x, s.y, s.psi])
            start_footprints[r_name] = apply_SE2_to_shapely_geo(footprint_poly, tform)

        for i, robot_name in enumerate(robots_sequence):
            print(f"Planning physics execution for {robot_name}...")
            r_start = time.time()
            self.debugger.start_robot(robot_name) # [NEW] Start logging

            # Lower priority robots as obstacles (Optional, can be removed to reduce 'Statue' problem)
            lower_priority_robots = robots_sequence[i+1:]
            temp_static_obstacles = [start_footprints[r] for r in lower_priority_robots]

            start_state = initial_states[robot_name]
            model = DiffDriveModel(x0=start_state, vg=geometries[robot_name], vp=params[robot_name])
            targets = waypoints_dict.get(robot_name, [])

            # USE BACKTRACKING PLANNER
            trajectory = self._plan_single_robot_backtracking(model, targets, extra_static_obstacles=temp_static_obstacles)
            final_plans[robot_name] = trajectory
            
            print(f"  -> Planned {robot_name} in {time.time() - r_start:.2f}s")

            # Reserve Space-Time
            footprint = model.vg.outline_as_polygon.buffer(self.margin)
            
            last_time_idx = 0
            final_poly = None

            for pt in trajectory:
                time_idx = int(round(pt.t / self.dt))
                last_time_idx = time_idx
                tform = SE2_from_xytheta([pt.x, pt.y, pt.theta])
                current_poly = apply_SE2_to_shapely_geo(footprint, tform)
                final_poly = current_poly

                if time_idx not in self.reservations:
                    self.reservations[time_idx] = []
                self.reservations[time_idx].append(current_poly)
            
            # Parking Reservation
            if final_poly:
                PARKING_HORIZON_STEPS = 2000 
                for k in range(1, PARKING_HORIZON_STEPS):
                    future_idx = last_time_idx + k
                    if future_idx not in self.reservations:
                        self.reservations[future_idx] = []
                    self.reservations[future_idx].append(final_poly)

        print(f"Total Exact Planning Time: {time.time() - total_start:.2f}s")
        
        # [NEW] Generate Plots
        self.debugger.plot_summary(self.static_obstacles)
        
        return final_plans

    def _plan_single_robot_backtracking_refined(self, model: DiffDriveModel, targets: List[Tuple[float, float]], extra_static_obstacles: List[Polygon] = []) -> List[PlanPoint]:
        # Refined version to handle the start state correctly
        trajectory: List[PlanPoint] = []
        current_time = 0.0
        
        # Keep copy of initial state for full reset
        initial_state_copy = DiffDriveState(x=model._state.x, y=model._state.y, psi=model._state.psi)

        # Physics Constants
        r = model.vg.wheelradius
        L = model.vg.wheelbase
        w_motor_max = model.vp.omega_limits[1]
        v_max = r * w_motor_max 
        w_max = (2 * r * w_motor_max) / L

        def get_cmds_inverse(v_des, w_des):
            omega_r = (v_des / r) + (w_des * L / (2 * r))
            omega_l = (v_des / r) - (w_des * L / (2 * r))
            return DiffDriveCommands(omega_l=omega_l, omega_r=omega_r)

        def is_safe(next_poly, time_idx):
            for obs in self.static_obstacles:
                if obs.intersects(next_poly): return False
            for obs in extra_static_obstacles:
                if obs.intersects(next_poly): return False
            if time_idx in self.reservations:
                for reserved_poly in self.reservations[time_idx]:
                    if reserved_poly.intersects(next_poly): return False
            return True

        target_idx = 0
        forced_wait = False

        # Max iterations to prevent infinite loops (if stuck)
        iter_count = 0
        MAX_ITERS = 50000

        while target_idx < len(targets):
            iter_count += 1
            self.debugger.record_iteration(iter_count, current_time) # [NEW] Log Progress

            if iter_count > MAX_ITERS:
                print(f"WARNING: Max iterations ({MAX_ITERS}) reached for {model}. Stopping.")
                break

            tx, ty = targets[target_idx]
            
            # --- Calc Distance & Command ---
            dx = tx - model._state.x
            dy = ty - model._state.y
            dist = math.hypot(dx, dy)

            # Check if Target Reached
            # Only advance target if we are NOT in a forced wait state
            if dist < 1e-3 and not forced_wait:
                target_idx += 1
                continue

            cmd_v = 0.0
            cmd_w = 0.0
            
            if not forced_wait:
                # Normal Greedy Move Logic
                heading_fwd = math.atan2(dy, dx)
                heading_rev = math.atan2(-dy, -dx)
                err_fwd = (heading_fwd - model._state.psi + math.pi) % (2 * math.pi) - math.pi
                err_rev = (heading_rev - model._state.psi + math.pi) % (2 * math.pi) - math.pi
                
                move_dir = 1.0
                target_psi = heading_fwd
                if abs(err_rev) < abs(err_fwd):
                    move_dir = -1.0
                    target_psi = heading_rev
                
                d_psi = (target_psi - model._state.psi + math.pi) % (2 * math.pi) - math.pi
                
                # Turn vs Move logic
                # [Optimization] If distance is huge, allow moving while turning slightly (Bank turn)?
                # No, stick to Turn-Then-Move for safety in tight spaces.
                if abs(d_psi) > 1e-4: 
                    max_step_rot = w_max * self.dt
                    if abs(d_psi) <= max_step_rot: cmd_w = d_psi / self.dt
                    else: cmd_w = np.sign(d_psi) * w_max
                else:
                    max_step_dist = v_max * self.dt
                    if dist <= max_step_dist: cmd_v_mag = dist / self.dt
                    else: cmd_v_mag = v_max
                    cmd_v = cmd_v_mag * move_dir
            else:
                # Forced Wait (We backtracked here because moving failed)
                forced_wait = False # Reset flag for NEXT iteration (unless we fail waiting too)

            # --- Predict & Check ---
            cmd = get_cmds_inverse(cmd_v, cmd_w)
            
            # Predict Next State
            next_model = DiffDriveModel(x0=model._state, vg=model.vg, vp=model.vp)
            next_model.update(cmd, self.decimal_dt)
            next_poly = next_model.get_footprint()
            next_time_idx = int(round((current_time + self.dt) / self.dt))
            
            if is_safe(next_poly, next_time_idx):
                # SAFE -> Commit
                model.update(cmd, self.decimal_dt)
                current_time += self.dt
                s = model._state
                trajectory.append(PlanPoint(s.x, s.y, s.psi, current_time, cmd_v, cmd_w, target_idx))
            else:
                # UNSAFE
                self.debugger.record_collision(next_poly.centroid.x, next_poly.centroid.y) # [NEW] Log Collision

                # 1. Check if we were already waiting.
                # If cmd was (0,0), it means we tried to wait and failed.
                already_waiting = (abs(cmd_v) < 1e-6 and abs(cmd_w) < 1e-6)
                
                wait_success = False
                
                # 2. If we were moving, TRY WAITING first (Greedy fallback)
                if not already_waiting:
                    wait_model = DiffDriveModel(x0=model._state, vg=model.vg, vp=model.vp)
                    wait_model.update(DiffDriveCommands(0,0), self.decimal_dt)
                    wait_poly = wait_model.get_footprint()
                    
                    if is_safe(wait_poly, next_time_idx):
                        # Wait Successful
                        model.update(DiffDriveCommands(0,0), self.decimal_dt)
                        current_time += self.dt
                        s = model._state
                        trajectory.append(PlanPoint(s.x, s.y, s.psi, current_time, 0.0, 0.0, target_idx))
                        wait_success = True
                
                if wait_success:
                    continue
                
                # 2. Backtrack (Wait Failed or was already waiting)
                while True:
                    if len(trajectory) == 0:
                        # Trapped at start. Force wait.
                        model.update(DiffDriveCommands(0,0), self.decimal_dt)
                        current_time += self.dt
                        s = model._state
                        trajectory.append(PlanPoint(s.x, s.y, s.psi, current_time, 0.0, 0.0, target_idx))
                        break # Break from backtrack loop to continue main loop

                    # POP last state
                    prev_pt = trajectory.pop()
                    
                    # Check if the popped state was a WAIT action
                    was_wait = (abs(prev_pt.v) < 1e-6 and abs(prev_pt.w) < 1e-6)
                    
                    if not was_wait:
                        # We popped a MOVE. We can now try to REPLACE this Move with a Wait.
                        # Restore Model State to the parent of the popped state
                        if len(trajectory) > 0:
                            restore_pt = trajectory[-1]
                            model._state = DiffDriveState(x=restore_pt.x, y=restore_pt.y, psi=restore_pt.theta)
                            current_time = restore_pt.t
                            target_idx = restore_pt.target_idx
                        else:
                            # Reset to absolute start
                            model._state = DiffDriveState(x=initial_state_copy.x, y=initial_state_copy.y, psi=initial_state_copy.psi)
                            current_time = 0.0
                            target_idx = 0 
                        
                        self.debugger.record_backtrack(iter_count, prev_pt.t, current_time) # [NEW] Log Backtrack
                        forced_wait = True
                        break # Done backtracking
                    
                    # If it WAS a wait, we loop again to pop the NEXT parent (Recursion)
                    # Because waiting at that step didn't work, so we need to undo the move that got us there.
                
                if len(trajectory) > 0 or forced_wait:
                     continue # Continue main loop with restored state

        return trajectory

    def _plan_single_robot_backtracking(self, model, targets, extra_static_obstacles):
        return self._plan_single_robot_backtracking_refined(model, targets, extra_static_obstacles)