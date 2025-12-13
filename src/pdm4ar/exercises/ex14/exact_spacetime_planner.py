import math
import numpy as np
from decimal import Decimal
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

from shapely.geometry import Polygon

# Import the provided simulation structures
from dg_commons.sim.models.diff_drive import DiffDriveModel, DiffDriveState, DiffDriveCommands
from dg_commons.sim.models.diff_drive_structures import DiffDriveGeometry, DiffDriveParameters
from dg_commons import apply_SE2_to_shapely_geo
import numpy as np


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


class ExactSpaceTimePlanner:
    def __init__(self, static_obstacles: List[Polygon], dt: float = 0.1):
        self.static_obstacles = static_obstacles
        self.dt = dt
        self.decimal_dt = Decimal(str(dt))

        # Space-Time Reservation Table
        # Maps time_step_index -> List of Polygons (occupied by higher priority robots)
        self.reservations: Dict[int, List[Polygon]] = {}

    def plan_prioritized(
        self,
        robots_sequence: List[str],
        initial_states: Dict[str, DiffDriveState],
        waypoints_dict: Dict[str, List[Tuple[float, float]]],
        geometries: Dict[str, DiffDriveGeometry],
        params: Dict[str, DiffDriveParameters],
    ) -> Dict[str, List[PlanPoint]]:
        """
        Plans for robots one by one in the given order.
        """
        final_plans = {}
        self.reservations.clear()

        # [NEW] Pre-calculate start footprints for ALL robots
        start_footprints = {}
        for r_name in robots_sequence:
            s = initial_states[r_name]
            vg = geometries[r_name]
            # [MODIFIED] Add safety margin
            footprint_poly = vg.outline_as_polygon.buffer(0.1)
            tform = SE2_from_xytheta([s.x, s.y, s.psi])
            start_footprints[r_name] = apply_SE2_to_shapely_geo(footprint_poly, tform)

        for i, robot_name in enumerate(robots_sequence):
            print(f"Planning physics execution for {robot_name}...")

            # [NEW] Identify lower-priority robots (those coming AFTER me in the sequence)
            # We treat their START positions as static obstacles to avoid "bulldozing" them.
            lower_priority_robots = robots_sequence[i+1:]
            
            temp_static_obstacles = []
            for other_r in lower_priority_robots:
                temp_static_obstacles.append(start_footprints[other_r])

            # 1. Initialize the simulation model for this robot
            start_state = initial_states[robot_name]
            model = DiffDriveModel(x0=start_state, vg=geometries[robot_name], vp=params[robot_name])

            # 2. Get the geometric path (waypoints)
            targets = waypoints_dict.get(robot_name, [])

            # 3. Run the "Turn-Move" simulation
            # [MODIFIED] Pass the temporary static obstacles
            trajectory = self._plan_single_robot(model, targets, extra_static_obstacles=temp_static_obstacles)
            final_plans[robot_name] = trajectory

            # 4. Reserve the space-time for this robot so subsequent robots avoid it
            # [MODIFIED] Add safety margin to the reservation footprint
            footprint = model.vg.outline_as_polygon.buffer(0.1)

            last_time_idx = 0
            final_poly = None

            for pt in trajectory:
                time_idx = int(round(pt.t / self.dt))
                last_time_idx = time_idx

                # Create the polygon at this specific time step
                tform = SE2_from_xytheta([pt.x, pt.y, pt.theta])
                current_poly = apply_SE2_to_shapely_geo(footprint, tform)
                final_poly = current_poly

                if time_idx not in self.reservations:
                    self.reservations[time_idx] = []
                self.reservations[time_idx].append(current_poly)
            
            # [NEW] Reserve the "Parked" state for the future
            # This prevents subsequent robots from colliding with this robot after it finishes.
            if final_poly:
                PARKING_HORIZON_STEPS = 1000  # Reserve for 100 seconds after finish
                for k in range(1, PARKING_HORIZON_STEPS):
                    future_idx = last_time_idx + k
                    if future_idx not in self.reservations:
                        self.reservations[future_idx] = []
                    self.reservations[future_idx].append(final_poly)

        return final_plans

    def _plan_single_robot(self, model: DiffDriveModel, targets: List[Tuple[float, float]], extra_static_obstacles: List[Polygon] = []) -> List[PlanPoint]:

        trajectory: List[PlanPoint] = []
        current_time = 0.0

        # Physics Constants
        r = model.vg.wheelradius
        L = model.vg.wheelbase
        w_motor_max = model.vp.omega_limits[1]

        # Calculate Max Velocity Limits
        v_max = r * w_motor_max  # Max linear speed [m/s]
        w_max = (2 * r * w_motor_max) / L  # Max rotational speed [rad/s]

        # Helper: Calculate Left/Right wheel speeds from V and W
        def get_cmds_inverse(v_des, w_des):
            omega_r = (v_des / r) + (w_des * L / (2 * r))
            omega_l = (v_des / r) - (w_des * L / (2 * r))
            return DiffDriveCommands(omega_l=omega_l, omega_r=omega_r)

        # Helper: Check if a future polygon collides
        def is_safe(next_poly, time_idx):
            # 1. Check PERMANENT Static Obstacles (Walls)
            for obs in self.static_obstacles:
                if obs.intersects(next_poly):
                    return False
            
            # 2. Check TEMPORARY Static Obstacles (Unplanned Robots)
            for obs in extra_static_obstacles:
                if obs.intersects(next_poly):
                    return False

            # 3. Check Higher Priority Robots (Space-Time)
            if time_idx in self.reservations:
                for reserved_poly in self.reservations[time_idx]:
                    if reserved_poly.intersects(next_poly):
                        return False
            return True
        
        # [NEW] Lookahead Safety Check
        def is_safe_horizon(next_poly, start_time_idx, steps=50):
            # Check future time steps for DYNAMIC obstacles (reservations) only.
            # Static obstacles are already checked in is_safe().
            for k in range(1, steps + 1):
                check_time = start_time_idx + k
                if check_time in self.reservations:
                    for reserved_poly in self.reservations[check_time]:
                        if reserved_poly.intersects(next_poly):
                            return False
            return True

        # --- Main Loop: Process every waypoint ---
        for tx, ty in targets:

            # [NEW] Determine Direction (Forward vs Backward)
            dx = tx - model._state.x
            dy = ty - model._state.y
            
            # Calculate heading for Forward motion
            heading_fwd = math.atan2(dy, dx)
            err_fwd = (heading_fwd - model._state.psi + math.pi) % (2 * math.pi) - math.pi
            
            # Calculate heading for Backward motion (facing opposite to target)
            heading_rev = math.atan2(-dy, -dx)
            err_rev = (heading_rev - model._state.psi + math.pi) % (2 * math.pi) - math.pi
            
            move_dir = 1.0
            target_psi = heading_fwd
            
            # If turning to reverse is shorter (e.g. error is < 90 deg when facing back, but > 90 deg when facing fwd)
            if abs(err_rev) < abs(err_fwd):
                move_dir = -1.0
                target_psi = heading_rev

            # === PHASE 1: ALIGN (Spot Turn) ===
            while True:
                # Re-calculate error to the CHOSEN target heading
                d_psi = (target_psi - model._state.psi + math.pi) % (2 * math.pi) - math.pi

                # Tolerance check (Stop turning if close enough)
                if abs(d_psi) < 1e-4:
                    break

                # Calculate the max rotation we can do in one timestep
                max_step_rot = w_max * self.dt

                # If we are close, use exact speed to finish. Else use max speed.
                if abs(d_psi) <= max_step_rot:
                    cmd_w = d_psi / self.dt
                else:
                    cmd_w = np.sign(d_psi) * w_max

                # Generate commands for this step
                cmd = get_cmds_inverse(v_des=0.0, w_des=cmd_w)

                # PREDICT COLLISION
                # We perform a "dry run" update on a copy of the model
                next_model = DiffDriveModel(x0=model._state, vg=model.vg, vp=model.vp)
                next_model.update(cmd, self.decimal_dt)
                next_poly = next_model.get_footprint()
                next_time_idx = int(round((current_time + self.dt) / self.dt))

                # [MODIFIED] Check Immediate Safety AND Future Safety (Lookahead)
                if is_safe(next_poly, next_time_idx) and is_safe_horizon(next_poly, next_time_idx, steps=50):
                    # EXECUTE: Apply to real model
                    model.update(cmd, self.decimal_dt)
                    current_time += self.dt
                    s = model._state
                    # Store the command used to get HERE
                    trajectory.append(PlanPoint(s.x, s.y, s.psi, current_time, 0.0, cmd_w))
                else:
                    # WAIT: Collision detected (immediate or future), stay still
                    # Check if waiting is safe (Immediate check is enough to detect trapping)
                    wait_poly = model.get_footprint()
                    if is_safe(wait_poly, next_time_idx):
                         model.update(DiffDriveCommands(0, 0), self.decimal_dt)
                         current_time += self.dt
                         s = model._state
                         trajectory.append(PlanPoint(s.x, s.y, s.psi, current_time, 0.0, 0.0))
                    else:
                         # We are trapped. Print warning but keep waiting (best effort).
                         # print(f"WARNING: Robot trapped at time {next_time_idx}!")
                         model.update(DiffDriveCommands(0, 0), self.decimal_dt)
                         current_time += self.dt
                         s = model._state
                         trajectory.append(PlanPoint(s.x, s.y, s.psi, current_time, 0.0, 0.0))

            # === PHASE 2: MOVE (Straight Line) ===
            while True:
                dx = tx - model._state.x
                dy = ty - model._state.y
                dist = math.hypot(dx, dy)

                # Tolerance check (Stop moving if close enough)
                if dist < 1e-3:
                    break

                # Calculate max distance we can do in one timestep
                max_step_dist = v_max * self.dt

                # If close, use exact speed. Else max speed.
                if dist <= max_step_dist:
                    cmd_v_mag = dist / self.dt
                else:
                    cmd_v_mag = v_max
                
                # [NEW] Apply direction
                cmd_v = cmd_v_mag * move_dir

                cmd = get_cmds_inverse(v_des=cmd_v, w_des=0.0)

                # PREDICT COLLISION
                next_model = DiffDriveModel(x0=model._state, vg=model.vg, vp=model.vp)
                next_model.update(cmd, self.decimal_dt)
                next_poly = next_model.get_footprint()
                next_time_idx = int(round((current_time + self.dt) / self.dt))

                # [MODIFIED] Check Immediate Safety AND Future Safety (Lookahead)
                if is_safe(next_poly, next_time_idx) and is_safe_horizon(next_poly, next_time_idx, steps=50):
                    # EXECUTE
                    model.update(cmd, self.decimal_dt)
                    current_time += self.dt
                    s = model._state
                    trajectory.append(PlanPoint(s.x, s.y, s.psi, current_time, cmd_v, 0.0))
                else:
                    # WAIT
                    wait_poly = model.get_footprint()
                    if is_safe(wait_poly, next_time_idx):
                         model.update(DiffDriveCommands(0, 0), self.decimal_dt)
                         current_time += self.dt
                         s = model._state
                         trajectory.append(PlanPoint(s.x, s.y, s.psi, current_time, 0.0, 0.0))
                    else:
                         # print(f"WARNING: Robot trapped at time {next_time_idx}!")
                         model.update(DiffDriveCommands(0, 0), self.decimal_dt)
                         current_time += self.dt
                         s = model._state
                         trajectory.append(PlanPoint(s.x, s.y, s.psi, current_time, 0.0, 0.0))

        return trajectory
