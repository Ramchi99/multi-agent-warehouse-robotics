import random
import math
import time
import copy
from dataclasses import dataclass
from typing import Mapping, Sequence, Tuple, List, Optional, Dict, Any

# --- TASK ALLOCATION LOGIC ---

@dataclass
class DeliveryTask:
    goal_id: str
    collection_id: str

    def clone(self):
        """Fast shallow copy"""
        return DeliveryTask(self.goal_id, self.collection_id)

@dataclass
class RobotSchedule:
    robot_name: str
    tasks: List[DeliveryTask]

    def clone(self):
        """
        Fast deep copy. 
        Much faster than copy.deepcopy() because we know the structure.
        """
        new_sched = RobotSchedule(self.robot_name, [])
        # List comprehension is faster than loops
        new_sched.tasks = [t.clone() for t in self.tasks]
        return new_sched

class TaskAllocatorBase:
    """Shared Data and Evaluation Logic"""
    def __init__(self,
                 cost_matrix: Dict[str, Dict[str, float]],
                 heading_matrix: Dict[str, Dict[str, Tuple[float, float]]],
                 initial_headings: Dict[str, float],
                 w_max: float,
                 robots: List[str],
                 goals: List[str],
                 collections: List[str]):
        self.matrix = cost_matrix
        self.heading_matrix = heading_matrix
        self.initial_headings = initial_headings
        self.w_max = w_max if w_max > 0.01 else 0.1
        self.robots = robots
        self.goals = goals
        self.collections = collections

    def _evaluate_makespan(self, solution: Dict[str, RobotSchedule]) -> float:
        max_time = 0.0
        for r_name, schedule in solution.items():
            current_node = r_name 
            current_heading = self.initial_headings.get(r_name, 0.0)
            total_time = 0.0
            
            for task in schedule.tasks:
                # --- A. MOVE TO GOAL ---
                d1 = self.matrix.get(current_node, {}).get(task.goal_id, float('inf'))
                if d1 == float('inf'): return float('inf')
                
                angles_1 = self.heading_matrix.get(current_node, {}).get(task.goal_id, (0.0, 0.0))
                diff = (angles_1[0] - current_heading + math.pi) % (2 * math.pi) - math.pi
                total_time += (abs(diff) / self.w_max) + d1
                current_heading = angles_1[1]
                
                # --- B. MOVE TO COLLECTION ---
                d2 = self.matrix.get(task.goal_id, {}).get(task.collection_id, float('inf'))
                if d2 == float('inf'): return float('inf')
                
                angles_2 = self.heading_matrix.get(task.goal_id, {}).get(task.collection_id, (0.0, 0.0))
                diff = (angles_2[0] - current_heading + math.pi) % (2 * math.pi) - math.pi
                total_time += (abs(diff) / self.w_max) + d2
                current_heading = angles_2[1]
                
                current_node = task.collection_id
            
            if total_time > max_time:
                max_time = total_time
        return max_time

    def _calculate_schedule_duration(self, r_name, tasks) -> float:
        """Calculates total time for a specific task list including turns."""
        current_node = r_name
        current_heading = self.initial_headings.get(r_name, 0.0)
        total_time = 0.0
        
        for task in tasks:
            # 1. To Goal
            d1 = self.matrix.get(current_node, {}).get(task.goal_id, 0.0)
            angles_1 = self.heading_matrix.get(current_node, {}).get(task.goal_id, (0.0, 0.0))
            diff1 = (angles_1[0] - current_heading + math.pi) % (2 * math.pi) - math.pi
            total_time += (abs(diff1) / self.w_max) + d1
            current_heading = angles_1[1]
            
            # 2. To Collection
            d2 = self.matrix.get(task.goal_id, {}).get(task.collection_id, 0.0)
            angles_2 = self.heading_matrix.get(task.goal_id, {}).get(task.collection_id, (0.0, 0.0))
            diff2 = (angles_2[0] - current_heading + math.pi) % (2 * math.pi) - math.pi
            total_time += (abs(diff2) / self.w_max) + d2
            current_heading = angles_2[1]
            
            current_node = task.collection_id
            
        return total_time
    
class TaskAllocatorSA(TaskAllocatorBase):
    """Original Simulated Annealing Implementation"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Cache best collections for greedy init (Static selection)
        self.best_collections = {} 
        for g in self.goals:
            best_c = None
            min_c_cost = float('inf')
            if g in self.matrix:
                for c in self.collections:
                    dist = self.matrix[g].get(c, float('inf'))
                    if dist < min_c_cost:
                        min_c_cost = dist
                        best_c = c
            self.best_collections[g] = best_c

    def solve(self, time_limit: float = 2.0) -> Dict[str, List[DeliveryTask]]:
        print(f"--- Running SA Allocator (Time Limit: {time_limit}s) ---")
        start_time = time.time()
        
        current_solution = self._generate_greedy_solution()
        current_cost = self._evaluate_makespan(current_solution)
        
        best_solution_global = {r: sched.clone() for r, sched in current_solution.items()}
        best_cost_global = current_cost
        
        temperature = 100.0
        cooling_rate = 0.95
        iterations = 0

        while (time.time() - start_time) < time_limit:
            iterations += 1
            neighbor_solution = {r: sched.clone() for r, sched in current_solution.items()}
            self._apply_random_mutation(neighbor_solution)
            neighbor_cost = self._evaluate_makespan(neighbor_solution)
            
            delta = neighbor_cost - current_cost
            if delta < 0 or random.random() < math.exp(-delta / max(temperature, 1e-5)):
                current_solution = neighbor_solution
                current_cost = neighbor_cost
                if current_cost < best_cost_global:
                    best_solution_global = {r: sched.clone() for r, sched in current_solution.items()}
                    best_cost_global = current_cost
            
            temperature *= cooling_rate
            if temperature < 0.5: temperature = 100.0 # Reheat

        print(f"SA Finished: Best Cost {best_cost_global:.2f} | Iterations: {iterations}")
        return {r: sched.tasks for r, sched in best_solution_global.items()}
    
    def _generate_greedy_solution(self) -> Dict[str, RobotSchedule]:
        """Assigns tasks to the robot that can finish it soonest, accounting for TURN COSTS."""
        schedules = {r: RobotSchedule(r, []) for r in self.robots}
        robot_completion_times = {r: 0.0 for r in self.robots}
        robot_locations = {r: r for r in self.robots} 
        
        robot_headings = {r: self.initial_headings.get(r, 0.0) for r in self.robots}

        unassigned = list(self.goals)
        
        while unassigned:
            best_r = None
            best_g = None
            best_c = None
            
            # We look for the assignment that results in the lowest TOTAL finish time
            min_finish_time = float('inf') 
            
            # Temp vars to update state after selection
            best_g_end_heading = 0.0
            best_c_end_heading = 0.0

            for g in unassigned:
                c = self.best_collections.get(g)
                if not c: continue
                
                for r in self.robots:
                    curr_loc = robot_locations[r]
                    curr_heading = robot_headings[r] # Get current heading
                    
                    # --- 1. Path: Current -> Goal ---
                    dist_g = self.matrix.get(curr_loc, {}).get(g, float('inf'))
                    if dist_g == float('inf'): continue

                    # Calculate Turn Cost (Transition to Path)
                    angles_g = self.heading_matrix.get(curr_loc, {}).get(g, (0.0, 0.0))
                    path_start_heading = angles_g[0]
                    
                    diff = path_start_heading - curr_heading
                    diff = (diff + math.pi) % (2 * math.pi) - math.pi
                    turn_cost_g = abs(diff) / self.w_max
                    
                    # --- 2. Path: Goal -> Collection ---
                    dist_c = self.matrix.get(g, {}).get(c, float('inf'))
                    if dist_c == float('inf'): continue

                    # Calculate Turn Cost (Transition at Goal)
                    # Robot arrives at goal with heading = angles_g[1]
                    path_end_heading_at_goal = angles_g[1]
                    
                    angles_c = self.heading_matrix.get(g, {}).get(c, (0.0, 0.0))
                    path_start_heading_coll = angles_c[0]

                    diff = path_start_heading_coll - path_end_heading_at_goal
                    diff = (diff + math.pi) % (2 * math.pi) - math.pi
                    turn_cost_c = abs(diff) / self.w_max

                    # --- Total Time Calculation ---
                    # Previous Time + Turn1 + Travel1 + Turn2 + Travel2
                    new_finish_time = robot_completion_times[r] + turn_cost_g + dist_g + turn_cost_c + dist_c
                    
                    if new_finish_time < min_finish_time:
                        min_finish_time = new_finish_time
                        best_r = r
                        best_g = g
                        best_c = c
                        # Store the final heading (arriving at collection)
                        best_c_end_heading = angles_c[1]

            if best_r:
                schedules[best_r].tasks.append(DeliveryTask(best_g, best_c))
                robot_completion_times[best_r] = min_finish_time
                robot_locations[best_r] = best_c 
                
                robot_headings[best_r] = best_c_end_heading
                
                unassigned.remove(best_g)
            else:
                break 
                
        return schedules
    
    def _apply_random_mutation(self, solution: Dict[str, RobotSchedule]):
        """Applies random swaps or dropoff changes."""
        r_names = list(solution.keys())
        r1_name = random.choice(r_names)
        sched1 = solution[r1_name]
        
        move_type = random.choice(['swap_owner', 'swap_order', 'change_dropoff'])
        
        if not sched1.tasks and move_type != 'swap_owner':
            move_type = 'swap_owner'

        if move_type == 'swap_owner':
            r2_name = random.choice(r_names)
            sched2 = solution[r2_name]
            
            if sched1.tasks or sched2.tasks:
                # Determine source and dest
                if sched1.tasks and sched2.tasks:
                    source, dest = (sched1, sched2) if random.random() < 0.5 else (sched2, sched1)
                elif sched1.tasks:
                    source, dest = sched1, sched2
                else:
                    source, dest = sched2, sched1
                
                # Pop and Insert
                task = source.tasks.pop(random.randint(0, len(source.tasks)-1))
                insert_idx = random.randint(0, len(dest.tasks)) # Can insert at end
                dest.tasks.insert(insert_idx, task)

        elif move_type == 'swap_order':
            if len(sched1.tasks) >= 2:
                idx1 = random.randint(0, len(sched1.tasks)-1)
                idx2 = random.randint(0, len(sched1.tasks)-1)
                sched1.tasks[idx1], sched1.tasks[idx2] = sched1.tasks[idx2], sched1.tasks[idx1]

        elif move_type == 'change_dropoff':
            if sched1.tasks:
                task = sched1.tasks[random.randint(0, len(sched1.tasks)-1)]
                # Try to find a valid random collection, not just any random string
                possible_cs = [c for c in self.collections if c != task.collection_id]
                if possible_cs:
                    new_c = random.choice(possible_cs)
                    # Only apply if reachable
                    if self.matrix.get(task.goal_id, {}).get(new_c, float('inf')) < float('inf'):
                        task.collection_id = new_c

class TaskAllocatorLNS(TaskAllocatorBase):
    """
    State-of-the-Art LNS with Post-Repair Smoothing.
    """
    def solve(self, time_limit: float = 2.0) -> Dict[str, List[DeliveryTask]]:
        print(f"--- Running LNS Allocator (Time Limit: {time_limit}s) ---")
        start_time = time.time()
        
        # 1. Start Empty
        current_sol = {r: RobotSchedule(r, []) for r in self.robots}
        
        # 2. Initial Construction
        all_tasks = [DeliveryTask(g, self.collections[0]) for g in self.goals]
        current_sol = self._repair_regret(current_sol, all_tasks)
        
        # CRITICAL FIX: Optimize dropoffs immediately after construction
        self._optimize_solution_dropoffs(current_sol)
        
        best_sol = {r: sched.clone() for r, sched in current_sol.items()}
        best_cost = self._evaluate_makespan(best_sol)
        
        iterations = 0
        n_remove = max(1, min(4, int(len(self.goals) * 0.4)))

        while (time.time() - start_time) < time_limit:
            iterations += 1
            
            # A. Clone
            temp_sol = {r: sched.clone() for r, sched in current_sol.items()}
            
            # B. Destroy
            if random.random() < 0.7:
                temp_sol, removed_tasks = self._destroy_random(temp_sol, n_remove)
            else:
                temp_sol, removed_tasks = self._destroy_worst(temp_sol, n_remove)
            
            # C. Repair (Regret Insertion)
            temp_sol = self._repair_regret(temp_sol, removed_tasks)
            
            # D. SMOOTHING PASS (The Missing Link)
            # Now that we have a full sequence, fix the blind spots.
            self._optimize_solution_dropoffs(temp_sol)
            
            # E. Evaluate
            new_cost = self._evaluate_makespan(temp_sol)
            
            # F. Acceptance
            if new_cost <= best_cost:
                best_cost = new_cost
                best_sol = {r: sched.clone() for r, sched in temp_sol.items()}
                current_sol = temp_sol
            elif random.random() < 0.05:
                current_sol = temp_sol

        print(f"LNS Finished: Best Cost {best_cost:.2f} | Iterations: {iterations}")
        return {r: sched.tasks for r, sched in best_sol.items()}

    def _optimize_solution_dropoffs(self, solution):
        """
        Iterates through the fixed sequence and updates Collection Points
        to minimize total turning + travel time based on the FULL path.
        """
        for r_name, sched in solution.items():
            if not sched.tasks: continue
            
            prev_loc = r_name
            prev_heading = self.initial_headings.get(r_name, 0.0)
            
            for i, task in enumerate(sched.tasks):
                # Identify Next Location
                if i + 1 < len(sched.tasks):
                    next_loc = sched.tasks[i+1].goal_id
                else:
                    next_loc = None # End of schedule
                    
                # Find Best C for: Prev -> Goal -> [C?] -> Next
                best_c = task.collection_id # Default to current
                min_cost = float('inf')
                
                # Pre-calc arrival at Goal
                d_pg = self.matrix.get(prev_loc, {}).get(task.goal_id, float('inf'))
                angles_pg = self.heading_matrix.get(prev_loc, {}).get(task.goal_id, (0.0, 0.0))
                
                # Turn to Goal
                diff_pg = (angles_pg[0] - prev_heading + math.pi) % (2 * math.pi) - math.pi
                cost_pg = (abs(diff_pg) / self.w_max) + d_pg
                heading_at_goal = angles_pg[1]
                
                # Iterate all Collections
                for c in self.collections:
                    # Goal -> C
                    d_gc = self.matrix.get(task.goal_id, {}).get(c, float('inf'))
                    if d_gc == float('inf'): continue
                    
                    angles_gc = self.heading_matrix.get(task.goal_id, {}).get(c, (0.0, 0.0))
                    diff_gc = (angles_gc[0] - heading_at_goal + math.pi) % (2 * math.pi) - math.pi
                    cost_gc = (abs(diff_gc) / self.w_max) + d_gc
                    heading_at_c = angles_gc[1]
                    
                    # C -> Next (if exists)
                    cost_cn = 0.0
                    if next_loc:
                        d_cn = self.matrix.get(c, {}).get(next_loc, float('inf'))
                        if d_cn == float('inf'): continue
                        angles_cn = self.heading_matrix.get(c, {}).get(next_loc, (0.0, 0.0))
                        diff_cn = (angles_cn[0] - heading_at_c + math.pi) % (2 * math.pi) - math.pi
                        cost_cn = (abs(diff_cn) / self.w_max) + d_cn
                    
                    total = cost_pg + cost_gc + cost_cn
                    
                    if total < min_cost:
                        min_cost = total
                        best_c = c
                
                # Update Task
                task.collection_id = best_c
                
                # Update pointers for next iteration
                prev_loc = best_c
                # Heading leaving C towards Next (or just sitting at C)
                # If next_loc exists, we leave C. If not, we stay.
                if next_loc:
                    prev_heading = self.heading_matrix.get(best_c, {}).get(next_loc, (0.0, 0.0))[1]
                else:
                    # Heading arriving at C
                    prev_heading = self.heading_matrix.get(task.goal_id, {}).get(best_c, (0.0, 0.0))[1]


    def _destroy_random(self, solution, n):
        all_tasks_flat = []
        for r, sched in solution.items():
            for t in sched.tasks: all_tasks_flat.append((r, t))
        if not all_tasks_flat: return solution, []
        to_remove = random.sample(all_tasks_flat, k=min(n, len(all_tasks_flat)))
        for r_name, task in to_remove:
            solution[r_name].tasks.remove(task)
        return solution, [t for r, t in to_remove]

    def _destroy_worst(self, solution, n):
        candidates = []
        for r_name, sched in solution.items():
            if not sched.tasks: continue
            prev_loc = r_name
            for i, task in enumerate(sched.tasks):
                next_loc = sched.tasks[i+1].goal_id if i+1 < len(sched.tasks) else None
                # Simple distance proxy for removal is usually sufficient and fast
                d1 = self.matrix.get(prev_loc, {}).get(task.goal_id, 0)
                d2 = self.matrix.get(task.goal_id, {}).get(task.collection_id, 0)
                d3 = self.matrix.get(task.collection_id, {}).get(next_loc, 0) if next_loc else 0
                shortcut = self.matrix.get(prev_loc, {}).get(next_loc, 0) if next_loc else 0
                savings = (d1 + d2 + d3) - shortcut
                candidates.append((savings, r_name, task))
                prev_loc = task.collection_id
        candidates.sort(key=lambda x: x[0], reverse=True)
        for _, r_name, task in candidates[:n]:
            if task in solution[r_name].tasks:
                solution[r_name].tasks.remove(task)
        return solution, [t for _, _, t in candidates[:n]]

    def _repair_regret(self, solution, tasks_to_insert):
        # Randomize order slightly to break ties
        random.shuffle(tasks_to_insert)
        
        # [NEW] Pre-calculate current duration for all robots to ensure Load Balancing
        robot_finish_times = {
            r: self._calculate_schedule_duration(r, sched.tasks) 
            for r, sched in solution.items()
        }

        while tasks_to_insert:
            regrets = []
            
            for task in tasks_to_insert:
                options = []
                
                for r_name, sched in solution.items():
                    current_r_time = robot_finish_times[r_name]
                    
                    # Try every index
                    for i in range(len(sched.tasks) + 1):
                        cost_increase, best_c = self._calc_insertion_cost(sched, task, i)
                        
                        # [CRITICAL FIX]
                        # The "Cost" is not just the added time, but the NEW FINISH TIME.
                        # This forces the algorithm to give tasks to idle robots.
                        new_finish_time = current_r_time + cost_increase
                        
                        options.append((new_finish_time, cost_increase, r_name, i, best_c))
                
                # [Old] Sort by LOWEST FINISH TIME (Min-Max Objective)
                # options.sort(key=lambda x: x[0])

                # [New] Weighted Objective (Balance Makespan vs Efficiency)
                # x[0] is new_finish_time (Makespan impact)
                # x[1] is cost_increase (Efficiency impact)
                alpha = 0.7 # Tuning parameter: 70% importance on Makespan, 30% on Efficiency
                options.sort(key=lambda x: (alpha * x[0]) + ((1 - alpha) * x[1]))
                
                if not options: continue
                
                best = options[0]
                second_best_time = options[1][0] if len(options) > 1 else float('inf')
                
                # REGRET: How much does the MAKESPAN increase if we don't pick the best?
                regret_val = second_best_time - best[0]
                
                regrets.append({
                    'regret': regret_val,
                    'task': task,
                    'robot': best[2],
                    'idx': best[3],
                    'coll': best[4],
                    'cost_increase': best[1] # Keep track to update cache
                })
            
            # Sort tasks by Regret (Highest first)
            if not regrets: break
            regrets.sort(key=lambda x: x['regret'], reverse=True)
            
            winner = regrets[0]
            
            # Execute Insertion
            t = winner['task']
            t.collection_id = winner['coll']
            r_name = winner['robot']
            
            solution[r_name].tasks.insert(winner['idx'], t)
            
            # [NEW] Update the cache for the specific robot
            robot_finish_times[r_name] += winner['cost_increase']
            
            tasks_to_insert.remove(t)
            
        return solution

    def _calc_insertion_cost(self, schedule, task, idx):
        # 1. Identify Context
        if idx == 0:
            prev_loc = schedule.robot_name
            prev_heading = self.initial_headings.get(prev_loc, 0.0)
        else:
            prev_task = schedule.tasks[idx-1]
            prev_loc = prev_task.collection_id
            # Estimate heading arriving at Prev Collection
            pg = prev_task.goal_id
            pc = prev_task.collection_id
            prev_heading = self.heading_matrix.get(pg, {}).get(pc, (0.0, 0.0))[1]

        next_loc = schedule.tasks[idx].goal_id if idx < len(schedule.tasks) else None
        
        # 2. Base Cost: Prev -> Goal (Fixed)
        d_pg = self.matrix.get(prev_loc, {}).get(task.goal_id, float('inf'))
        if d_pg == float('inf'): return float('inf'), None
        
        angles_pg = self.heading_matrix.get(prev_loc, {}).get(task.goal_id, (0.0, 0.0))
        turn_pg = abs((angles_pg[0] - prev_heading + math.pi) % (2*math.pi) - math.pi) / self.w_max
        cost_pg = turn_pg + d_pg
        heading_at_goal = angles_pg[1]
        
        # 3. Optimize C
        best_c = None
        min_seg = float('inf')
        
        for c in self.collections:
            # Goal -> C
            d_gc = self.matrix.get(task.goal_id, {}).get(c, float('inf'))
            if d_gc == float('inf'): continue
            
            angles_gc = self.heading_matrix.get(task.goal_id, {}).get(c, (0.0, 0.0))
            turn_gc = abs((angles_gc[0] - heading_at_goal + math.pi) % (2*math.pi) - math.pi) / self.w_max
            cost_gc = turn_gc + d_gc
            heading_at_c = angles_gc[1]
            
            # C -> Next (Approximate if next exists)
            cost_cn = 0.0
            if next_loc:
                d_cn = self.matrix.get(c, {}).get(next_loc, float('inf'))
                if d_cn == float('inf'): continue
                angles_cn = self.heading_matrix.get(c, {}).get(next_loc, (0.0, 0.0))
                turn_cn = abs((angles_cn[0] - heading_at_c + math.pi) % (2*math.pi) - math.pi) / self.w_max
                cost_cn = turn_cn + d_cn
            
            if (cost_gc + cost_cn) < min_seg:
                min_seg = cost_gc + cost_cn
                best_c = c
                
        if best_c is None: return float('inf'), None
        
        # 4. Marginal Delta
        added = cost_pg + min_seg
        removed = 0.0
        if next_loc:
            d_pn = self.matrix.get(prev_loc, {}).get(next_loc, float('inf'))
            if d_pn < float('inf'):
                angles_pn = self.heading_matrix.get(prev_loc, {}).get(next_loc, (0.0, 0.0))
                turn_pn = abs((angles_pn[0] - prev_heading + math.pi) % (2*math.pi) - math.pi) / self.w_max
                removed = turn_pn + d_pn
                
        return added - removed, best_c
    
class TaskAllocatorLNS2(TaskAllocatorLNS):
    """
    Experimental Variant of LNS with Alternative Destruction and Repair Strategies.
    Currently not in use.
    """
    pass