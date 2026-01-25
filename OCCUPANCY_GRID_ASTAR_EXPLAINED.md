# OCCUPANCY GRID AND A* INTEGRATION - DETAILED EXPLANATION

This document explains how the occupancy grid works and how it's integrated with the A* path planning algorithm in the MPC Wall-Aware Node.

---

## PART 1: OCCUPANCY GRID FUNDAMENTALS

### 1.1 WHAT IS AN OCCUPANCY GRID?

An occupancy grid is a probabilistic spatial representation that discretizes continuous space into a grid of cells. Each cell stores a probability value (0.0 to 1.0) representing the likelihood that the cell is occupied by an obstacle.

**VISUALIZATION:**
- 0.0 = Free space (definitely clear)
- 0.5 = Uncertain (might be occupied)
- 1.0 = Occupied (definitely has an obstacle)

**Example 5x5 grid:**
```
[0.0  0.0  0.0  0.0  0.0]
[0.0  0.3  0.8  0.3  0.0]
[0.0  0.5  1.0  0.5  0.0]
[0.0  0.3  0.8  0.3  0.0]
[0.0  0.0  0.0  0.0  0.0]
```

Here, the center cell (1.0) is definitely occupied, while surrounding cells have lower probabilities representing uncertainty or inflation zones.

### 1.2 GRID CONFIGURATION

Location: [mpc_wall_aware_node.py:61-71](drone_ws/src/drone/drone/mpc_wall_aware_node.py#L61-L71)

**Parameters:**
- resolution: 0.2 meters per cell
- width: 100 cells
- height: 100 cells
- origin_x: -10.0 meters (world coordinates)
- origin_y: -10.0 meters (world coordinates)

**Total Coverage:**
- Grid size: 100 × 100 = 10,000 cells
- Physical area: 20m × 20m = 400 square meters
- Grid covers world coordinates from (-10, -10) to (10, 10)

**COORDINATE SYSTEM:**
```
World coordinates (continuous) → Grid coordinates (discrete)

Example conversion:
    World position (0.5, 0.5) meters
    → Grid cell: gx = (0.5 - (-10.0)) / 0.2 = 52
                  gy = (0.5 - (-10.0)) / 0.2 = 52
    → Grid[52, 52]
```

---

## PART 2: HOW THE OCCUPANCY GRID IS BUILT

### 2.1 THE UPDATE PIPELINE (EVERY CONTROL CYCLE)

The grid is updated every control cycle (default: 10 Hz) with fresh LiDAR data.

**STEP-BY-STEP PROCESS:**

#### 1. LiDAR SCAN PROCESSING (Control Loop: lines 705-712)

**Input:** Raw LiDAR scan (LaserScan message)
- Contains: ranges[], angle_min, angle_max, angle_increment
- Example: 360 laser beams, each reporting distance to nearest obstacle

**Process:**

a) `scan_to_world_points()`:
- Converts polar coordinates (angle, range) to world XYZ positions
- Accounts for drone's current position and orientation
- Result: raw_points (N × 3 array)

b) `cluster_points()`:
- Groups nearby points together (within 0.3m)
- Reduces noise and redundant detections
- Result: clustered_points (M × 3 array, where M < N)

**Example:**
- Raw LiDAR: 500 points
- After clustering: 50 obstacle centers

#### 2. TEMPORAL DECAY (update() method: line 97)

Before adding new observations, decay old evidence:

```python
self.grid *= 0.95
```

**Purpose:** Implements "memory decay"
- Obstacles not seen recently fade away
- Prevents stale data from blocking paths
- Allows grid to adapt to dynamic environments

**Effect over time (if obstacle not re-detected):**
```
Cycle 0: grid[y, x] = 1.0   (fully occupied)
Cycle 1: grid[y, x] = 0.95  (slight fade)
Cycle 2: grid[y, x] = 0.90
Cycle 5: grid[y, x] = 0.77
Cycle 10: grid[y, x] = 0.60
Cycle 20: grid[y, x] = 0.36
Cycle 50: grid[y, x] = 0.08  (almost forgotten)
```

#### 3. EVIDENCE ACCUMULATION (update() method: lines 100-108)

For each detected obstacle:

```python
for obs in obstacles:
    gx, gy = self.world_to_grid(obs)  # Convert to grid coordinates
    if self.is_valid(gx, gy):         # Check bounds
        # Add evidence, capped at 1.0
        self.grid[gy, gx] = min(1.0, self.grid[gy, gx] + 0.5)
```

**Evidence Accumulation Logic:**
- Each detection adds +0.5 to cell probability
- Maximum value is 1.0 (cannot exceed)
- Creates confidence through repeated observations

**Example timeline for a stationary wall:**
```
Before detection: grid[y, x] = 0.0
Cycle 1 (detected): 0.0 + 0.5 = 0.5
Cycle 2 (detected): 0.5 * 0.95 (decay) = 0.475, then 0.475 + 0.5 = 0.975
Cycle 3 (detected): 0.975 * 0.95 = 0.926, then min(1.0, 0.926 + 0.5) = 1.0
Cycle 4+ (detected): Stays at 1.0 (max)
```

Equilibrium: For continuously detected obstacles, value stabilizes at 1.0

#### 4. OBSTACLE INFLATION (update() method: line 111)

After marking obstacles, inflate them for safety:

```python
self._inflate_obstacles(inflation_cells=2)
```

**Process (lines 113-118):**
a) Identify occupied cells: `occupied = self.grid > 0.5`
b) Apply binary dilation (morphological operation):
- Expands each obstacle outward by 2 cells (0.4 meters)
- Uses a 3×3 structuring element
- Each iteration grows obstacles by 1 cell in all 8 directions

c) Update grid values:
- In inflated regions: set to at least 0.7
- Keeps original value if higher
- Non-inflated regions: unchanged

**VISUAL EXAMPLE:**

Original grid (values > 0.5 shown):
```
. . . . . . .
. . . . . . .
. . . X . . .    X = obstacle at 1.0
. . . . . . .
. . . . . . .
```

After 2 iterations of dilation:
```
. . # # # . .
. # # # # # .
. # # X # # .    X = original (1.0)
. # # # # # .    # = inflated (0.7)
. . # # # . .    . = free (original value)
```

**Purpose:**
- Safety margin for drone physical size
- Accounts for control errors
- Conservative path planning

### 2.2 GRID QUERYING

The grid provides a method to check if a position is free:

```python
def is_free(self, pos: np.ndarray, threshold: float = 0.3) -> bool:
    gx, gy = self.world_to_grid(pos)
    if not self.is_valid(gx, gy):
        return False
    return self.grid[gy, gx] < threshold
```

**Usage:**
- threshold = 0.3 (default): cells with probability < 0.3 are "free"
- threshold = 0.5 (in A*): more conservative, only < 0.5 is free
- Returns False if position is outside grid bounds

---

## PART 3: A* PATH PLANNING ALGORITHM

### 3.1 WHAT IS A*?

A* is a graph-based path planning algorithm that finds the shortest path from a start node to a goal node while avoiding obstacles.

**Key Components:**
- Open Set: Nodes to explore (priority queue)
- Closed Set: Already explored nodes
- g_score: Cost from start to current node
- h_score: Heuristic (estimated cost to goal)
- f_score: g_score + h_score (total estimated cost)

**Algorithm:**
1. Start at initial position
2. Explore neighbors with lowest f_score first
3. Skip occupied cells (from occupancy grid)
4. Repeat until goal is reached or no path exists

### 3.2 A* IMPLEMENTATION

Location: [mpc_wall_aware_node.py:128-298](drone_ws/src/drone/drone/mpc_wall_aware_node.py#L128-L298)

**Entry Point (lines 137-148):**

```python
def plan(self, start: np.ndarray, goal: np.ndarray,
         occupied_threshold: float = 0.5) -> Optional[List[np.ndarray]]:

    # Convert world coordinates to grid coordinates
    start_gx, start_gy = self.grid.world_to_grid(start)
    goal_gx, goal_gy = self.grid.world_to_grid(goal)

    # Validate coordinates
    if not self.grid.is_valid(start_gx, start_gy) or
       not self.grid.is_valid(goal_gx, goal_gy):
        return None
```

### 3.3 OCCUPANCY GRID INTEGRATION WITH A*

The occupancy grid is used to determine which cells are traversable:

**KEY INTEGRATION POINT (lines 202-204):**

```python
# Check if occupied
if self.grid.grid[neighbor_y, neighbor_x] > occupied_threshold:
    continue  # Skip this cell, it's occupied!
```

**How it works:**
1. A* generates a candidate neighbor cell (neighbor_x, neighbor_y)
2. Looks up the occupancy value: `self.grid.grid[neighbor_y, neighbor_x]`
3. Compares to threshold (0.5 by default)
4. If value > 0.5 → cell is occupied → skip it
5. If value ≤ 0.5 → cell is free → consider it for path

**INTERPRETATION OF THRESHOLD = 0.5:**
- Values 0.0 - 0.5: Treated as free space (safe to traverse)
- Values 0.5 - 1.0: Treated as occupied (avoid)
- This conservative approach ensures safety

**Example grid during A* search:**

```
Grid values:
[0.0  0.0  0.0  0.0  0.0]
[0.0  0.3  0.8  0.3  0.0]   0.8 > 0.5 → BLOCKED
[0.0  0.5  1.0  0.5  0.0]   1.0 > 0.5 → BLOCKED
[0.0  0.3  0.8  0.3  0.0]   0.5 = 0.5 → BLOCKED (boundary case)
[0.0  0.0  0.0  0.0  0.0]

A* path will navigate around the blocked cells.
```

### 3.4 NEIGHBOR EXPLORATION (lines 168, 192-215)

A* uses 8-connected grid navigation:

```python
neighbors = [(-1,-1), (-1,0), (-1,1),  # Top row
             (0,-1),          (0,1),    # Middle row
             (1,-1),  (1,0),  (1,1)]   # Bottom row
```

For each current cell, A* checks all 8 neighbors:

```python
for dx, dy in neighbors:
    neighbor_x = current_x + dx
    neighbor_y = current_y + dy

    # Validation checks:
    if not self.grid.is_valid(neighbor_x, neighbor_y):
        continue  # Out of bounds

    if (neighbor_x, neighbor_y) in closed_set:
        continue  # Already explored

    # OCCUPANCY GRID CHECK (KEY INTEGRATION):
    if self.grid.grid[neighbor_y, neighbor_x] > occupied_threshold:
        continue  # Occupied by obstacle

    # Calculate cost
    move_cost = sqrt(dx² + dy²)  # Diagonal = 1.414, straight = 1.0
    tentative_g = g_score[(current_x, current_y)] + move_cost

    # Update if better path found
    if neighbor not in g_score or tentative_g < g_score[neighbor]:
        came_from[neighbor] = current
        g_score[neighbor] = tentative_g
        h = heuristic(neighbor, goal)
        f_score[neighbor] = tentative_g + h
        heappush(open_set, (f_score[neighbor], neighbor_x, neighbor_y))
```

### 3.5 HEURISTIC FUNCTION (lines 220-222)

A* uses Euclidean distance as the heuristic:

```python
def _heuristic(self, x1, y1, x2, y2):
    return math.sqrt((x2 - x1)² + (y2 - y1)²)
```

This is admissible (never overestimates) and guarantees optimal paths.

### 3.6 PATH RECONSTRUCTION (lines 224-239)

When goal is reached, A* reconstructs the path by following the came_from links:

```python
def _reconstruct_path(self, came_from, current_x, current_y):
    path = []
    current = (current_x, current_y)

    while current in came_from:
        pos = self.grid.grid_to_world(current[0], current[1])  # Grid → World
        path.append(pos)
        current = came_from[current]

    path.append(self.grid.grid_to_world(current[0], current[1]))  # Add start
    path.reverse()  # Start → Goal order
    return path
```

Output: List of waypoints in world coordinates (not grid coordinates)

### 3.7 PATH SIMPLIFICATION (lines 241-297)

Raw A* paths can have many redundant waypoints. Simplification removes unnecessary intermediate points using line-of-sight checks:

```python
def _simplify_path(self, path, threshold):
    if len(path) <= 2:
        return path

    simplified = [path[0]]  # Always keep start
    i = 0

    while i < len(path) - 1:
        # Find farthest visible point from path[i]
        j = len(path) - 1
        while j > i + 1:
            if self._has_line_of_sight(path[i], path[j], threshold):
                break  # Found farthest visible point
            j -= 1

        simplified.append(path[j])
        i = j

    return simplified
```

**Line-of-Sight Check (Bresenham's Algorithm, lines 265-297):**
- Traces a line between two waypoints on the grid
- Checks all cells along the line
- Returns True only if ALL cells are free (< threshold)
- Returns False if any cell is occupied or out of bounds

**Effect:**
```
Original path:  [A] → [B] → [C] → [D] → [E] → [F]
Simplified:     [A] ---------> [D] ---------> [F]

(Removed B, C, E because A can "see" D directly, and D can "see" F)
```

---

## PART 4: COMPLETE INTEGRATION WORKFLOW

### 4.1 HIERARCHICAL CONTROL ARCHITECTURE

The system uses a two-layer approach:

```
┌─────────────────────────────────────────────┐
│  GLOBAL LAYER (Strategic Planning)          │
│  - Occupancy Grid Mapping                   │
│  - A* Path Planning                         │
│  - Waypoint Generation                      │
└──────────────┬──────────────────────────────┘
               │ Waypoints
               ▼
┌─────────────────────────────────────────────┐
│  LOCAL LAYER (Tactical Control)             │
│  - Model Predictive Control (MPC)           │
│  - Smooth Trajectory Optimization           │
│  - Real-time Obstacle Avoidance             │
└─────────────────────────────────────────────┘
```

### 4.2 GLOBAL REPLANNING (lines 717-721, 769-786)

Global planning runs periodically (default: every 20 control cycles = 2 seconds):

```python
self.replan_counter += 1
if self.replan_counter >= self.replan_interval:
    self.replan_counter = 0
    self._replan_global_path()
```

**Replanning Process:**

```python
def _replan_global_path(self):
    # Use A* to find path from current position to final goal
    path = self.planner.plan(self.state.position, self.goal)

    if path is not None and len(path) > 1:
        # Remove first waypoint (too close to drone)
        if len(path) > 2:
            path = path[1:]

        # Update waypoint queue
        self.waypoint_manager.set_path(path)
        self.get_logger().info(f'Global path planned: {len(path)} waypoints')
    else:
        # No path found, attempt direct navigation
        self.waypoint_manager.clear()
        self.get_logger().warn('No global path found, going direct to goal')
```

**Why Periodic Replanning?**
- Occupancy grid changes as drone moves and sees new obstacles
- Dynamic obstacles may appear/disappear
- Initially blocked paths may become available
- Ensures path stays optimal as environment updates

### 4.3 WAYPOINT MANAGEMENT (lines 304-349)

The WaypointManager maintains a queue of waypoints from A*:

```python
class WaypointManager:
    waypoint_radius: float = 1.0  # Switch threshold
    waypoints: deque  # Queue of upcoming waypoints
    current_waypoint: Optional[np.ndarray]  # Active target
```

**Update Logic (every control cycle):**

```python
def update(self, drone_pos, final_goal):
    if self.current_waypoint is None:
        return final_goal  # No waypoints, target final goal

    # Check if reached current waypoint
    dist = norm(drone_pos[:2] - self.current_waypoint[:2])

    if dist < self.waypoint_radius:  # Within 1.0 meter
        # Advance to next waypoint
        if self.waypoints:
            self.current_waypoint = self.waypoints.popleft()
        else:
            # Reached all waypoints
            self.current_waypoint = None
            return final_goal

    return self.current_waypoint
```

**Example sequence:**
```
Path from A*: [W1, W2, W3]

Cycle 1-50: Target W1, distance = 3.2m → Keep targeting W1
Cycle 51:   Target W1, distance = 0.8m → Switch to W2
Cycle 52-100: Target W2, distance = 2.5m → Keep targeting W2
Cycle 101:  Target W2, distance = 0.9m → Switch to W3
Cycle 102-150: Target W3, distance = 1.8m → Keep targeting W3
Cycle 151:  Target W3, distance = 0.7m → Switch to final_goal
Cycle 152+: Target final_goal
```

### 4.4 LOCAL MPC CONTROL (lines 723-738)

Once waypoints are set, MPC handles smooth trajectory execution:

```python
# Get current target waypoint
active_goal = self.waypoint_manager.update(self.state.position, self.goal)

# Build reference trajectory to active goal
reference = self._build_straight_reference(active_goal)
yaw_ref = self._build_yaw_reference(active_goal)

# Solve MPC optimization
obstacles = ObstacleSet(clustered_points, self.mpc_config)
result = self.mpc_solver.solve(
    state=self.state,
    goal=active_goal,  # Not final goal, but current waypoint!
    obstacles=obstacles,
    reference_trajectory=reference,
    yaw_reference=yaw_ref
)
```

**Key:** MPC targets the current waypoint from A*, not the final goal directly. This allows it to follow the globally optimal path while handling local obstacles smoothly.

### 4.5 COMPLETE EXAMPLE SCENARIO

**Scenario:** Drone needs to reach goal but there's a wall in the way

**Initial Setup:**
- Drone position: (0, 0, 1.5)
- Final goal: (10, 0, 1.5)
- Wall obstacle: x = 5, spanning y = -2 to y = 2

**CYCLE 1-10: Initial exploration**
- LiDAR detects wall
- Occupancy grid marks cells around x=5, y=-2 to y=2 as occupied
- Grid values build up to 1.0
- Inflation expands wall to ~2 cells (0.4m) on each side

**CYCLE 11: First global planning**
- A* runs on occupancy grid
- Finds wall blocking direct path
- Plans path around wall (either above or below)
- Example path: [(0,0), (3,3), (7,3), (10,0)]
- Waypoint queue: [W1=(3,3), W2=(7,3)]

**CYCLE 11-50: Navigate to W1**
- active_goal = (3, 3)
- MPC generates smooth trajectory toward W1
- Handles small local obstacles
- Grid continues updating with fresh LiDAR data

**CYCLE 51: Reached W1**
- Distance to W1 < 1.0m
- Waypoint manager switches to W2
- active_goal = (7, 3)

**CYCLE 51-100: Navigate to W2**
- MPC targets W2
- Passes around the wall
- Wall remains in occupancy grid but drone is past it

**CYCLE 101: Reached W2**
- Distance to W2 < 1.0m
- No more waypoints in queue
- active_goal = (10, 0) [final goal]

**CYCLE 101-150: Navigate to final goal**
- MPC directly targets final goal
- No obstacles in the way
- Smooth approach to goal

**CYCLE 151: Goal reached**
- Distance to goal < 0.5m (goal_threshold)
- Mission complete

**Throughout:**
- Occupancy grid continuously updates (decay + new observations)
- Global replanning every 20 cycles (can adapt to changes)
- MPC always follows waypoints from global plan
- Hierarchical approach solves local minima problem

---

## PART 5: KEY ADVANTAGES OF THIS APPROACH

### 5.1 WHY OCCUPANCY GRID?

**Advantages:**
- ✓ Handles noisy LiDAR data gracefully (probabilistic approach)
- ✓ Temporal decay allows adaptation to dynamic environments
- ✓ Compact representation (100×100 grid vs thousands of points)
- ✓ Fast lookups for collision checking
- ✓ Works well with graph-based planners like A*

**Disadvantages:**
- ✗ Fixed resolution (can't represent fine details < 0.2m)
- ✗ Memory usage grows with area (100×100 = 10,000 cells)
- ✗ Limited to local area around drone

### 5.2 WHY HIERARCHICAL PLANNING?

**Problem: Pure MPC Struggles with Walls**
- MPC optimizes locally (short horizon, ~2 seconds)
- Can't "see" that it needs to go around long obstacles
- Gets stuck in local minima

**Solution: Global + Local Layers**
- Global layer (A*): Finds strategic waypoints around obstacles
- Local layer (MPC): Executes smooth trajectories between waypoints
- Combines long-term planning with short-term optimization

### 5.3 WHY A* ON OCCUPANCY GRID?

**Advantages of A*:**
- ✓ Guarantees optimal path (if one exists)
- ✓ Fast with good heuristic
- ✓ Works naturally with grid representation
- ✓ Robust to complex obstacle configurations

**Integration benefits:**
- ✓ Occupancy grid provides clean binary decisions (free/occupied)
- ✓ Inflation ensures safety margins automatically
- ✓ Temporal decay adapts to environment changes
- ✓ Path simplification reduces waypoint count for MPC

---

## PART 6: VISUALIZATION OF DATA FLOW

Complete Pipeline (every 0.1 seconds):

```
LiDAR Sensor
    │
    ├──> scan_to_world_points()
    │        │
    │        └──> Raw Points (500 points)
    │
    ├──> cluster_points()
    │        │
    │        └──> Clustered Points (50 obstacles)
    │
    ▼
OccupancyGridMapper.update()
    │
    ├──> Decay: grid *= 0.95
    │
    ├──> Mark: grid[y,x] += 0.5 for each obstacle
    │
    ├──> Inflate: binary_dilation(grid, iterations=2)
    │
    └──> Updated Grid (100×100 probabilities)
         │
         ▼
    [Every 20 cycles: Replan]
         │
         ├──> AStarPlanner.plan(start, goal)
         │        │
         │        ├──> Explore grid cells
         │        ├──> Skip cells where grid[y,x] > 0.5
         │        ├──> Find optimal path
         │        └──> Simplify path
         │        │
         │        └──> Waypoints (3-5 points)
         │
         └──> WaypointManager.set_path(waypoints)
              │
              └──> Waypoint Queue: [W1, W2, W3]

    [Every cycle: Local control]
         │
         ├──> WaypointManager.update()
         │        │
         │        └──> active_goal = W1 (or W2, W3, or final_goal)
         │
         └──> MPCSolver.solve(active_goal, obstacles)
                   │
                   ├──> Optimize trajectory
                   ├──> Avoid local obstacles
                   └──> Generate acceleration command
                        │
                        └──> Drone executes command
```

---

## PART 7: PARAMETER TUNING GUIDE

### 7.1 OCCUPANCY GRID PARAMETERS

**resolution (0.2m):**
- Smaller → More detail, higher memory/computation
- Larger → Faster, less detail
- Rule: Should be ~2× safety_radius / grid_size

**decay_rate (0.95):**
- Higher (0.99) → Longer memory, slow adaptation
- Lower (0.90) → Shorter memory, fast adaptation
- Current: ~20 cycles to drop from 1.0 to 0.36

**evidence_increment (0.5):**
- Higher → Fewer observations needed to mark occupied
- Lower → More robust to noise, slower buildup
- Current: 2 detections → 100% confidence

**inflation_cells (2):**
- More → Larger safety margin, more conservative paths
- Less → Closer to obstacles, riskier
- Current: 0.4m inflation (2 × 0.2m)

### 7.2 A* PARAMETERS

**occupied_threshold (0.5):**
- Higher → More permissive, can traverse uncertain areas
- Lower → More conservative, only traverse very free areas
- Should match evidence_increment (0.5)

**replan_interval (20 cycles = 2 seconds):**
- Shorter → More adaptive, higher computation
- Longer → Less computation, slower adaptation
- Balance: Environment dynamics vs computational budget

**waypoint_radius (1.0m):**
- Smaller → More precise waypoint following
- Larger → Faster progress, less precise
- Should be > resolution for stability

---

## SUMMARY

The occupancy grid and A* integration provides a robust hierarchical navigation system:

1. **OCCUPANCY GRID** maps the environment probabilistically
   - Temporal decay for dynamic adaptation
   - Evidence accumulation for robustness
   - Inflation for safety margins

2. **A* PLANNING** finds optimal paths around obstacles
   - Uses occupancy values to determine traversability
   - Operates on the discrete grid representation
   - Generates waypoints for MPC to follow

3. **HIERARCHICAL CONTROL** combines global and local optimization
   - Global: A* handles long-term planning around large obstacles
   - Local: MPC handles smooth execution and local obstacle avoidance
   - Integration: Waypoint manager coordinates between layers

This architecture solves the fundamental limitation of pure MPC (local minima) while maintaining smooth, optimal trajectories for drone navigation.

---

**END OF DOCUMENT**
