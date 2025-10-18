import sys
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import copy
from statistics import mean
import matplotlib.pyplot as plt
import matplotlib.animation as animation

parent_dir = os.path.dirname(os.getcwd())
sys.path.append(parent_dir)

from agents import *
from agents import XYEnvironment, Agent
from search import Problem, breadth_first_graph_search, uniform_cost_search, iterative_deepening_search,depth_first_graph_search
import time



# =====================================================================================================
                           # PART 1.1: BUILDING YOUR WORLD
# =====================================================================================================

# =====================================================================================================
# --- Environment tile codes ---
# I used a grid-based environment, so defining separate Thing subclasses (like Floor or Wall) was unnecessary.
# Custom: Created for this project to represent 2D floor plan tiles.
# (AIMA does not include this exact representation.)
# =====================================================================================================
FLOOR = 0
WALL = 1
STAIRS = 2
OBSTACLE = 3
GOAL = 4

# =====================================================================================================
""" A. Environment Setup """
# Custom environment built using AIMA’s XYEnvironment base class.
# XYEnvironment → AIMA library (agents.py)
# setup_environment(), percept(), execute_action() are custom extensions.
# =====================================================================================================
"""
In this part, I created a 2D environment that represents a house floor plan. 
The idea is that the agent moves with a white cane, trying to reach a known goal 
while avoiding stairs or obstacles that it discovers through its sensors. 
The world is partially observable, deterministic, and static.
"""
# =====================================================================================================

class NavigationEnvironment(XYEnvironment):
    """2D house-like grid world for the navigation task."""

    def __init__(self, width=10, height=10, obstacle_prob=0.1):
        # AIMA’s XYEnvironment init call (unchanged)
        super().__init__(width, height)
        self.obstacle_prob = obstacle_prob
        self.width = width
        self.height = height
        self.grid = np.zeros((height, width), dtype=int)
        self.goal_location = None
        self.setup_environment() # Custom initialization

    def setup_environment(self):
        """Custom: Defines walls, obstacles, stairs, and goal dynamically."""
        """Set walls, obstacles, stairs and goal."""
        # walls around borders
        self.grid[0, :] = WALL
        self.grid[-1, :] = WALL
        self.grid[:, 0] = WALL
        self.grid[:, -1] = WALL

        # some inner walls to form rooms
        self.grid[5, 2:8] = WALL
        self.grid[2:5, 5] = WALL

        # stairs as dangerous zones
        self.grid[7:9, 7:9] = STAIRS

        # random furniture as obstacles
        for x in range(1, self.width - 1):
            for y in range(1, self.height - 1):
                if self.grid[y, x] == FLOOR and random.random() < self.obstacle_prob:
                    self.grid[y, x] = OBSTACLE

        # set known goal location dynamically based on world size
        gx = min(self.width - 2, 8)   
        gy = min(2, self.height - 2)  
        self.goal_location = (gx, gy)
        self.grid[gy, gx] = GOAL

    def get_random_start_and_goal(self):
        """Returns a random reachable (FLOOR) start and goal location."""
        floor_positions = [
            (x, y)
            for y in range(1, self.height - 1)
            for x in range(1, self.width - 1)
            if self.grid[y, x] == FLOOR
        ]

        if len(floor_positions) < 2:
            raise ValueError("Not enough floor tiles for start/goal placement.")

        start = random.choice(floor_positions)
        goal = random.choice(floor_positions)
        while goal == start:
            goal = random.choice(floor_positions)

        self.goal_location = goal
        self.grid[goal[1], goal[0]] = GOAL
        return start, goal
    

    def percept(self, agent):
        """Custom: Returns what the white-cane agent senses around itself."""
        # Uses AIMA percept convention (returns a dictionary percept)
        """
        The agent perceives its immediate surroundings with a white cane.
        It knows its own position and the goal, but not all obstacles in advance.
        """
        x, y = agent.location
        return {
            'location': (x, y),
            'goal': self.goal_location,
            'current': self.grid[y, x],
            'front': self.grid[y - 1, x] if y > 0 else WALL,
            'back': self.grid[y + 1, x] if y < self.height - 1 else WALL,
            'left': self.grid[y, x - 1] if x > 0 else WALL,
            'right': self.grid[y, x + 1] if x < self.width - 1 else WALL
        }

    def execute_action(self, agent, action):
        """Custom override of AIMA’s default. Adds scoring logic and penalties."""
        # Uses deterministic transitions + custom performance metrics
        """Move the agent and update performance depending on what it hits."""
        x, y = agent.location
        nx, ny = x, y

        # move based on chosen action
        if action == 'Forward' and y > 0:
            ny -= 1
        elif action == 'Back' and y < self.height - 1:
            ny += 1
        elif action == 'Left' and x > 0:
            nx -= 1
        elif action == 'Right' and x < self.width - 1:
            nx += 1

        tile = self.grid[ny, nx]
        agent.performance -= 1  # small step cost

        if tile in (FLOOR, GOAL):
            agent.location = (nx, ny)
            if tile == GOAL:
                agent.performance += 1000  # reaching goal reward
        elif tile == OBSTACLE:
            agent.performance -= 10  # bump into furniture
        elif tile == STAIRS:
            agent.performance -= 100  # big penalty for danger

    def is_done(self):
        """Custom: Episode ends when agent reaches the goal tile."""
        # Inspired by AIMA’s done() pattern but written for navigation world
        """Episode ends when the goal is reached."""
        for ag in self.agents:
            if ag.location == self.goal_location:
                return True
        return False
    
    


# =====================================================================================================
""" B. Agent Types """
# i built on top of AIMA’s Agent superclass.
# SimpleReflexNavigationAgent, ModelBasedNavigationAgent, GoalBasedNavigationAgent
# are all CUSTOM — inspired by AIMA’s reflex and model-based agent templates.
# =====================================================================================================
"""
I built three types of agents for this navigation problem:
1. Simple Reflex Agent: reacts immediately based on current percept.
2. Model-Based Agent: keeps a small memory of visited tiles.
3. Goal-Based Agent: knows the goal and stops when it’s reached.
"""   

# =====================================================================================================
# --- 1. Simple Reflex Agent ---
# =====================================================================================================
class SimpleReflexNavigationAgent(Agent):
    # Custom: based on AIMA SimpleReflexAgent idea, but rewritten for 2D movement.
    """Moves toward goal if the path looks safe, otherwise picks any safe direction."""
    def __init__(self):
        super().__init__(program=self.program)

    def program(self, percept):
        loc = percept['location']
        goal = percept['goal']
        x, y = loc
        gx, gy = goal

        if percept['current'] == GOAL or loc == goal:
            return 'NoOp'  # goal reached

        dx, dy = gx - x, gy - y
        # move greedily toward goal
        if dx > 0 and percept['right'] in (FLOOR, GOAL): return 'Right'
        if dx < 0 and percept['left'] in (FLOOR, GOAL): return 'Left'
        if dy > 0 and percept['back'] in (FLOOR, GOAL): return 'Back'
        if dy < 0 and percept['front'] in (FLOOR, GOAL): return 'Forward'

        # fallback safe movement
        for move, tile in (('Forward', percept['front']),
                           ('Right', percept['right']),
                           ('Left', percept['left']),
                           ('Back', percept['back'])):
            if tile == FLOOR:
                return move
        return 'NoOp'


# =====================================================================================================
# --- 2. Model-Based Agent ---
    # Custom: keeps visited-state memory, inspired by AIMA ModelBasedReflexAgent.
# =====================================================================================================
    
class ModelBasedNavigationAgent(Agent):
    """Tracks visited tiles and avoids revisiting the same places if possible."""
    def __init__(self, width=10, height=10):
        super().__init__(program=self.program)
        self.mental_map = {}
        self.visited = set()
        self.width = width
        self.height = height

    def program(self, percept):
        loc = percept['location']
        goal = percept['goal']
        x, y = loc

        # update internal map
        self.visited.add(loc)
        self.mental_map[(x, y)] = percept['current']

        if y > 0: self.mental_map[(x, y - 1)] = percept['front']
        if y < self.height - 1: self.mental_map[(x, y + 1)] = percept['back']
        if x > 0: self.mental_map[(x - 1, y)] = percept['left']
        if x < self.width - 1: self.mental_map[(x + 1, y)] = percept['right']

        if loc == goal:
            return 'NoOp'

        # prefer unvisited safe moves closer to goal
        dx, dy = goal[0] - x, goal[1] - y
        options = []
        for name, (dx_, dy_), tile in [
            ('Right', (1, 0), percept['right']),
            ('Left', (-1, 0), percept['left']),
            ('Back', (0, 1), percept['back']),
            ('Forward', (0, -1), percept['front'])
        ]:
            nx, ny = x + dx_, y + dy_
            if tile in (FLOOR, GOAL):
                visited = (nx, ny) in self.visited
                dist = abs(goal[0] - nx) + abs(goal[1] - ny)
                options.append(((visited, dist), name))

        if options:
            options.sort()
            return options[0][1]

        # last resort
        for move, tile in (('Forward', percept['front']),
                           ('Right', percept['right']),
                           ('Left', percept['left']),
                           ('Back', percept['back'])):
            if tile == FLOOR:
                return move
        return 'NoOp'


# =====================================================================================================
""" B. Goal-Based Navigation Agent """
# Custom hybrid design — uses rule-based “greedy” planning logic.
# AIMA has GoalBasedAgent, but this version adds path planning and internal mapping.
# =====================================================================================================
"""
In this section, I extended the model-based agent into a goal-based navigation agent. 
The agent’s objective is to reach a goal location safely, using the white cane for sensing.

Main idea:
- Define a Goal: The agent aims to reach the goal tile without hitting obstacles or stairs.
- World Model: It builds an internal map (mental model) of what it has sensed so far.
- Percept History: Keeps track of visited tiles and updates the map as it moves.
- Rule-Driven Planning: Uses simple rules to choose the next move toward the goal.

Greedy-like Behavior:
This version does not yet use real search algorithms (like BFS or Greedy Best-First Search). 
However, the agent behaves *greedy-like* it always tries to move closer to the goal using 
rule-based logic, selecting safe directions that reduce distance. 
If blocked, it replans or explores new paths.

This makes the agent a goal-based type it, has a goal, memory, and partial planning, 
but no formal search mechanism yet. In the next stage, this rule-driven logic will be replaced 
with true search-based planning.
"""
# =====================================================================================================

class GoalBasedNavigationAgent(Agent):
    # Added: plan_toward_goal() and explore() are new methods (custom BFS-like search logic)
    # Added: uses mental_map to store partially observable world knowledge.
    """Goal-based navigation agent using simple greedy-like movement rules."""
    
    def __init__(self, width=10, height=10):
        # Use explicit program argument to override AIMA’s default input() behaviour
        super().__init__(program=self.goal_based_program)
        self.width = width
        self.height = height
        self.mental_map = {}
        self.visited = set()
        self.path_plan = []
        self.stuck_counter = 0

    def goal_based_program(self, percept):
        """Main decision-making for the goal-based agent."""
        location = percept['location']
        goal = percept['goal']
        
        # Update internal state with white cane data
        x, y = location
        self.visited.add(location)
        surroundings = {
            (x, y-1): percept['front'],
            (x, y+1): percept['back'],
            (x-1, y): percept['left'],
            (x+1, y): percept['right']
        }
        for pos, tile in surroundings.items():
            if 0 <= pos[0] < self.width and 0 <= pos[1] < self.height:
                self.mental_map[pos] = tile

        # Stop if goal reached
        if location == goal:
            print(f"GOAL REACHED! Final location: {location}")
            return 'NoOp'
        
        # Rule-based greedy-like plan toward goal
        if not self.path_plan or self.stuck_counter > 3:
            self.path_plan = self.plan_toward_goal(location, goal)
            self.stuck_counter = 0
        
        # Execute next step from plan
        if self.path_plan:
            next_step = self.path_plan[0]
            dx = next_step[0] - x
            dy = next_step[1] - y

            if self.mental_map.get(next_step, 0) in [0, 4]:  # floor or goal
                self.path_plan.pop(0)
                if dx == 1:
                    return 'Right'
                elif dx == -1:
                    return 'Left'
                elif dy == 1:
                    return 'Back'
                elif dy == -1:
                    return 'Forward'
            else:
                self.path_plan = []
                self.stuck_counter += 1

        return self.explore(percept)

    def plan_toward_goal(self, start, goal):
        """Simplified greedy-like rule-based path planner."""
        from collections import deque
        queue = deque([(start, [])])
        visited = {start}

        while queue:
            current, path = queue.popleft()
            if current == goal:
                return path
            
            # Explore in 4 directions
            for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                next_pos = (current[0] + dx, current[1] + dy)
                if (0 <= next_pos[0] < self.width and 
                    0 <= next_pos[1] < self.height and 
                    next_pos not in visited):
                    
                    tile = self.mental_map.get(next_pos, 0)
                    if tile in [0, 4]:
                        visited.add(next_pos)
                        queue.append((next_pos, path + [next_pos]))
        return []

    def explore(self, percept):
        """Explore nearby safe and unvisited tiles."""
        x, y = percept['location']
        options = []

        # Prefer unvisited safe directions
        if percept['front'] == 0 and (x, y-1) not in self.visited:
            options.append('Forward')
        if percept['right'] == 0 and (x+1, y) not in self.visited:
            options.append('Right')
        if percept['left'] == 0 and (x-1, y) not in self.visited:
            options.append('Left')
        if percept['back'] == 0 and (x, y+1) not in self.visited:
            options.append('Back')

        # Random choice among options
        if options:
            return random.choice(options)

        # Safe random fallback
        safe_moves = []
        if percept['front'] == 0:
            safe_moves.append('Forward')
        if percept['right'] == 0:
            safe_moves.append('Right')
        if percept['left'] == 0:
            safe_moves.append('Left')
        if percept['back'] == 0:
            safe_moves.append('Back')
        
        return random.choice(safe_moves) if safe_moves else 'NoOp'


# =====================================================================================================
""" D. Agent Comparison """
# =====================================================================================================
"""
Now I wanted to see how each agent performs in the same environment.
The metrics are simple: average performance score, number of steps, and success rate.
This helps visualize how memory and goal-awareness affect efficiency.
"""
# =====================================================================================================

def env_factory():
    # Custom helper for consistent world creation
    return NavigationEnvironment(width=10, height=10, obstacle_prob=0.1)

def compare_agents(agent_factories, n=10, steps=120):
    # Custom: calculates success, performance, and average steps
    # Not part of AIMA.
    """Run each agent type multiple times and record average results."""
    results = []
    for make_agent in agent_factories:
        total_perf = total_steps = successes = 0
        for _ in range(n):
            env = env_factory()
            agent = make_agent()
            agent.location = (1, 1)
            agent.performance = 0
            env.add_thing(agent, agent.location)

            step = 0
            while not env.is_done() and step < steps:
                p = env.percept(agent)
                a = agent.program(p)
                env.execute_action(agent, a)
                step += 1

            total_perf += agent.performance
            total_steps += step
            if env.is_done():
                successes += 1

        results.append((make_agent().__class__.__name__,
                        total_perf / n, total_steps / n,
                        100 * successes / n))
    return results


def visualize(results):
    """Custom visualization using matplotlib to compare performance."""
    # Not from AIMA. Simple bar plots for results comparison.
    """Simple bar plot to compare average performance and success."""
    names = [r[0] for r in results]
    perf = [r[1] for r in results]
    succ = [r[3] for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.bar(names, perf)
    ax1.set_title("Average Performance")
    ax2.bar(names, succ)
    ax2.set_title("Success Rate (%)")
    plt.tight_layout()
    plt.show()


# =====================================================================================================
""" E. Main Test """
# Custom: runs tests for each agent + scalability tests.
# run_all(), scalability_demo(), question_1_1() are 100% project-specific.
# =====================================================================================================
"""
At the end, I tested all three agents in the same setup.
This final run shows their relative strengths and weaknesses.
"""
# =====================================================================================================
def run_all():
    agent_factories = [
        lambda: SimpleReflexNavigationAgent(),
        lambda: ModelBasedNavigationAgent(),
        lambda: GoalBasedNavigationAgent()
    ]

    print("\n=== Running Navigation Agent Comparison ===")
    print("Each agent runs 10 trials for up to 120 steps.\n")
    results = compare_agents(agent_factories, n=10, steps=120)

    print(f"{'Agent Type':<30} | {'Avg Perf':>9} | {'Avg Steps':>9} | {'Success %':>9}")
    print("-" * 70)
    for name, avg_perf, avg_steps, success in results:
        print(f"{name:<30} | {avg_perf:>9.2f} | {avg_steps:>9.2f} | {success:>9.1f}")

    visualize(results)

# =====================================================================================================
# SCALABILITY TEST — How agents perform in larger/smaller worlds
# =====================================================================================================
def scalability_demo(sizes=(6, 10, 14), trials=5, step_factor=10):
    # Custom scalability test — evaluates each agent’s performance in larger worlds.
    """
    Test how agents scale as the grid size grows.
    No plotting here (pure console) to avoid blocking.
    """
    agent_classes = [SimpleReflexNavigationAgent, ModelBasedNavigationAgent, GoalBasedNavigationAgent]
    print("\n=== Scalability across world sizes ===")

    for sz in sizes:
        print(f"\nWorld Size: {sz}x{sz}")
        print(f"{'Agent':<30} | {'AvgPerf':>8} | {'Success%':>8}")
        print("-" * 56)

        max_steps = sz * step_factor

        for A in agent_classes:
            total_perf = 0
            successes = 0

            for _ in range(trials):
                env = NavigationEnvironment(width=sz, height=sz, obstacle_prob=0.1)
                try:
                    agent = A(width=sz, height=sz)
                except TypeError:
                    agent = A()

                start = (1, 1)
                agent.location = start
                agent.performance = 0
                env.add_thing(agent, start)

                steps = 0
                while not env.is_done() and steps < max_steps:
                    percept = env.percept(agent)
                    action = agent.program(percept)
                    if action == 'NoOp':
                        break
                    env.execute_action(agent, action)
                    steps += 1

                total_perf += agent.performance
                if env.is_done():
                    successes += 1

            avg_perf = total_perf / trials
            success_pct = 100.0 * successes / trials
            print(f"{A.__name__:<30} | {avg_perf:>8.1f} | {success_pct:>7.1f}%")


# =====================================================================================================
# PART 1.1: BUILDING YOUR WORLD
# =====================================================================================================
def run_1_1():
    """
    Runs all main tests and demonstrations for Part 1.1.
    Includes: environment setup, agent initialization, and goal-based navigation test.
    """
    print("Running environment and agent tests...\n")

    # List of agents to demonstrate
    agents = [
        ("Simple Reflex Agent", SimpleReflexNavigationAgent),
        ("Model-Based Agent", ModelBasedNavigationAgent),
        ("Goal-Based Agent", GoalBasedNavigationAgent)
    ]

    # Test each agent individually
    for name, AgentType in agents:
        print(f"\n--- Testing {name} ---")
        env = NavigationEnvironment(width=10, height=10)

        # Some agents accept (width, height), others don't — handle both
        try:
            agent = AgentType(width=10, height=10)
        except TypeError:
            agent = AgentType()

        env.add_thing(agent, (1, 1))  # Starting position

        step_count = 0
        while not env.is_done() and step_count < 200:
            percept = env.percept(agent)
            action = agent.program(percept)
            if action == 'NoOp':
                break
            env.execute_action(agent, action)
            step_count += 1

        print(f"{name} finished after {step_count} steps.\n")

    # Compare all agent types (quantitative performance)
    print("Now comparing all agents and visualizing results...\n")
    run_all()

    # Scalability analysis for different world sizes
    scalability_demo()

def question_1_1():
    """Part 1.1: Agent-based world implementation"""
    print("=== PART 1.1: Agent-Based World ===\n")
    run_1_1()  # runs Part 1.1 world simulation


# =====================================================================================================
                             # PART 1.2 — SEARCHING YOUR WORLD
# =====================================================================================================
"""
This section connects the navigation environment to a formal search problem.
It is based on AIMA’s Problem class (search.py) but customized for the navigation world.
"""
# =====================================================================================================
# Problem Formulation
# =====================================================================================================

class NavigationSearchProblem(Problem):
    """
    Custom search problem for the visually impaired navigation world.
    Extends AIMA’s Problem class and uses NavigationEnvironment’s grid.
    """

    def __init__(self, env, initial_state, goal_state):
        # --- based on AIMA structure but adapted for grid navigation
        super().__init__(initial_state, goal_state)
        self.env = env
        self.width = env.width
        self.height = env.height
        self.grid = env.grid

    # -------------------------------------------------------------------------------------------------
    def actions(self, state):
        """
        Returns possible movement actions from the current location.
        AIMA pattern: list of valid actions at given state.
        """
        # --- Custom logic adapted from your environment (walls, obstacles, stairs)
        x, y = state
        possible = []
        moves = {
            'Up': (x, y - 1),
            'Down': (x, y + 1),
            'Left': (x - 1, y),
            'Right': (x + 1, y)
        }

        for act, (nx, ny) in moves.items():
            if 0 <= nx < self.width and 0 <= ny < self.height:
                tile = self.grid[ny, nx]
                # Only walkable or goal tiles are valid
                if tile in (FLOOR, GOAL):
                    possible.append(act)
        return possible

    # -------------------------------------------------------------------------------------------------
    def result(self, state, action):
        """
        Returns the new state after taking an action.
        AIMA equivalent: transition model.
        """
        x, y = state
        if action == 'Up':
            y -= 1
        elif action == 'Down':
            y += 1
        elif action == 'Left':
            x -= 1
        elif action == 'Right':
            x += 1
        return (x, y)

    # -------------------------------------------------------------------------------------------------
    def goal_test(self, state):
        """
        AIMA equivalent: returns True if goal is reached.
        """
        return state == self.goal

    # -------------------------------------------------------------------------------------------------
    def path_cost(self, c, state1, action, state2):
        """
        Custom cost model for this world.
        - Step = +1 cost
        - Hitting walls, stairs, or obstacles = penalty
        """
        x, y = state2
        tile = self.grid[y, x]

        # --- Custom penalty system integrated with AIMA’s path_cost pattern
        if tile == OBSTACLE:
            return c + 10    # bump into furniture
        elif tile == STAIRS:
            return c + 100   # dangerous zone
        elif tile == WALL:
            return c + 999   # invalid (never chosen normally)
        else:
            return c + 1     # normal movement


# =====================================================================================================
# CREATING SEARCHABLE PROBLEM (Dynamic world size)
# =====================================================================================================

def create_search_problem(world_size=12, obstacle_prob=0.1):
    """
    Converts the navigation environment into an AIMA-compatible search problem.
    'world_size' lets you change the grid without affecting Part 1 code.
    """
    env = NavigationEnvironment(width=world_size, height=world_size, obstacle_prob=obstacle_prob)
    initial_state, goal_state = env.get_random_start_and_goal()
    problem = NavigationSearchProblem(env, initial_state, goal_state)

    print("\n[Created Search Problem]")
    print(f"World: {world_size}x{world_size} | Initial: {initial_state} | Goal: {goal_state}")
    return problem



# =====================================================================================================
# UNINFORMED SEARCH TECHNIQUES
# =====================================================================================================
"""
Implements BFS, UCS, and IDS for the navigation world.
All algorithms are from AIMA’s search.py module.
The environment is connected through NavigationSearchProblem.
Each algorithm is compared experimentally based on performance measures.
"""
# =====================================================================================================

def run_uninformed_searches(problem):
    """
    Runs BFS, UCS, and IDS using AIMA’s built-in implementations.
    """

    search_algorithms = [
        ("Breadth-First Search", breadth_first_graph_search),
        ("Uniform-Cost Search", uniform_cost_search),
        ("Iterative Deepening Search", iterative_deepening_search)
    ]

    results = []

    for name, algorithm in search_algorithms:
        start_time = time.time()
        node = algorithm(problem)
        duration = time.time() - start_time

        if node:
            path = node.solution()
            cost = node.path_cost
            found = True
        else:
            path = []
            cost = float('inf')
            found = False

        results.append({
            'algorithm': name,
            'found': found,
            'path_length': len(path),
            'cost': cost,
            'time': round(duration, 4)
        })

    return results


def print_uninformed_results(results):
    """
    Prints results of uninformed search algorithms (experimental comparison only).
    """
    print("\n=== Uninformed Search Comparison Results ===\n")
    print(f"{'Algorithm':<30} | {'Found':<6} | {'Path Len':<9} | {'Cost':<6} | {'Time(s)':<8}")
    print("-" * 70)
    for r in results:
        print(f"{r['algorithm']:<30} | "
              f"{'Yes' if r['found'] else 'No ':<6} | "
              f"{r['path_length']:<9} | "
              f"{r['cost']:<6} | "
              f"{r['time']:<8}")



def question_1_2():
    """
    Part 1.2: Search techniques implementation
    """
    print("\n\n\n=== PART 1.2: SEARCHING YOUR WORLD ===\n")
    print("--- Uninformed Search Techniques (BFS, UCS, IDS) ---\n")

    # --- Create the searchable problem (choose size dynamically)
    problem = create_search_problem(world_size=10, obstacle_prob=0.07)

    # --- Run all uninformed searches
    results = run_uninformed_searches(problem)

    # --- Print summarized results
    print_uninformed_results(results)
# =====================================================================================================
# MAIN EXECUTION
# =====================================================================================================



if __name__ == '__main__':
    #question_1_1()
    question_1_2()

