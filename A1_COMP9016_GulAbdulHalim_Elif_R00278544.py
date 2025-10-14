import sys
import os
import random

parent_dir = os.path.dirname(os.getcwd())
sys.path.append(parent_dir)

from agents import *

# ============================================
# TILE TYPES (AIMA Thing class)
# ============================================
class Floor(Thing):
    """Walkable floor tile,accessible"""
    pass

class Stairs(Thing):
    """Stairs tile, inaccessible, randomly placed"""
    pass

class Wall(Thing):
    """Wall tile,  inaccessible"""
    pass

class Obstacle(Thing):
    """Obstacle (furniture or other room objects), inaccessible, randomly placed"""
    pass

class Goal(Thing):
    """Goal tile"""
    pass


# ============================================
# ENVIRONMENT
# Source: Extended from AIMA agents.py XYEnvironment class
# ============================================
class VisuallyImpairedEnvironment(XYEnvironment):
    """
    Grid-based environment for visually impaired navigation.
    Extends AIMA XYEnvironment for 2D grid representation.
    
    Reference: Russell & Norvig AIMA - agents.py
    Custom implementation: Tile placement logic
    """
    
    def __init__(self, width=5, height=5):
        # AIMA XYEnvironment initialization
        super().__init__(width, height)
        self.width = width
        self.height = height
        self.goal_location = (width-1, height-1)
        self.setup_environment()
    
    def setup_environment(self):
        """
        Place tiles in environment.
        Custom implementation for domain-specific tile placement.
        Uses AIMA add_thing() method from Environment class.
        """
        # Goal placement
        self.add_thing(Goal(), self.goal_location)  # AIMA method
        
        # Stairs placement (custom logic)
        num_stairs = max(1, self.width // 3)
        for _ in range(num_stairs):
            x = random.randint(1, self.width-2)  # Python random
            y = random.randint(1, self.height-2)
            if (x, y) != self.goal_location and (x, y) != (0, 0):
                self.add_thing(Stairs(), (x, y))  # AIMA method
        
        # Wall boundaries (custom logic)
        for x in range(self.width):
            self.add_thing(Wall(), (x, 0))  # AIMA method
            self.add_thing(Wall(), (x, self.height-1))
        for y in range(self.height):
            self.add_thing(Wall(), (0, y))
            self.add_thing(Wall(), (self.width-1, y))
        
        # Obstacle placement (custom logic)
        num_obstacles = max(1, self.width // 4)
        for _ in range(num_obstacles):
            x = random.randint(1, self.width-2)
            y = random.randint(1, self.height-2)
            if (x, y) != self.goal_location and (x, y) != (0, 0):
                self.add_thing(Obstacle(), (x, y))  # AIMA method
    def percept(self, agent):
        """
        Return agent's perception of environment.
        
        Source: AIMA agents.py Environment.percept() pattern
        Custom implementation: White cane sensing (4 adjacent cells)
        
        Returns:
            tuple: (location, adjacent_tiles)
        """
        location = agent.location
        adjacent = self.get_adjacent_tiles(location)
        return (location, adjacent)
    
    def get_adjacent_tiles(self, location):
        """
        Get tile types in 4 adjacent cells (white cane sweep).
        Custom implementation for tactile sensing.
        
        Returns:
            dict: {'Up': tile_type, 'Down': tile_type, ...}
        """
        x, y = location
        adjacent = {}
        
        directions = {
            'Up': (x, y-1),
            'Down': (x, y+1),
            'Left': (x-1, y),
            'Right': (x+1, y)
        }
        
        for dir_name, (nx, ny) in directions.items():
            if 0 <= nx < self.width and 0 <= ny < self.height:
                tile_type = self.get_tile_type((nx, ny))
                adjacent[dir_name] = tile_type
            else:
                adjacent[dir_name] = 'Wall'  # Out of bounds
        
        return adjacent
    
    def get_tile_type(self, location):
        """
        Identify tile type at location.
        Uses AIMA things list to check tile types.
        
        Returns:
            str: 'Stairs', 'Wall', 'Obstacle', 'Goal', or 'Floor'
        """
        for thing in self.things:
            if thing.location == location:
                if isinstance(thing, Stairs):
                    return 'Stairs'
                elif isinstance(thing, Wall):
                    return 'Wall'
                elif isinstance(thing, Obstacle):
                    return 'Obstacle'
                elif isinstance(thing, Goal):
                    return 'Goal'
        return 'Floor'  # Empty cell = floor
    
    def execute_action(self, agent, action):
        """
        Execute agent action with white cane safety system.
        
        Source: AIMA agents.py Environment.execute_action() pattern
        Custom implementation: White cane obstacle detection + vibration
        
        Performance updates:
        - Floor move: -1
        - Obstacle detected: -10 (vibration warning, blocked)
        - Goal reached: +1000
        """
        if action == 'NoOp':
            return
        
        # Calculate target location
        x, y = agent.location
        moves = {
            'Up': (x, y-1),
            'Down': (x, y+1),
            'Left': (x-1, y),
            'Right': (x+1, y)
        }
        
        target = moves.get(action)
        if not target:
            return
        
        # Check if target is valid
        if not self.is_valid_location(target):
            agent.performance -= 10  # Out of bounds warning
            return
        
        # White cane detection
        tile_type = self.get_tile_type(target)
        
        if tile_type in ['Stairs', 'Wall', 'Obstacle']:
            # White cane vibration warning - movement blocked
            agent.performance -= 10
            return  # Don't move
        
        # Safe to move
        agent.location = target
        
        if tile_type == 'Goal':
            agent.performance += 1000  # Goal reached!
        else:  # Floor
            agent.performance -= 1  # Movement cost
    
    def is_valid_location(self, location):
        """Check if location within grid bounds"""
        x, y = location
        return 0 <= x < self.width and 0 <= y < self.height


# ============================================
# AGENT 1: SIMPLE REFLEX
# Source: AIMA agents.py Agent class pattern
# Custom implementation: Rule-based decision making
# ============================================
class SimpleReflexNavigationAgent(Agent):
    """
    Simple reflex agent using condition-action rules.
    No memory, purely reactive.
    
    Reference: AIMA Chapter 2 - Simple Reflex Agent
    """
    
    def __init__(self):
        super().__init__()
        self.program = self.simple_reflex_program
    
    def simple_reflex_program(self, percept):
        """
        Rule-based decision making.
        Custom implementation: White cane-based navigation rules.
        
        Rules:
        1. If goal adjacent → move to goal
        2. If obstacle detected → avoid
        3. Else → choose random safe direction
        """
        location, adjacent = percept
        
        # Rule 1: Goal visible?
        for direction, tile in adjacent.items():
            if tile == 'Goal':
                return direction
        
        # Rule 2: Find safe directions (avoid obstacles)
        safe_directions = []
        for direction, tile in adjacent.items():
            if tile == 'Floor':
                safe_directions.append(direction)
        
        # Rule 3: Move to safe direction
        if safe_directions:
            return random.choice(safe_directions)  # Python random
        
        return 'NoOp'  # Stuck

# ============================================
# PART 1.1: BUILDING YOUR WORLD
# ============================================
def question_1_1():
    """
    Part 1.1: Agent-based world implementation
    """
    print("=== PART 1.1: Agent-Based World ===")
    # Buraya kod gelecek
    
    # Create environment
    env = VisuallyImpairedEnvironment(width=5, height=5)
    print(f"✓ Environment: {env.width}x{env.height}")
    print(f"✓ Goal: {env.goal_location}\n")
    
    # Create Simple Reflex Agent
    agent = SimpleReflexNavigationAgent()
    agent.location = (1, 1)
    agent.performance = 0
    env.add_thing(agent, agent.location)
    
    print("--- Simple Reflex Agent Test ---")
    print(f"Start: {agent.location}, Performance: {agent.performance}")
    
    # Run 10 steps
    for step in range(10):
        percept = env.percept(agent)
        action = agent.program(percept)
        env.execute_action(agent, action)
        
        print(f"Step {step+1}: {action} → Location: {agent.location}, Performance: {agent.performance}")
        
        if percept[1].get(action) == 'Goal':
            print("✓ GOAL REACHED!")
            break

    pass

# ============================================
# PART 1.2: SEARCHING YOUR WORLD
# ============================================
def question_1_2():
    """
    Part 1.2: Search techniques implementation
    """
    print("=== PART 1.2: Search Techniques ===")
    # Buraya kod gelecek
    pass

# ============================================
# MAIN EXECUTION
# ============================================
if __name__ == '__main__':
    question_1_1()
    question_1_2()
