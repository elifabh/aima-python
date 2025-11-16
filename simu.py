import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, FancyArrowPatch
import matplotlib.lines as mlines

# ===== RENK PALETLERI =====
COLORS = {
    'start': '#4CAF50',
    'process': '#2196F3',
    'decision': '#FF9800',
    'end': '#D32F2F',
    'data': '#9C27B0',
}

def add_box(ax, x, y, width, height, text, box_type='process', fontsize=10):
    """Box ekle"""
    colors = {
        'start': '#C8E6C9',
        'process': '#BBDEFB',
        'decision': '#FFE0B2',
        'end': '#FFCDD2',
        'data': '#E1BEE7',
    }
    edge_colors = {
        'start': '#388E3C',
        'process': '#1976D2',
        'decision': '#F57C00',
        'end': '#D32F2F',
        'data': '#7B1FA2',
    }
    
    if box_type == 'decision':
        # Elmas şekli
        diamond = mpatches.Polygon(
            [[x, y + height/2], [x + width/2, y], 
             [x + width, y + height/2], [x + width/2, y + height]],
            facecolor=colors[box_type],
            edgecolor=edge_colors[box_type],
            linewidth=2.5
        )
        ax.add_patch(diamond)
        ax.text(x + width/2, y + height/2, text, fontsize=fontsize, ha='center', va='center',
               fontweight='bold')
    else:
        # Dikdörtgen
        rect = FancyBboxPatch((x, y), width, height,
                             boxstyle="round,pad=0.1",
                             facecolor=colors[box_type],
                             edgecolor=edge_colors[box_type],
                             linewidth=2.5)
        ax.add_patch(rect)
        ax.text(x + width/2, y + height/2, text, fontsize=fontsize, ha='center', va='center',
               fontweight='bold', wrap=True)

def add_arrow(ax, x1, y1, x2, y2, label='', color='#333333'):
    """Add arrow"""
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                           arrowstyle='->',
                           mutation_scale=25,
                           linewidth=2.5,
                           color=color)
    ax.add_patch(arrow)
    if label:
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mid_x + 0.3, mid_y, label, fontsize=9, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

# =====================================================
# 1. BFS - BREADTH FIRST SEARCH FLOWCHART
# =====================================================

def draw_bfs_flowchart():
    """BFS flowchart"""
    fig, ax = plt.subplots(figsize=(12, 16))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 20)
    ax.axis('off')
    
    # Title
    ax.text(5, 19.5, 'BFS: BREADTH-FIRST SEARCH', fontsize=14, fontweight='bold', ha='center',
           bbox=dict(boxstyle='round,pad=0.7', facecolor='#E3F2FD', edgecolor='#1976D2', linewidth=2.5))
    
    # START
    add_box(ax, 3.5, 18, 3, 0.8, 'START', 'start')
    add_arrow(ax, 5, 18, 5, 17.2)
    
    # Queue initialize
    add_box(ax, 2.5, 16.2, 5, 1, 'Queue ← [Start]\nVisited ← {Start}', 'data')
    add_arrow(ax, 5, 16.2, 5, 15.4)
    
    # Queue empty?
    add_box(ax, 2.5, 14.4, 5, 1, 'Queue empty?', 'decision')
    add_arrow(ax, 7.5, 14.9, 8.5, 14.9, 'YES', '#D32F2F')
    add_arrow(ax, 5, 14.4, 5, 13.6, 'NO', '#4CAF50')
    
    # End - No path
    add_box(ax, 8.2, 14.5, 1.5, 0.8, 'No Path', 'end', fontsize=9)
    
    # Current ← Queue.pop()
    add_box(ax, 2.5, 12.6, 5, 1, 'Current ← Queue.pop()\n(FIFO)', 'process')
    add_arrow(ax, 5, 12.6, 5, 11.8)
    
    # Goal check
    add_box(ax, 2.5, 10.8, 5, 1, 'Current == Goal?', 'decision')
    add_arrow(ax, 7.5, 11.3, 8.5, 11.3, 'YES', '#D32F2F')
    add_arrow(ax, 5, 10.8, 5, 10, 'NO', '#4CAF50')
    
    # End - Path found
    add_box(ax, 8.2, 10.9, 1.5, 0.8, 'Path\nFOUND', 'start', fontsize=9)
    
    # Explore neighbors - 4 directions
    add_box(ax, 2.5, 9, 5, 1, 'Explore 4 directions:\n(0,-1), (0,1), (-1,0), (1,0)', 'data')
    add_arrow(ax, 5, 9, 5, 8.2)
    
    # For each neighbor
    add_box(ax, 2.5, 7.2, 5, 1, 'FOR each Neighbor', 'process')
    add_arrow(ax, 5, 7.2, 5, 6.4)
    
    # Visited check
    add_box(ax, 2.5, 5.4, 5, 1, 'Already\nvisited?', 'decision')
    add_arrow(ax, 7.5, 5.9, 8.5, 5.9, 'YES', '#D32F2F')
    add_arrow(ax, 5, 5.4, 5, 4.6, 'NO', '#4CAF50')
    
    # Skip
    add_box(ax, 8.2, 5.5, 1.5, 0.8, 'SKIP', 'end', fontsize=9)
    
    # Add
    add_box(ax, 2.5, 3.6, 5, 1, 'Queue.append(Neighbor)\nVisited.add(Neighbor)', 'process')
    add_arrow(ax, 5, 3.6, 5, 2.8)
    
    # Loop back
    add_box(ax, 1.5, 1.8, 2, 1, 'END FOR', 'data')
    add_arrow(ax, 1.5, 2.3, 0.5, 2.3)
    add_arrow(ax, 0.5, 2.3, 0.5, 7.7)
    add_arrow(ax, 0.5, 7.7, 2.5, 7.7)
    
    ax.text(0.2, 5, '↻ Loop', fontsize=10, fontweight='bold', color='#D32F2F',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFEBEE'))
    
    # Main loop back
    add_arrow(ax, 1.5, 1.8, 1.5, 14.9)
    add_arrow(ax, 1.5, 14.9, 2.5, 14.9)
    
    ax.text(0.8, 10, '↻ Main\nLoop', fontsize=9, fontweight='bold', color='#D32F2F',
           bbox=dict(boxstyle='round,pad=0.2', facecolor='#FFEBEE'))
    
    # Description
    ax.text(5, 0.5, '📋 FIFO Queue: Layer by layer expansion | Optimal: YES | Memory: High', 
           fontsize=9, ha='center', style='italic',
           bbox=dict(boxstyle='round,pad=0.4', facecolor='#F5F5F5'))
    
    plt.tight_layout()
    plt.show()


# =====================================================
# 2. UCS - UNIFORM COST SEARCH FLOWCHART
# =====================================================

def draw_ucs_flowchart():
    """UCS flowchart"""
    fig, ax = plt.subplots(figsize=(12, 16))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 20)
    ax.axis('off')
    
    ax.text(5, 19.5, 'UCS: UNIFORM-COST SEARCH', fontsize=14, fontweight='bold', ha='center',
           bbox=dict(boxstyle='round,pad=0.7', facecolor='#E8F5E9', edgecolor='#388E3C', linewidth=2.5))
    
    # START
    add_box(ax, 3.5, 18, 3, 0.8, 'START', 'start')
    add_arrow(ax, 5, 18, 5, 17.2)
    
    # Priority Queue initialize
    add_box(ax, 1.5, 16.2, 7, 1, 'PriorityQueue ← [(0, Start)]\nVisited ← {}  |  Cost ← {Start: 0}', 'data')
    add_arrow(ax, 5, 16.2, 5, 15.4)
    
    # Queue empty?
    add_box(ax, 2.5, 14.4, 5, 1, 'PriorityQueue empty?', 'decision')
    add_arrow(ax, 7.5, 14.9, 8.5, 14.9, 'YES', '#D32F2F')
    add_arrow(ax, 5, 14.4, 5, 13.6, 'NO', '#4CAF50')
    
    # End
    add_box(ax, 8.2, 14.5, 1.5, 0.8, 'No Path', 'end', fontsize=9)
    
    # Pop minimum cost node
    add_box(ax, 1.5, 12.6, 7, 1, 'Current ← PQ.pop()\n(LOWEST COST)', 'process')
    add_arrow(ax, 5, 12.6, 5, 11.8)
    
    # Already visited?
    add_box(ax, 2.5, 10.8, 5, 1, 'Already\nvisited?', 'decision')
    add_arrow(ax, 7.5, 11.3, 8.5, 11.3, 'YES', '#D32F2F')
    add_arrow(ax, 5, 10.8, 5, 10, 'NO', '#4CAF50')
    
    # Skip
    add_box(ax, 8.2, 10.9, 1.5, 0.8, 'SKIP', 'end', fontsize=9)
    
    # Mark visited
    add_box(ax, 2.5, 9, 5, 1, 'Visited.add(Current)', 'process')
    add_arrow(ax, 5, 9, 5, 8.2)
    
    # Goal check
    add_box(ax, 2.5, 7.2, 5, 1, 'Current == Goal?', 'decision')
    add_arrow(ax, 7.5, 7.7, 8.5, 7.7, 'YES', '#D32F2F')
    add_arrow(ax, 5, 7.2, 5, 6.4, 'NO', '#4CAF50')
    
    # End
    add_box(ax, 8.2, 7.3, 1.5, 0.8, 'MIN-COST\nPath FOUND', 'start', fontsize=8)
    
    # Explore neighbors
    add_box(ax, 2.5, 5.4, 5, 1, 'FOR each Neighbor\n4 directions', 'data')
    add_arrow(ax, 5, 5.4, 5, 4.6)
    
    # Calculate cost
    add_box(ax, 1.5, 3.6, 7, 1, 'NewCost = g(Current) + GetCost(Neighbor)\nObstacle: +100 | Stairs: +100', 'process')
    add_arrow(ax, 5, 3.6, 5, 2.8)
    
    # Better cost?
    add_box(ax, 2.5, 1.8, 5, 1, 'NewCost <\nCost[Neighbor]?', 'decision')
    add_arrow(ax, 7.5, 2.3, 8.5, 2.3, 'NO', '#D32F2F')
    add_arrow(ax, 5, 1.8, 5, 0.8, 'YES', '#4CAF50')
    
    # Skip
    add_box(ax, 8.2, 1.9, 1.5, 0.8, 'SKIP', 'end', fontsize=9)
    
    # Add to PQ
    add_box(ax, 1.5, -0.2, 7, 0.9, 'PQ.add((NewCost, Neighbor))\nCost[Neighbor] = NewCost', 'process')
    
    # Loop
    add_arrow(ax, 1.5, -0.2, 0.5, -0.2)
    add_arrow(ax, 0.5, -0.2, 0.5, 5.9)
    add_arrow(ax, 0.5, 5.9, 2.5, 5.9)
    
    ax.text(0.2, 3, '↻ Neighbor\nLoop', fontsize=9, fontweight='bold', color='#D32F2F',
           bbox=dict(boxstyle='round,pad=0.2', facecolor='#FFEBEE'))
    
    # Main loop back
    add_arrow(ax, 1.5, -0.2, 1.5, 14.9)
    add_arrow(ax, 1.5, 14.9, 2.5, 14.9)
    
    # Description
    ax.text(5, -1.5, '📋 PriorityQueue (g(n)): Sort by cost | Optimal: YES | Memory: High', 
           fontsize=9, ha='center', style='italic',
           bbox=dict(boxstyle='round,pad=0.4', facecolor='#F5F5F5'))
    
    plt.tight_layout()
    plt.show()


# =====================================================
# 3. IDS - ITERATIVE DEEPENING SEARCH FLOWCHART
# =====================================================

def draw_ids_flowchart():
    """IDS flowchart"""
    fig, ax = plt.subplots(figsize=(12, 18))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 22)
    ax.axis('off')
    
    ax.text(5, 21.5, 'IDS: ITERATIVE DEEPENING SEARCH', fontsize=14, fontweight='bold', ha='center',
           bbox=dict(boxstyle='round,pad=0.7', facecolor='#FFF3E0', edgecolor='#E65100', linewidth=2.5))
    
    # START
    add_box(ax, 3.5, 20.2, 3, 0.8, 'START', 'start')
    add_arrow(ax, 5, 20.2, 5, 19.4)
    
    # Depth limit = 0
    add_box(ax, 2.5, 18.4, 5, 1, 'Depth_Limit ← 0', 'data')
    add_arrow(ax, 5, 18.4, 5, 17.6)
    
    # DFS with limit start
    add_box(ax, 2.5, 16.6, 5, 1, 'DFS(Start, Depth_Limit)', 'process')
    add_arrow(ax, 5, 16.6, 5, 15.8)
    
    # DFS BOX
    ax.text(1.2, 15.3, 'DFS-with-Limit(node, limit):', fontsize=10, fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='#E1BEE7'))
    
    # DFS start
    add_box(ax, 2.5, 14.3, 5, 1, 'Stack ← [node]\nVisited ← {}', 'data', fontsize=9)
    add_arrow(ax, 5, 14.3, 5, 13.5)
    
    # Stack empty?
    add_box(ax, 2.5, 12.5, 5, 1, 'Stack empty?', 'decision')
    add_arrow(ax, 7.5, 13, 8.5, 13, 'YES', '#D32F2F')
    add_arrow(ax, 5, 12.5, 5, 11.7, 'NO', '#4CAF50')
    
    # Return False
    add_box(ax, 8.2, 12.6, 1.5, 0.8, 'Return\nFALSE', 'end', fontsize=9)
    
    # Pop node
    add_box(ax, 2.5, 10.7, 5, 1, 'Current ← Stack.pop()', 'process')
    add_arrow(ax, 5, 10.7, 5, 9.9)
    
    # Depth check
    add_box(ax, 2.5, 8.9, 5, 1, 'Depth(Current)\n> Limit?', 'decision')
    add_arrow(ax, 7.5, 9.4, 8.5, 9.4, 'YES', '#D32F2F')
    add_arrow(ax, 5, 8.9, 5, 8.1, 'NO', '#4CAF50')
    
    # Skip
    add_box(ax, 8.2, 8.5, 1.5, 0.8, 'SKIP', 'end', fontsize=9)
    
    # Goal?
    add_box(ax, 2.5, 7.1, 5, 1, 'Current == Goal?', 'decision')
    add_arrow(ax, 7.5, 7.6, 8.5, 7.6, 'YES', '#D32F2F')
    add_arrow(ax, 5, 7.1, 5, 6.3, 'NO', '#4CAF50')
    
    # Return True
    add_box(ax, 8.2, 7.2, 1.5, 0.8, 'Return\nTRUE', 'start', fontsize=9)
    
    # Add neighbors
    add_box(ax, 2.5, 5.3, 5, 1, 'Stack.append(Neighbors)', 'process')
    add_arrow(ax, 5, 5.3, 5, 4.5)
    
    # DFS loop back
    add_arrow(ax, 5, 4.5, 5, 4)
    add_arrow(ax, 5, 4, 1.5, 4)
    add_arrow(ax, 1.5, 4, 1.5, 12.5)
    add_arrow(ax, 1.5, 12.5, 2.5, 12.5)
    
    ax.text(0.5, 8.2, '↻ DFS\nLoop', fontsize=9, fontweight='bold', color='#D32F2F',
           bbox=dict(boxstyle='round,pad=0.2', facecolor='#FFEBEE'))
    
    # DFS end
    add_arrow(ax, 8.5, 13, 8.5, 3.4)
    add_arrow(ax, 8.5, 3.4, 6.5, 3.4)
    
    # Main algorithm - DFS return check
    add_box(ax, 2, 2.4, 6, 1, 'DFS return True?', 'decision')
    add_arrow(ax, 8, 2.9, 8.8, 2.9, 'YES', '#D32F2F')
    add_arrow(ax, 2, 2.9, 1.2, 2.9, 'NO', '#4CAF50')
    
    # Success
    add_box(ax, 8.5, 2.5, 1.5, 0.8, 'Path\nFOUND', 'start', fontsize=9)
    
    # Increase limit
    add_box(ax, 0.5, 2.5, 1.5, 0.8, 'Depth_Limit\n+= 1', 'data', fontsize=9)
    add_arrow(ax, 0.75, 2.5, 0.75, 1.3)
    add_arrow(ax, 0.75, 1.3, 5, 1.3)
    
    # Limit check
    add_box(ax, 2.5, 0.3, 5, 1, 'Depth_Limit\n< MAX_LIMIT?', 'decision')
    add_arrow(ax, 2.5, 0.8, 1.2, 0.8, 'YES', '#4CAF50')
    add_arrow(ax, 7.5, 0.8, 8.5, 0.8, 'NO', '#D32F2F')
    
    # End failure
    add_box(ax, 8.2, 0.4, 1.5, 0.8, 'No\nPath', 'end', fontsize=9)
    
    ax.text(5, -1.2, '📋 Depth Limit ↑: Low Memory | Optimal: YES | DFS repeatedly', 
           fontsize=9, ha='center', style='italic',
           bbox=dict(boxstyle='round,pad=0.4', facecolor='#F5F5F5'))
    
    plt.tight_layout()
    plt.show()


# =====================================================
# 4. GREEDY BEST-FIRST SEARCH FLOWCHART
# =====================================================

def draw_greedy_flowchart():
    """Greedy flowchart"""
    fig, ax = plt.subplots(figsize=(12, 16))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 20)
    ax.axis('off')
    
    ax.text(5, 19.5, 'GREEDY BEST-FIRST SEARCH', fontsize=14, fontweight='bold', ha='center',
           bbox=dict(boxstyle='round,pad=0.7', facecolor='#F3E5F5', edgecolor='#7B1FA2', linewidth=2.5))
    
    # START
    add_box(ax, 3.5, 18, 3, 0.8, 'START', 'start')
    add_arrow(ax, 5, 18, 5, 17.2)
    
    # PQ initialize (h only)
    add_box(ax, 1.5, 16.2, 7, 1, 'PriorityQueue ← [(h(Start), Start)]\nVisited ← {}', 'data')
    add_arrow(ax, 5, 16.2, 5, 15.4)
    
    # Queue empty?
    add_box(ax, 2.5, 14.4, 5, 1, 'PriorityQueue empty?', 'decision')
    add_arrow(ax, 7.5, 14.9, 8.5, 14.9, 'YES', '#D32F2F')
    add_arrow(ax, 5, 14.4, 5, 13.6, 'NO', '#4CAF50')
    
    # End
    add_box(ax, 8.2, 14.5, 1.5, 0.8, 'No Path', 'end', fontsize=9)
    
    # Pop nearest heuristic node
    add_box(ax, 1.5, 12.6, 7, 1, 'Current ← PQ.pop()\n(ONLY h(n) BASED)', 'process')
    add_arrow(ax, 5, 12.6, 5, 11.8)
    
    # Goal check
    add_box(ax, 2.5, 10.8, 5, 1, 'Current == Goal?', 'decision')
    add_arrow(ax, 7.5, 11.3, 8.5, 11.3, 'YES', '#D32F2F')
    add_arrow(ax, 5, 10.8, 5, 10, 'NO', '#4CAF50')
    
    # End - Path found (but suboptimal!)
    add_box(ax, 8.2, 10.9, 1.5, 0.8, 'Path\n(⚠️ SUB-OPT)', 'end', fontsize=8)
    
    # Explore neighbors
    add_box(ax, 2.5, 9, 5, 1, 'Explore 4 directions:\n(0,-1), (0,1), (-1,0), (1,0)', 'data')
    add_arrow(ax, 5, 9, 5, 8.2)
    
    # For each neighbor
    add_box(ax, 2.5, 7.2, 5, 1, 'FOR each Neighbor', 'process')
    add_arrow(ax, 5, 7.2, 5, 6.4)
    
    # Visited check
    add_box(ax, 2.5, 5.4, 5, 1, 'Already visited?', 'decision')
    add_arrow(ax, 7.5, 5.9, 8.5, 5.9, 'YES', '#D32F2F')
    add_arrow(ax, 5, 5.4, 5, 4.6, 'NO', '#4CAF50')
    
    # Skip
    add_box(ax, 8.2, 5.5, 1.5, 0.8, 'SKIP', 'end', fontsize=9)
    
    # Calculate h(n) - ONLY HEURISTIC!
    add_box(ax, 1.5, 3.6, 7, 1, 'h = Manhattan(Neighbor → Goal)\nf(n) = h(n)  ← NO g(n)!', 'process')
    add_arrow(ax, 5, 3.6, 5, 2.8)
    
    # Add to PQ
    add_box(ax, 2.5, 1.8, 5, 1, 'PQ.add((h, Neighbor))\nVisited.add(Neighbor)', 'process')
    
    # Loop
    add_arrow(ax, 2.5, 2.3, 0.5, 2.3)
    add_arrow(ax, 0.5, 2.3, 0.5, 7.7)
    add_arrow(ax, 0.5, 7.7, 2.5, 7.7)
    
    ax.text(0.2, 5, '↻ Loop', fontsize=10, fontweight='bold', color='#D32F2F',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFEBEE'))
    
    # Main loop back
    add_arrow(ax, 0.5, 2.3, 0.5, 14.9)
    add_arrow(ax, 0.5, 14.9, 2.5, 14.9)
    
    # Description
    ax.text(5, 0.5, '📋 ONLY h(n): Fast but NOT optimal | ⚠️ Suboptimal Path', 
           fontsize=9, ha='center', style='italic',
           bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFEBEE'))
    
    plt.tight_layout()
    plt.show()


# =====================================================
# 5. A* SEARCH FLOWCHART
# =====================================================

def draw_astar_flowchart():
    """A* flowchart"""
    fig, ax = plt.subplots(figsize=(12, 18))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 22)
    ax.axis('off')
    
    ax.text(5, 21.5, 'A*: A-STAR SEARCH', fontsize=14, fontweight='bold', ha='center',
           bbox=dict(boxstyle='round,pad=0.7', facecolor='#F8BBD0', edgecolor='#C2185B', linewidth=2.5))
    
    ax.text(5, 20.7, 'f(n) = g(n) + h(n)', fontsize=11, ha='center', style='italic', fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.4', facecolor='#FCE4EC'))
    
    # START
    add_box(ax, 3.5, 19.5, 3, 0.8, 'START', 'start')
    add_arrow(ax, 5, 19.5, 5, 18.7)
    
    # PQ initialize
    add_box(ax, 1, 17.7, 8, 1, 'PriorityQueue ← [(f(Start)=g+h, Start)]\nVisited ← {}  |  g-values ← {Start: 0}', 'data')
    add_arrow(ax, 5, 17.7, 5, 16.9)
    
    # Queue empty?
    add_box(ax, 2.5, 15.9, 5, 1, 'PriorityQueue empty?', 'decision')
    add_arrow(ax, 7.5, 16.4, 8.5, 16.4, 'YES', '#D32F2F')
    add_arrow(ax, 5, 15.9, 5, 15.1, 'NO', '#4CAF50')
    
    # End
    add_box(ax, 8.2, 15.5, 1.5, 0.8, 'No Path', 'end', fontsize=9)
    
    # Pop minimum f(n)
    add_box(ax, 1.5, 14.1, 7, 1, 'Current ← PQ.pop()\n(MIN f(n) = g + h)', 'process')
    add_arrow(ax, 5, 14.1, 5, 13.3)
    
    # Visited check
    add_box(ax, 2.5, 12.3, 5, 1, 'Already\nvisited?', 'decision')
    add_arrow(ax, 7.5, 12.8, 8.5, 12.8, 'YES', '#D32F2F')
    add_arrow(ax, 5, 12.3, 5, 11.5, 'NO', '#4CAF50')
    
    # Skip
    add_box(ax, 8.2, 12.4, 1.5, 0.8, 'SKIP', 'end', fontsize=9)
    
    # Mark visited
    add_box(ax, 2.5, 10.5, 5, 1, 'Visited.add(Current)', 'process')
    add_arrow(ax, 5, 10.5, 5, 9.7)
    
    # Goal?
    add_box(ax, 2.5, 8.7, 5, 1, 'Current == Goal?', 'decision')
    add_arrow(ax, 7.5, 9.2, 8.5, 9.2, 'YES', '#D32F2F')
    add_arrow(ax, 5, 8.7, 5, 7.9, 'NO', '#4CAF50')
    
    # Success
    add_box(ax, 8.2, 8.8, 1.5, 0.8, 'OPTIMAL\nPath\nFOUND', 'start', fontsize=8)
    
    # Neighbors
    add_box(ax, 2.5, 6.9, 5, 1, 'FOR each Neighbor\n4 directions', 'data')
    add_arrow(ax, 5, 6.9, 5, 6.1)
    
    # Calculate g(n)
    add_box(ax, 1.5, 5.1, 7, 1, 'NewG = g(Current) + Cost(Neighbor)\nNewH = h(Neighbor)  [Manhattan Distance]', 'process')
    add_arrow(ax, 5, 5.1, 5, 4.3)
    
    # Calculate f(n)
    add_box(ax, 2, 3.3, 6, 1, 'f = NewG + NewH\n(TOTAL ESTIMATED COST)', 'data')
    add_arrow(ax, 5, 3.3, 5, 2.5)
    
    # Better g?
    add_box(ax, 2.5, 1.5, 5, 1, 'NewG <\ng(Neighbor)?', 'decision')
    add_arrow(ax, 7.5, 2, 8.5, 2, 'NO', '#D32F2F')
    add_arrow(ax, 5, 1.5, 5, 0.5, 'YES', '#4CAF50')
    
    # Skip
    add_box(ax, 8.2, 1.6, 1.5, 0.8, 'SKIP', 'end', fontsize=9)
    
    # Add
    add_box(ax, 1.5, -0.5, 7, 0.9, 'PQ.add((f, Neighbor))\ng(Neighbor) = NewG', 'process')
    
    # Loop
    add_arrow(ax, 1.5, -0.5, 0.5, -0.5)
    add_arrow(ax, 0.5, -0.5, 0.5, 6.4)
    add_arrow(ax, 0.5, 6.4, 2.5, 6.4)
    
    ax.text(0.1, 3.5, '↻ Neighbor\nLoop', fontsize=9, fontweight='bold', color='#D32F2F',
           bbox=dict(boxstyle='round,pad=0.2', facecolor='#FFEBEE'))
    
    # Main loop back
    add_arrow(ax, 1.5, -0.5, 1.5, 16.4)
    add_arrow(ax, 1.5, 16.4, 2.5, 16.4)
    
    # Description
    ax.text(5, -1.8, '📋 f(n)=g(n)+h(n): Balances cost and heuristic | Optimal: YES | BEST CHOICE!', 
           fontsize=9, ha='center', style='italic',
           bbox=dict(boxstyle='round,pad=0.4', facecolor='#C8E6C9'))
    
    plt.tight_layout()
    plt.show()


# =====================================================
# 6. RBFS SEARCH FLOWCHART
# =====================================================

def draw_rbfs_flowchart():
    """RBFS flowchart"""
    fig, ax = plt.subplots(figsize=(12, 18))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 22)
    ax.axis('off')
    
    ax.text(5, 21.5, 'RBFS: RECURSIVE BEST-FIRST SEARCH', fontsize=14, fontweight='bold', ha='center',
           bbox=dict(boxstyle='round,pad=0.7', facecolor='#B2DFDB', edgecolor='#00897B', linewidth=2.5))
    
    ax.text(5, 20.7, 'Memory-Efficient (Recursive Stack)', fontsize=11, ha='center', style='italic', 
           fontweight='bold', bbox=dict(boxstyle='round,pad=0.4', facecolor='#E0F2F1'))
    
    # START
    add_box(ax, 2.5, 19.5, 5, 0.8, 'START', 'start')
    add_arrow(ax, 5, 19.5, 5, 18.7)
    
    # Call RBFS
    add_box(ax, 1.5, 17.7, 7, 1, 'RBFS(Start, Goal, f_limit=∞)', 'data')
    add_arrow(ax, 5, 17.7, 5, 16.9)
    
    ax.text(1.2, 16.4, 'RBFS-Recursive(node, goal, f_limit):', fontsize=10, fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='#E1BEE7'))
    
    # f value check
    add_box(ax, 2.5, 15.4, 5, 1, 'f = g(node) + h(node)', 'data')
    add_arrow(ax, 5, 15.4, 5, 14.6)
    
    # f > limit?
    add_box(ax, 2.5, 13.6, 5, 1, 'f > f_limit?', 'decision')
    add_arrow(ax, 7.5, 14.1, 8.5, 14.1, 'YES', '#D32F2F')
    add_arrow(ax, 5, 13.6, 5, 12.8, 'NO', '#4CAF50')
    
    # Return f (budgeted)
    add_box(ax, 8.2, 13.7, 1.5, 0.8, 'Return f\n(Budgeted)', 'end', fontsize=8)
    
    # Goal check
    add_box(ax, 2.5, 11.8, 5, 1, 'node == goal?', 'decision')
    add_arrow(ax, 7.5, 12.3, 8.5, 12.3, 'YES', '#D32F2F')
    add_arrow(ax, 5, 11.8, 5, 11, 'NO', '#4CAF50')
    
    # Return Success
    add_box(ax, 8.2, 11.9, 1.5, 0.8, 'Return\nSUCCESS', 'start', fontsize=8)
    
    # For neighbors
    add_box(ax, 2, 10, 6, 1, 'best_f ← ∞\nFOR each Neighbor:', 'data')
    add_arrow(ax, 5, 10, 5, 9.2)
    
    # Recursive call
    add_box(ax, 1.5, 8.2, 7, 1, 'result_f ← RBFS(Neighbor, Goal,\nf_limit)', 'process')
    add_arrow(ax, 5, 8.2, 5, 7.4)
    
    # Success?
    add_box(ax, 2.5, 6.4, 5, 1, 'result == SUCCESS?', 'decision')
    add_arrow(ax, 7.5, 6.9, 8.5, 6.9, 'YES', '#D32F2F')
    add_arrow(ax, 5, 6.4, 5, 5.6, 'NO', '#4CAF50')
    
    # Return Success
    add_box(ax, 8.2, 6.5, 1.5, 0.8, 'Return\nSUCCESS', 'start', fontsize=8)
    
    # Update best_f
    add_box(ax, 2, 4.6, 6, 1, 'best_f ← min(best_f, result_f)', 'process')
    add_arrow(ax, 5, 4.6, 5, 3.8)
    
    # END FOR
    add_box(ax, 2.5, 2.8, 5, 1, 'END FOR', 'data')
    add_arrow(ax, 5, 2.8, 5, 2)
    
    # Return best_f
    add_box(ax, 2, 1, 6, 0.9, 'Return best_f\n(Backtrack)', 'process')
    
    # Loop
    add_arrow(ax, 2, 1.45, 0.5, 1.45)
    add_arrow(ax, 0.5, 1.45, 0.5, 8.7)
    add_arrow(ax, 0.5, 8.7, 1.5, 8.7)
    
    ax.text(0.1, 5, '↻ Recursion\nStack', fontsize=9, fontweight='bold', color='#D32F2F',
           bbox=dict(boxstyle='round,pad=0.2', facecolor='#FFEBEE'))
    
    # Description
    ax.text(5, -0.5, '📋 Recursive: Memory O(bd) | Optimal: YES | Backtrack with f_limit update', 
           fontsize=9, ha='center', style='italic',
           bbox=dict(boxstyle='round,pad=0.4', facecolor='#E0F2F1'))
    
    plt.tight_layout()
    plt.show()


# =====================================================
# MAIN DEMO
# =====================================================

if __name__ == '__main__':
    print("=" * 70)
    print("SEARCH ALGORITHMS - FLOWCHARTS")
    print("=" * 70)
    
    print("\n1️⃣  BFS: BREADTH-FIRST SEARCH")
    print("-" * 70)
    draw_bfs_flowchart()
    
    print("\n2️⃣  UCS: UNIFORM-COST SEARCH")
    print("-" * 70)
    draw_ucs_flowchart()
    
    print("\n3️⃣  IDS: ITERATIVE DEEPENING SEARCH")
    print("-" * 70)
    draw_ids_flowchart()
    
    print("\n4️⃣  GREEDY BEST-FIRST SEARCH")
    print("-" * 70)
    draw_greedy_flowchart()
    
    print("\n5️⃣  A*: A-STAR SEARCH")
    print("-" * 70)
    draw_astar_flowchart()
    
    print("\n6️⃣  RBFS: RECURSIVE BEST-FIRST SEARCH")
    print("-" * 70)
    draw_rbfs_flowchart()
    
    print("\n✅ All flowcharts completed!")