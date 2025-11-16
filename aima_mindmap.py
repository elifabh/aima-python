import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import numpy as np

def draw_aima_mindmap():
    """AIMA - Artificial Intelligence: A Modern Approach Mind Map"""
    fig, ax = plt.subplots(figsize=(20, 14))
    ax.set_xlim(-12, 12)
    ax.set_ylim(-8, 8)
    ax.axis('off')
    
    # Color scheme
    colors = {
        'center': '#FF6B6B',
        'part1': '#4ECDC4',
        'part2': '#45B7D1',
        'part3': '#FFA07A',
        'part4': '#98D8C8',
        'part5': '#F7DC6F',
        'topic': '#E8F4F8',
    }
    
    # Helper function to draw box
    def draw_box(ax, x, y, width, height, text, color, fontsize=9, fontweight='normal'):
        box = FancyBboxPatch((x - width/2, y - height/2), width, height,
                            boxstyle="round,pad=0.1",
                            facecolor=color, edgecolor='#333', linewidth=2)
        ax.add_patch(box)
        ax.text(x, y, text, fontsize=fontsize, ha='center', va='center',
               fontweight=fontweight, wrap=True)
    
    # Helper function to draw line
    def draw_line(ax, x1, y1, x2, y2, color='#666', linewidth=2):
        ax.plot([x1, x2], [y1, y2], color=color, linewidth=linewidth, zorder=0)
    
    # Center node
    draw_box(ax, 0, 0, 3.5, 1.2, 'ARTIFICIAL\nINTELLIGENCE\n(AIMA)', 
            colors['center'], fontsize=14, fontweight='bold')
    
    # Part I: FOUNDATIONS (Top Left)
    part1_x, part1_y = -9, 5.5
    draw_box(ax, part1_x, part1_y, 3, 0.8, 'PART I: FOUNDATIONS', 
            colors['part1'], fontsize=11, fontweight='bold')
    
    # Draw lines from center to Part I
    draw_line(ax, -1.5, 0.5, part1_x, part1_y - 0.4)
    
    # Part I topics
    topics_part1 = [
        ('Intelligent Agents', -9, 4.2),
        ('Search Algorithms', -9, 3.2),
        ('Adversarial Search', -9, 2.2),
        ('Constraint Satisfaction', -9, 1.2),
    ]
    
    for topic, x, y in topics_part1:
        draw_box(ax, x, y, 2.5, 0.6, topic, colors['topic'], fontsize=8)
        draw_line(ax, part1_x, part1_y - 0.4, x, y + 0.3, linewidth=1.5)
    
    # Sub-topics for Search
    search_subtopics = [
        ('BFS/DFS', -11.5, 3.2),
        ('UCS/A*', -11.5, 2.8),
        ('IDS/RBFS', -11.5, 2.4),
    ]
    for sub, x, y in search_subtopics:
        draw_box(ax, x, y, 1.8, 0.4, sub, '#F0F0F0', fontsize=7)
        draw_line(ax, -9 - 1.25, 3.2, x + 0.9, y, linewidth=1)
    
    # Part II: KNOWLEDGE & REASONING (Top Right)
    part2_x, part2_y = 9, 5.5
    draw_box(ax, part2_x, part2_y, 3, 0.8, 'PART II: KNOWLEDGE\n& REASONING', 
            colors['part2'], fontsize=11, fontweight='bold')
    
    draw_line(ax, 1.5, 0.5, part2_x, part2_y - 0.4)
    
    topics_part2 = [
        ('Knowledge Representation', 9, 4.2),
        ('Inference & Logic', 9, 3.2),
        ('Automated Planning', 9, 2.2),
        ('Ontologies', 9, 1.2),
    ]
    
    for topic, x, y in topics_part2:
        draw_box(ax, x, y, 2.8, 0.6, topic, colors['topic'], fontsize=8)
        draw_line(ax, part2_x, part2_y - 0.4, x, y + 0.3, linewidth=1.5)
    
    # Sub-topics for Logic
    logic_subtopics = [
        ('Propositional', 11.5, 3.2),
        ('First-Order', 11.5, 2.8),
        ('Horn Clauses', 11.5, 2.4),
    ]
    for sub, x, y in logic_subtopics:
        draw_box(ax, x, y, 1.8, 0.4, sub, '#F0F0F0', fontsize=7)
        draw_line(ax, 9 + 1.4, 3.2, x - 0.9, y, linewidth=1)
    
    # Part III: UNCERTAINTY (Middle Left)
    part3_x, part3_y = -9, -1.5
    draw_box(ax, part3_x, part3_y, 3, 0.8, 'PART III: UNCERTAINTY\n& REASONING', 
            colors['part3'], fontsize=11, fontweight='bold')
    
    draw_line(ax, -1.5, -0.5, part3_x, part3_y + 0.4)
    
    topics_part3 = [
        ('Probability & Bayes', -9, -2.8),
        ('Bayesian Networks', -9, -3.8),
        ('Decision Theory', -9, -4.8),
        ('Markov Models', -9, -5.8),
    ]
    
    for topic, x, y in topics_part3:
        draw_box(ax, x, y, 2.8, 0.6, topic, colors['topic'], fontsize=8)
        draw_line(ax, part3_x, part3_y - 0.4, x, y - 0.3, linewidth=1.5)
    
    # Sub-topics for Probability
    prob_subtopics = [
        ('Joint Dist.', -11.5, -2.8),
        ('Conditional', -11.5, -3.2),
        ('Independence', -11.5, -3.6),
    ]
    for sub, x, y in prob_subtopics:
        draw_box(ax, x, y, 1.8, 0.4, sub, '#F0F0F0', fontsize=7)
        draw_line(ax, -9 - 1.4, -2.8, x + 0.9, y, linewidth=1)
    
    # Part IV: LEARNING (Middle Right)
    part4_x, part4_y = 9, -1.5
    draw_box(ax, part4_x, part4_y, 3, 0.8, 'PART IV: LEARNING', 
            colors['part4'], fontsize=11, fontweight='bold')
    
    draw_line(ax, 1.5, -0.5, part4_x, part4_y + 0.4)
    
    topics_part4 = [
        ('Supervised Learning', 9, -2.8),
        ('Unsupervised Learning', 9, -3.8),
        ('Reinforcement Learning', 9, -4.8),
        ('Deep Learning', 9, -5.8),
    ]
    
    for topic, x, y in topics_part4:
        draw_box(ax, x, y, 2.8, 0.6, topic, colors['topic'], fontsize=8)
        draw_line(ax, part4_x, part4_y - 0.4, x, y - 0.3, linewidth=1.5)
    
    # Sub-topics for Supervised
    supervised_subtopics = [
        ('Decision Trees', 11.5, -2.8),
        ('Neural Networks', 11.5, -3.2),
        ('SVM/Regression', 11.5, -3.6),
    ]
    for sub, x, y in supervised_subtopics:
        draw_box(ax, x, y, 1.9, 0.4, sub, '#F0F0F0', fontsize=7)
        draw_line(ax, 9 + 1.4, -2.8, x - 0.95, y, linewidth=1)
    
    # Part V: NLP & APPLICATIONS (Bottom)
    part5_x, part5_y = 0, -7
    draw_box(ax, part5_x, part5_y, 3.5, 0.8, 'PART V: NLP, VISION\n& ROBOTICS', 
            colors['part5'], fontsize=11, fontweight='bold')
    
    draw_line(ax, 0, -0.6, part5_x, part5_y + 0.4)
    
    topics_part5 = [
        ('Natural Language\nProcessing', -7, -7.5),
        ('Computer Vision', 0, -7.7),
        ('Robotics', 7, -7.5),
    ]
    
    for topic, x, y in topics_part5:
        draw_box(ax, x, y, 2.5, 0.8, topic, colors['topic'], fontsize=8)
        draw_line(ax, part5_x, part5_y - 0.4, x, y + 0.4, linewidth=1.5)
    
    # Legend
    legend_y = 7.5
    ax.text(-11, legend_y, '📚 AIMA Book Structure - 6 Major Parts', 
           fontsize=13, fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFF9E6', edgecolor='#333', linewidth=2))
    
    # Key concepts box
    key_concepts = """
    🔑 KEY CONCEPTS ACROSS ALL PARTS:
    • Agent Architecture & Design
    • State Spaces & Problem Formulation
    • Heuristics & Evaluation Functions
    • Optimality & Completeness
    • Knowledge Representation
    • Reasoning Under Uncertainty
    • Learning from Experience
    • Real-world Applications
    """
    
    ax.text(-11, -6.5, key_concepts, fontsize=8,
           bbox=dict(boxstyle='round,pad=0.5', facecolor='#E8F8F5', 
                    edgecolor='#333', linewidth=1.5),
           verticalalignment='top', family='monospace')
    
    # Important algorithms box
    algorithms = """
    ⚙️ IMPORTANT ALGORITHMS:
    Search: BFS, DFS, UCS, A*, RBFS, IDS
    Logic: Forward/Backward Chaining, Resolution
    ML: Decision Trees, Neural Nets, SVM, K-Means
    RL: Q-Learning, Policy Iteration, Value Iteration
    """
    
    ax.text(5, -6.5, algorithms, fontsize=8,
           bbox=dict(boxstyle='round,pad=0.5', facecolor='#FCF3CF', 
                    edgecolor='#333', linewidth=1.5),
           verticalalignment='top', family='monospace')
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    print("=" * 70)
    print("AIMA - ARTIFICIAL INTELLIGENCE: A MODERN APPROACH")
    print("Mind Map Generation")
    print("=" * 70)
    
    draw_aima_mindmap()
    
    print("\n✅ Mind map created successfully!")
    print("\n📖 AIMA Book Overview:")
    print("  • Part I: Foundations (Agents, Search, Games)")
    print("  • Part II: Knowledge & Reasoning (Logic, Planning)")
    print("  • Part III: Uncertainty (Probability, Decision Making)")
    print("  • Part IV: Learning (Supervised, Unsupervised, RL)")
    print("  • Part V: Applications (NLP, Vision, Robotics)")
    print("\n🎯 Total Chapters: ~25")
    print("📊 Total Pages: ~1000+")