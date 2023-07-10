import simpy
import tkinter as tk
import random
import networkx as nx

"""
Create network graph representation
"""

G = nx.Graph()

intersection_nodes = {
    1: (100, 100),
    2: (100, 200),
    3: (200, 200),
    4: (0, 200),
    5: (100, 300)
}

edges = [
    (1, 2),
    (2, 3),
    (2, 4),
    (2, 5)
]

for index, pos in intersection_nodes.items():
    G.add_node(index, pos=pos)

G.add_edges_from(edges)

print(G)

"""
Now drawing the road network using the graph
"""

env = simpy.Environment()
root = tk.Tk()
canvas = tk.Canvas(root, width=800, height=600)
canvas.pack()


intersection_radius = 3

def draw_intersection(x, y):
    x0 = x - intersection_radius
    y0 = y - intersection_radius
    x1 = x + intersection_radius
    y1 = y + intersection_radius
    canvas.create_oval(x0, y0, x1, y1, fill="blue")

def draw_line_from_edge(a, b):
    """
    Accepts 2 nodes, position will be extracted from the intersection_nodes dictionary
    """
    a_pos = intersection_nodes[a]
    b_pos = intersection_nodes[b]

    canvas.create_line(*a_pos, *b_pos)

for index, pos in intersection_nodes.items():
    draw_intersection(*pos)

for edge in edges:
    draw_line_from_edge(*edge)


root.mainloop()