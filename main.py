import simpy
import tkinter as tk
import random
import networkx as nx

G = nx.Graph()

intersection_nodes = {
    1: (100, 100),
    2: (100, 200)
}

edges = [
    (1, 2)
]

for index, pos in intersection_nodes.items():
    G.add_node(index, pos=pos)

G.add_edges_from(edges)

print(G)

env = simpy.Environment()
root = tk.Tk()
canvas = tk.Canvas(root, width=800, height=600)
canvas.pack()

"""
Now drawing the road network using the graph
"""
car_radius = 3

def draw_intersection(x, y):
    x0 = x - car_radius
    y0 = y - car_radius
    x1 = x + car_radius
    y1 = y + car_radius
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