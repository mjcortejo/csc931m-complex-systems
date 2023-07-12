import simpy
import tkinter as tk
import random
import networkx as nx
import threading

"""
Create network graph representation
"""

G = nx.Graph()
lane_offset = 10

intersection_nodes = {
    1: (100, 100),
    2: (100, 200),
    3: (200, 200),
    4: (0, 200),
    5: (100, 300)
}

edges = [
    (1, 2),
    # (2, 3),
    # (2, 4),
    # (2, 5)
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

intersection_radius = 4

def draw_intersection(x, y):
    x0 = x - intersection_radius
    y0 = y - intersection_radius
    x1 = x + intersection_radius
    y1 = y + intersection_radius
    canvas.create_oval(x0, y0, x1, y1, fill="blue")

def draw_line_from_edge(a, b):
    """
    Accepts 2 nodes, position will be extracted from the intersection_nodes dictionary which contains the X and Y position respectively
    """
    a_pos = intersection_nodes[a]
    b_pos = intersection_nodes[b]

    canvas.create_line(*a_pos, *b_pos)

def place_car(x, y, car_radius=3):
    """
    Places a car in the grid given x and y
    """
    x0 = x - car_radius
    y0 = y - car_radius
    x1 = x + car_radius
    y1 = y + car_radius
    car = canvas.create_oval(x0, y0, x1, y1, fill="yellow")
    return car

for index, pos in intersection_nodes.items():
    draw_intersection(*pos)

for edge in edges:
    draw_line_from_edge(*edge)

# number_of_cars = 1
car1 = place_car(100, 100)

def task(env):
    while True:
        canvas.move(car1, 0, 1)
        print(canvas.coords(car1))
        yield env.timeout(1)


fps = 60
def run():
    env = simpy.rt.RealtimeEnvironment(factor=1/60, strict = False)
    env.process(task(env))
    env.run()

thread = threading.Thread(target=run)
thread.start()

root.mainloop()