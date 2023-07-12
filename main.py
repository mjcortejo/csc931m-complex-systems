import math
import simpy
import tkinter as tk
import random
import networkx as nx
import threading
import numpy as np

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

"""
Draw the nodes as intersection junctions
"""
for index, pos in intersection_nodes.items():
    draw_intersection(*pos)

"""
Draw a line using the edge information
"""
for edge in edges:
    draw_line_from_edge(*edge)


"""
Car Class
"""
class Car:
    def __init__(self, index):
        self.index = index
        self.pos_x = None
        self.pos_y = None
        self.org_x = None
        self.org_y = None
        self.des_x = None
        self.des_y = None
        
        self.speed = 1
        self.car = None
        self.car_radius = 3
    
    def place_car(self, x, y):
        x0 = x - self.car_radius
        y0 = y - self.car_radius
        x1 = x + self.car_radius
        y1 = y + self.car_radius

        self.pos_x = x
        self.pos_y = y
        self.car = canvas.create_oval(x0, y0, x1, y1, fill="yellow")
            # return car
    
    def move_to(self, x, y):
        x0 = x - self.car_radius
        y0 = y - self.car_radius
        x1 = x + self.car_radius
        y1 = y + self.car_radius
        
        # self.pos_x = x
        # self.pos_y = y
        canvas.coords(self.car, x0, y0, x1, y1)

    def set_origin(self, origin_x, origin_y):
        self.org_x = origin_x
        self.org_y = origin_y

    def set_destination(self, destination_x, destination_y):
        self.des_x = destination_x
        self.des_y = destination_y
    

"""
Draw cars in the grid, and assign their origin and destination
"""
number_of_cars = 1
cars = []
for index in range(number_of_cars):
    car = Car(index)
    edge_choice = random.choice(edges)
    # for edge in edges():
    p1 = intersection_nodes[edge[0]]
    p2 = intersection_nodes[edge[1]]

    #place at middle part of those edges for now
    midpoint_x = (p1[0] + p2[0]) / 2
    midpoint_y = (p1[1] + p2[1]) / 2

    car.set_origin(p1[0], p1[1]) #p1 is x and y respectively
    car.place_car(midpoint_x, midpoint_y)
    car.set_destination(p2[0], p2[1]) #p2 is x and y respectivelyu
    # car = place_car(midpoint_x, midpoint_y)
    cars.append(car)


def task(env):
    while True:
        for each_car in cars:
            dx = each_car.des_x - each_car.pos_x
            dy = each_car.des_y - each_car.pos_y
            distance = math.sqrt(dx ** 2 + dy ** 2)
            if distance > 1:
                step = min(each_car.speed, distance)
                each_car.pos_x += (dx / distance) * step
                each_car.pos_y += (dy / distance) * step
                each_car.move_to(each_car.pos_x, each_car.pos_y)

        # canvas.move(car1, 0, 1)
        # print(canvas.coords(car1))
        yield env.timeout(1)


fps = 60
def run():
    env = simpy.rt.RealtimeEnvironment(factor=1/60, strict = False)
    env.process(task(env))
    env.run()

thread = threading.Thread(target=run)
thread.start()

root.mainloop()