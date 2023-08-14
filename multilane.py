import simpy
import networkx as nx
import tkinter as tk
from tkinter import Canvas
import multiprocessing


# Define TrafficManager class
class TrafficManager():
    def __init__(self, G):
        self.G = G
        self.edges = G.edges()
        self.intersection_states = {}

    def change_light_state(self, intersection_node, neighboring_node, color_state=None, timer=None):
        # Implementation for changing light state
        pass

# Define Car class
class Car():
    def __init__(self, env, id, path):
        self.env = env
        self.id = id
        self.path = path
        self.current_node = None
        self.is_parked = False
        self.action = env.process(self.run())

    def run(self):
        for origin, destination in zip(self.path[:-1], self.path[1:]):
            yield self.env.timeout(1)  # Time to travel between nodes
            self.current_node = destination

# Simulation process for each car
def car_process(env, car, tm):
    yield env.timeout(0)  # Start immediately
    # yield env.process(car.action)
# Function to update car positions on the canvas
def update_canvas():
    canvas.delete("all")  # Clear the canvas
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos)
    nx.draw_networkx_edges(G, pos)
    nx.draw_networkx_labels(G, pos)

    # Draw cars as ovals
    for car in cars:
        pos_car = pos[car.current_node]
        x, y = pos_car[0] * canvas_width, pos_car[1] * canvas_height
        canvas.create_oval(x - 10, y - 10, x + 10, y + 10, fill="blue")

    canvas.after(100, update_canvas)  # Update every 100 ms

# Create a MultiDiGraph and TrafficManager instance
G = nx.MultiDiGraph()
G.add_nodes_from([1, 2, 3, 4, 5])
G.add_edges_from([(1, 2, 0), (2, 3, 0), (3, 4, 0), (4, 5, 0)])  # 0 represents lane number

tm = TrafficManager(G)

# Create simulation environment
fps=60
env = simpy.rt.RealtimeEnvironment(factor=1/fps, strict=False)

# Create cars and processes
cars = []
paths = [(1, 2, 3, 4, 5), (2, 3, 4, 5, 1)]

for i, path in enumerate(paths):
    car = Car(env, id=i, path=path)
    cars.append(car)
    env.process(car_process(env, car, tm))

# Run the simulation
env.run(until=100)

# Create Tkinter GUI
root = tk.Tk()
root.title("Traffic Simulation")

canvas_width = 600
canvas_height = 400
canvas = Canvas(root, width=canvas_width, height=canvas_height)
canvas.pack()

# Start updating canvas
update_canvas()

root.mainloop()  # Start the GUI event loop
