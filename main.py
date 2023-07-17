import math
import simpy
import tkinter as tk
import random
import networkx as nx
import threading
import numpy as np

random.seed(42)

"""
Now drawing the road network using the graph
"""

env = simpy.Environment()
root = tk.Tk()
canvas = tk.Canvas(root, width=800, height=600)
canvas.pack()

"""
Create network graph representation
"""
class TrafficManager():
    def __init__(self):
        self.G = nx.Graph()
        self.intersection_nodes = {}
        self.edge_list = []
        self.edges = None
        self.intersection_states = {}
        self.intersection_radius = 4
        self.default_intersection_time = 500

        self.__build_network__()

    def change_light_state(self, intersection_node, neighboring_node, color_state=None, timer=None):
        #change color state
        if color_state is None:
            if self.intersection_states[intersection_node][neighboring_node]["color"] == "red":
                color_state = "green"
            else:
                color_state = "red"
        
        print(f"Changing the light state of intersection {intersection_node} heading to {neighboring_node} to {color_state}")
        self.intersection_states[intersection_node][neighboring_node]["color"] = color_state

        #set timer
        self.intersection_states[intersection_node][neighboring_node]["timer"] = self.default_intersection_time if timer is None else timer

    def get_intersection_light_state(self, intersection_node, neighboring_node):
        # print(f"intersection_node: {intersection_node} origin_node: {neighboring_node}")
        return self.intersection_states[intersection_node][neighboring_node]["color"]
    
    def destination_has_intersection(self, intersection_node):
        if intersection_node in self.intersection_states:
            return True
        else:
            return False

    def __build_network__(self):
        # self.entry_nodes = {
        #     'E1': (200, 50),
        #     'E2': ()
        # }
        self.intersection_nodes = {
            1: (100, 100),
            2: (200, 100),
            3: (300, 100),
            4: (100, 200),
            5: (200, 200),
            6: (300, 200),
            7: (100, 300),
            8: (200, 300),
            9: (300, 300)
        }
        self.edge_list = [
            (2, 5),
            (4, 5),
            (5, 6),
            (5, 8),
        ]

        self.edges = {i: {'has_accident': False, 'road_speed': 50, 'one_way': False} for i in self.edge_list}

        for index, pos in self.intersection_nodes.items():
            self.G.add_node(index, pos=pos)

        self.G.add_edges_from(self.edges.keys())

        #loop all nodes and check which nodes have more than 2 edges, and apply intersection states for each edge
        for n in self.G:
            # print(f"G Degree of {n}: {self.G.degree[n]}")
            if self.G.degree[n] > 2: #check if node has more than 2 neighbors then apply intersection light states
                neighbor_nodes = list(self.G.neighbors(n))
                self.intersection_states[n] = {}
                for index, neighbor in enumerate(neighbor_nodes): #needed to enumerate so I can use module to alternate values
                    color_state = "green"
                    if index % 2 == 0: #alternate light states between nodes
                        color_state = "red"
                    self.intersection_states[n][neighbor] = {
                        "color": color_state,
                        "timer": self.default_intersection_time
                    }

        """
        Draw the nodes as intersection junctions
        """
        for index, pos in self.intersection_nodes.items():
            self.__draw_intersection__(*pos, index=index)

        """
        Draw a line using the edge information
        """
        for edge in self.edges:
            self.__draw_line_from_edge__(*edge)
            

    def __draw_intersection__(self, x, y, index=None, offset=5):

        """
        Now drawing the road network using the graph
        """
        x0 = x - self.intersection_radius
        y0 = y - self.intersection_radius
        x1 = x + self.intersection_radius
        y1 = y + self.intersection_radius

        # color = intersection_states[index]

        canvas.create_oval(x0, y0, x1, y1, fill="blue")
        canvas.create_text(x + offset, y + offset, text=index)

    def __draw_line_from_edge__(self, a, b):
        """
        Accepts 2 nodes, position will be extracted from the intersection_nodes dictionary which contains the X and Y position respectively
        """
        a_pos = self.intersection_nodes[a]
        b_pos = self.intersection_nodes[b]

        lane_width = 3
        num_lanes = 2

        # Render each lane
        for i in range(num_lanes):
            start_x, start_y = a_pos[0], a_pos[1]
            end_x, end_y = b_pos[0], b_pos[1]

            # Render the lane
            canvas.create_line(start_x, start_y, end_x, end_y, width=lane_width)

tm = TrafficManager()

"""
Car Class
"""
class Car:
    def __init__(self, index):
        self.index = index
        self.pos_x = None
        self.pos_y = None

        self.origin_node = None
        self.node_paths = None
        self.next_destination_node = None
        self.final_destination_node = None

        self.speed = 1
        self.car = None
        self.car_radius = 3
        self.arrived = False

        self.light_observation_distance = 10
    
    def place_car(self, x, y):
        x0 = x - self.car_radius
        y0 = y - self.car_radius
        x1 = x + self.car_radius
        y1 = y + self.car_radius

        self.pos_x = x
        self.pos_y = y
        self.car = canvas.create_oval(x0, y0, x1, y1, fill="yellow")
            # return car
    
    def _move_to(self, x, y):
        x0 = x - self.car_radius
        y0 = y - self.car_radius
        x1 = x + self.car_radius
        y1 = y + self.car_radius
        
        canvas.coords(self.car, x0, y0, x1, y1)

    def compute_shortest_path(self, next_destination_node = None):
        paths = nx.shortest_path(tm.G, self.origin_node, self.final_destination_node)
        print(f"Car {self.index} from origin: {self.origin_node} paths: {paths}")
        # if next_destination_node:
        #     paths.insert(0, next_destination_node)
        self.node_paths = iter(paths[1:]) #ommitting first index, since it is already the origin
        self.next_destination_node = next(self.node_paths)

    def travel(self):
        #get the paths
        des_x, des_y = tm.intersection_nodes[self.next_destination_node]

        dx = des_x - self.pos_x #use euclidean distance to judge the movement of the car even in an angle
        dy = des_y - self.pos_y
        distance = math.sqrt(dx ** 2 + dy ** 2)

        # print(f"Car {self.index} distance: {distance}")

        def __move():
            step = min(self.speed, distance)
            self.pos_x += (dx / distance) * step
            self.pos_y += (dy / distance) * step
            self._move_to(self.pos_x, self.pos_y)

        if distance > self.light_observation_distance:
            __move()
            # if tm.destination_has_intersection(self.next_destination_node):
            #     if tm.get_intersection_light_state(self.next_destination_node, self.origin_node) == "red":
            #         #do not move if intersection is red
            #         pass
            #     else:
            #         __move()
            # else:
            #     __move()

        elif distance > 0:
            if tm.destination_has_intersection(self.next_destination_node):
                if tm.get_intersection_light_state(self.next_destination_node, self.origin_node) == "red":
                    #do not move if intersection is red
                    pass
                else:
                    __move()
            else:
                __move()

        # elif  self.pos_x == des_x and self.pos_y == des_y:
        else:
            try:
                print(f"Car {self.index} now heading to {self.next_destination_node} from {self.origin_node}")
                self.origin_node = self.next_destination_node
                self.next_destination_node = next(self.node_paths)
            except StopIteration:
                print(self.next_destination_node)
                self.arrived = True
                print(f"Car {self.index} has arrived to destination")
            
    def set_origin(self, origin):
        """
        origin (Integer): node index from intersection_nodes dictionary (e.g. 1)
        """
        self.origin_node = origin

    def set_destination(self, destination):
        """
        destination (Integer): node index from intersection_nodes dictionary (e.g. 1)
        """
        self.final_destination_node = destination  

    def remove_car(self):
        canvas.delete(self.car)
    

"""
Draw cars in the grid, and assign their origin and destination
"""
# non_intersection_edges = 
number_of_cars = 3
cars = []
for index in range(number_of_cars):
    car = Car(index)

    #select from a list of nodes that are not an intersection node
    edge_choice = list(random.choice(list(tm.edges.keys())))
    origin_choice = random.choice(edge_choice)

    """
    TEST 
    """
    # edge_choice = list(list(tm.edges.keys())[0])
    # origin_choice = edge_choice[1]
    """
    END TEST
    """

    edge_choice.remove(origin_choice)
    next_immediate_destination = edge_choice[0] # the remaining of the edges list

    # print("origin_choice", origin_choice)
    origin = origin_choice
    destination = 8
    # edge_choice = edges[0]

    p1 = tm.intersection_nodes[origin]
    p2 = tm.intersection_nodes[next_immediate_destination]

    #place at middle part of those edges for now
    midpoint_x = (p1[0] + p2[0]) / 2
    midpoint_y = (p1[1] + p2[1]) / 2

    car.set_origin(next_immediate_destination) #p1 is x and y respectively
    car.place_car(midpoint_x, midpoint_y)
    car.set_destination(destination) #p2 is x and y respectively
    car.compute_shortest_path()

    # print(f"Car {index} origin {origin}, immediate destination: {next_immediate_destination} destination: {destination}")

    cars.append(car)


car_task_delay = 5
def car_task(env):
    while True:
        for index, each_car in enumerate(cars):
            each_car.travel()
            
            if each_car.arrived:
                each_car.remove_car()
                cars.pop(index)
                
        yield env.timeout(car_task_delay)

def traffic_manager_task(env):
    while True:
        for each_intersection in tm.intersection_states:
            for each_neighbor in tm.intersection_states[each_intersection]:
                if tm.intersection_states[each_intersection][each_neighbor]["timer"] <= 0:
                    tm.change_light_state(each_intersection, each_neighbor)
                tm.intersection_states[each_intersection][each_neighbor]["timer"] -= 1
        yield env.timeout(1)
fps = 60
def run():
    env = simpy.rt.RealtimeEnvironment(factor=1/60, strict = False)
    env.process(car_task(env))
    env.process(traffic_manager_task(env))
    env.run()

thread = threading.Thread(target=run)
thread.start()

root.mainloop()