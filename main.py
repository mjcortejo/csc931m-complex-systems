from typing import Type
import math
import simpy
import tkinter as tk
import random
import networkx as nx
import threading
import numpy as np

random.seed(27)

"""
Now drawing the road network using the graph
"""

env = simpy.Environment()
root = tk.Tk()
canvas = tk.Canvas(root, width=800, height=600)
canvas.pack()    

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

        self.is_spawned = False

        self.light_observation_distance = 10
    
    def place_car(self, x, y):
        x0 = x - self.car_radius
        y0 = y - self.car_radius
        x1 = x + self.car_radius
        y1 = y + self.car_radius

        self.pos_x = x
        self.pos_y = y
        self.car = canvas.create_oval(x0, y0, x1, y1, fill="yellow")

    def get_coords(self, xy_only=True):
        x0, y0, x1, y1 = canvas.coords(self.car)
        if xy_only:
            return x0, y0
        else:
            return x0 + self.car_radius, y0 + self.car_radius, x1 - self.car_radius, y1 - self.car_radius
    
    def _move_to(self, x, y):
        x0 = x - self.car_radius
        y0 = y - self.car_radius
        x1 = x + self.car_radius
        y1 = y + self.car_radius
        
        canvas.coords(self.car, x0, y0, x1, y1)

    def compute_shortest_path(self):
        """
        Compute the car agent's shortest path using NetworkX's shortest_path function (default: Djikstra)
        """        
        paths = nx.shortest_path(tm.G, self.origin_node, self.final_destination_node)
        print(f"Car {self.index} from origin: {self.origin_node} paths: {paths}")
        self.node_paths = iter(paths[1:]) #ommitting first index, since it is already the origin
        self.next_destination_node = next(self.node_paths)
        tm.manage_car_from_edge(self, self.origin_node, self.next_destination_node, how="add")

    def spawn(self, origin, next_immediate_destination, final_destination):
        """
         Spawn car at the origin node (usually an entry node).
         
         @param origin - origin of car to spawn
         @param next_immediate_destination - next destination of car to spawn
         @param final_destination - final destination of car to spawn
        """
        p1 = tm.intersection_nodes[origin]
        # p2 = tm.intersection_nodes[next_immediate_destination]

        #place at middle part of those edges for now
        # midpoint_x = (p1[0] + p2[0]) / 2
        # midpoint_y = (p1[1] + p2[1]) / 2
        self.set_origin(origin) #p1 is x and y respectively
        self.place_car(p1[0], p1[1])
        self.set_destination(final_destination) #p2 is x and y respectively
        self.compute_shortest_path()
        self.is_spawned = True

    def travel(self):
        """
         Move the car to the next destination based on the speed and distance. This is called every frame
        """
        des_x, des_y = tm.intersection_nodes[self.next_destination_node]

        dx = des_x - self.pos_x #use euclidean distance to judge the movement of the car even in an angle
        dy = des_y - self.pos_y
        distance = math.sqrt(dx ** 2 + dy ** 2)

        def __move():
            """
             Move the agent to the next position based on the speed and distance.

            #WARN: Starting to get performance issues

            """

            # Get the cars that are in the same edge as the current car. but also remove self in that list to prevent measuring its own position
            cars_in_the_same_edge = tm.get_cars_in_edge(self.origin_node, self.next_destination_node)
            cars_in_the_same_edge.remove(self)

            #get distance of other cars
            distance_to_other_cars = [math.dist(adjacent_car.get_coords(), self.get_coords()) for adjacent_car in cars_in_the_same_edge]
            print(distance_to_other_cars)

            step = min(self.speed, distance)
            self.pos_x += (dx / distance) * step
            self.pos_y += (dy / distance) * step
            self._move_to(self.pos_x, self.pos_y)

        # This method is called when the distance is below the light observation distance threshold.
        if distance > self.light_observation_distance:
            __move()

        elif distance > 0:
            # Move the destination to the next destination node if the intersection is green
            if tm.destination_has_intersection(self.next_destination_node):
                if tm.get_intersection_light_state(self.next_destination_node, self.origin_node) == "red":
                    #do not move if intersection is red
                    pass
                else:
                    __move()
            else:
                __move()
        else:
            try:
                tm.manage_car_from_edge(self, self.origin_node, self.next_destination_node, how="remove")

                print(f"Car {self.index} now heading to {self.next_destination_node} from {self.origin_node}")
                self.origin_node = self.next_destination_node
                self.next_destination_node = next(self.node_paths)

                tm.manage_car_from_edge(self, self.origin_node, self.next_destination_node, how="add")

            except StopIteration:
                print(f"StopIteration {self.next_destination_node}")
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
Create network graph representation
"""
class TrafficManager():
    def __init__(self):
        self.G = nx.DiGraph()
        self.intersection_nodes = {}
        self.edge_list = []
        self.edges = None
        self.intersection_states = {}
        self.intersection_radius = 4
        self.default_intersection_time = 500

        self.entry_nodes = []
        self.entry_edges = []

        self.__build_network__()

    def change_light_state(self, intersection_node, neighboring_node, color_state=None, timer=None):
        #change color state
        # Set color state to red if color is red or red
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
        """
         Get the light state of a neighboring node. This is used to determine whether or not a node is intersecting another node.
         
         @param intersection_node - The node that is intersecting.
         @param neighboring_node - The node that is a neighboring node.
         
         @return The light state of the neighboring node in the color
        """
        return self.intersection_states[intersection_node][neighboring_node]["color"]
    
    def destination_has_intersection(self, intersection_node):
        """
         Checks if the destination node has an intersection. This is used to determine if there is a point in the destination to be intersected with the source
         
         @param intersection_node - The node that we are interested in
         @return true if the intersection node is in the intersection states
        """
        if intersection_node in self.intersection_states:
            return True
        else:
            return False

    def __build_network__(self):
        """
         Builds the network
        """
        self.intersection_nodes = {
            #ENTRY NODES
            'E1': (200, 50), #just above Node 2
            'E2': (350, 200), #just right of Node 6
            #PROPER NODES
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

        #use a real layout like EDSA or Ayala or BGC
        self.edge_list = [
            ('E1', 2),
            ('E2', 6),
            (2, 5),
            (4, 5),
            (5, 6),
            (5, 8),
        ]

        self.edges = {i: {'cars_occupied': [], 'has_accident': False, 'road_speed': 50, 'one_way': False} for i in self.edge_list}

        # Add a node to the graph.
        for index, pos in self.intersection_nodes.items():
            # Add index to the list of entries in the entry_nodes list.
            if type(index) == str and "E" in index:
                self.entry_nodes.append(index)
            self.G.add_node(index, pos=pos)

        # Add edges to the graph.
        for edges in self.edge_list:
            # Add edges to the entry edges list
            if any(isinstance(edge, str) for edge in edges) and any("E" in edge for edge in edges):
                #for now we will assume that the entry edge is at the first element
                self.entry_edges.append(edges)

            self.G.add_edge(edges[0], edges[1])
            self.G.add_edge(edges[1], edges[0])

        # Loop all nodes and check which nodes have more than 2 edges, and apply intersection states for each edge
        for n in self.G:
            # print(f"G Degree of {n}: {self.G.degree[n]}")
            # Apply intersection light states between nodes
            if self.G.degree[n] > 2: #check if node has more than 2 neighbors then apply intersection light states
                neighbor_nodes = list(self.G.neighbors(n))
                self.intersection_states[n] = {}
                # This function is used to generate a dictionary of light states between nodes and neighbors
                for index, neighbor in enumerate(neighbor_nodes): #needed to enumerate so I can use module to alternate values
                    color_state = "green"
                    if index % 2 == 0: #alternate light states between nodes
                        color_state = "red"
                    self.intersection_states[n][neighbor] = {
                        "color": color_state,
                        "timer": self.default_intersection_time
                    }

        # Draw the intersection of all nodes in the intersection_nodes.
        for index, pos in self.intersection_nodes.items():
            self.__draw_intersection__(*pos, index=index)

        # Draw all lines from the edges of the graph.
        for edge in self.edges:
            self.__draw_line_from_edge__(*edge)

    #TODO: Change car_object type annotation to Car,by using from typing import Type
    def manage_car_from_edge(self, car_object: Car, origin: int, destination: int, how: str):
        orientation = None

        # Example tuple (1, 2) Where 1 is how it is placed in the edges list originally and 2 is the immediate destination
        # And a instance where an agent is going from (2, 1) we still want to recognize this as (1, 2) as it was defined in the edges list
        # This is what this function does by switching its orientation if this instance happens to be happening
        if any(((origin, destination) in self.edges.keys(), (destination, origin) in self.edges.keys())):
            orientation = (origin, destination) if (origin, destination) in self.edges.keys() else (destination, origin)

        if orientation:
            if how == "add":
                self.edges[orientation]['cars_occupied'].append(car_object)
                print(f"Added {car_object.index} to {orientation}")
            elif how == "remove":
                self.edges[orientation]['cars_occupied'].remove(car_object)
                print(f"Removing {car_object.index} to {orientation}")
            else: raise Exception("Invalid 'how' value, must be 'add' or 'remove'")
        else:
            raise KeyError(f"Cannot find the edge {(origin, destination)} or {(destination,origin)}")
        
    def get_cars_in_edge(self, origin, destination) -> list[Car]:
        orientation = None
        cars_in_edge = None
        if any(((origin, destination) in self.edges.keys(), (destination, origin) in self.edges.keys())):
            orientation = (origin, destination) if (origin, destination) in self.edges.keys() else (destination, origin)
            cars_in_edge = self.edges[orientation]['cars_occupied'].copy() #shallow copy as we don't want to alter original list of cars

        return cars_in_edge

    def __draw_intersection__(self, x, y, index=None, offset=5, color="blue"):
        """
         Draw the road network using the graph. This is the function that is called by __draw_network
         
         @param x - X coordinate of the top left corner of the graph
         @param y - Y coordinate of the top left corner of the graph
         @param index - Index of the node to draw. If None the node is drawn at the edge
         @param offset - Offset to draw the node at
         @param color - Color of the node
        """

        x0 = x - self.intersection_radius
        y0 = y - self.intersection_radius
        x1 = x + self.intersection_radius
        y1 = y + self.intersection_radius

        # color = intersection_states[index]
        
        if type(index) == str and "E" in index:
            color="purple"

        canvas.create_oval(x0, y0, x1, y1, fill=color)
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
Draw cars in the grid, and assign their origin and destination
"""
number_of_cars = 6
cars = []
for index in range(number_of_cars):
    car = Car(index)
    cars.append(car)

spawn_delay = 50
def car_spawn_task(env):
    while True:
        for each_car in cars:
            if not each_car.is_spawned:
                edge_choice = list(random.choice(list(tm.entry_edges)))
                origin = edge_choice[0]
                
                next_immediate_destination = edge_choice[1]
                final_destination = 8
                each_car.spawn(origin, next_immediate_destination, final_destination)
            yield env.timeout(spawn_delay)
         
car_task_delay = 1
def car_task(env):
    while True:
        for index, each_car in enumerate(cars):
            if not each_car.is_spawned:
                #wait for spawn
                pass
            else:
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
    env.process(car_spawn_task(env))
    env.process(car_task(env))
    env.process(traffic_manager_task(env))
    env.run()

thread = threading.Thread(target=run)
thread.start()

root.mainloop()