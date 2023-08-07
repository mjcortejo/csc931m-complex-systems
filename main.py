from typing import Type
import math
import simpy
import tkinter as tk
import random
import networkx as nx
import threading
import numpy as np
import logging
from layouts import *

from concurrent.futures import ThreadPoolExecutor

random.seed(27)

"""
Now drawing the road network using the graph
"""

env = simpy.Environment()
root = tk.Tk()
# child = tk.Toplevel()

canvas = tk.Canvas(root, width=800, height=600)
# child_canvas = tk.Canvas(child, width=500, height=1000)

# child_canvas.pack()
canvas.pack()    

color_list = [
    "snow",
    "ghost white",
    "gainsboro",
    "old lace",
    "linen",
    "antique white",
    "papaya whip",
    "blanched almond",
    "bisque",
    "peach puff",
    "navajo white",
    "lemon chiffon",
    "mint cream",
    "azure",
    "alice blue",
    "lavender",
    "lavender blush",
    "misty rose",
    "turquoise", 
    "aquamarine", 
    "powder blue", 
    "sky blue", 
    "steel blue", 
    "cadet blue", 
    "deep sky blue", 
    "dodger blue", 
    "cornflower blue", 
    "medium aquamarine", 
    "medium turquoise", 
    "light sea green", 
    "medium sea green"
]


"""
Car Class
"""
class Car:
    def __init__(self, index):
        # X Y coordinates, and index of car for canvas ID-ing
        self.index = index
        self.pos_x = None
        self.pos_y = None
        # Node position and destination information.
        self.origin_node = None #current origin
        self.last_origin_node = None # last recorded origin
        self.next_destination_node = None # next immediate destination
        self.final_destination_node = None # final destination
        self.node_paths = None # collection of node paths to take by the shortest path finding algorithm
        self.next_edge = None
        #Attributes
        self.speed = 1 # put range of numbers for variability
        self.car = None # Car canvas object itself
        self.car_radius = 3 # Car canvas radius size
        self.wait_time = 0
        # States
        self.arrived = False
        self.is_spawned = False
        self.is_moving = False
        # Awareness Attributes
        self.light_observation_distance = 5
        self.car_collision_observe_distance = 8
        self.cars_in_front = None # collection of other car objects
    
    def place_car(self, x, y):
        x0 = x - self.car_radius
        y0 = y - self.car_radius
        x1 = x + self.car_radius
        y1 = y + self.car_radius

        self.pos_x = x
        self.pos_y = y

        choice_color = random.choice(color_list)
        self.car = canvas.create_oval(x0, y0, x1, y1, fill=choice_color)
        # self.car.bind("<Enter>", self.__on_hover)

    def get_coords(self, xy_only=True):
        x0, y0, x1, y1 = canvas.coords(self.car)
        if xy_only:
            return x0, y0
        else:
            return x0 + self.car_radius, y0 + self.car_radius, x1 - self.car_radius, y1 - self.car_radius
        
    # def __on_hover(self, event):
    #     print(f'You hovered car at Car {self.index} {event.x} X {event.y}.')
    
    def _move_to(self, x, y):
        x0 = x - self.car_radius
        y0 = y - self.car_radius
        x1 = x + self.car_radius
        y1 = y + self.car_radius
        
        canvas.coords(self.car, x0, y0, x1, y1)

    def __check_subsequence__(self, paths):
        paths_to_check = paths.copy()

        if self.last_origin_node is not None:
            paths_to_check.insert(0, self.last_origin_node)
        for sequence, node_to_remove in tm.disallowed_sequences.items():
            sequence = list(sequence)
            n = len(sequence)
            for i in range(len(paths_to_check)):
                if paths_to_check[i:i+n] == sequence:
                    return True, node_to_remove
            return False, None

    def compute_shortest_path(self):
        """
        Compute the car agent's shortest path using NetworkX's shortest_path function (default: Djikstra)
        """
        # print(self.origin_node, self.final_destination_node)        
        paths = nx.dijkstra_path(tm.G, self.origin_node, self.final_destination_node, weight='weight')
        self.next_edge = None
        if tm.disallowed_sequences is not None:
            is_illegal_path, node_to_remove = self.__check_subsequence__(paths)
            if is_illegal_path:
                temp_G = tm.G.copy()
                temp_G.remove_node(node_to_remove)
                paths = nx.dijkstra_path(temp_G, self.origin_node, self.final_destination_node, weight='weight')

        self.node_paths = iter(paths[1:]) #ommitting first index, since it is already the origin
        self.next_destination_node = next(self.node_paths)
        post_next_destination_node = next(self.node_paths, None)
        if post_next_destination_node:
            self.next_edge = (self.next_destination_node, post_next_destination_node)
    
    def spawn(self, origin, final_destination):
        """
         Spawn car at the origin node (usually an entry node).
         
         @param origin - origin of car to spawn
         @param final_destination - final destination of car to spawn
        """
        p1 = tm.intersection_nodes[origin]

        self.set_origin(origin) #p1 is x and y respectively
        self.place_car(p1[0], p1[1])
        self.set_destination(final_destination) #p2 is x and y respectively
        self.compute_shortest_path()
        tm.manage_car_from_edge(self, self.origin_node, self.next_destination_node, how="add")
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
            current_car_index = cars_in_the_same_edge.index(self)

            cars_in_the_same_edge = cars_in_the_same_edge[:current_car_index]

            #get distance of other cars, variable name originally distance_to_other_cars
            distance_to_front_cars = [math.dist(adjacent_car.get_coords(), self.get_coords()) for adjacent_car in cars_in_the_same_edge] if cars_in_the_same_edge else None

            self.cars_in_front = distance_to_front_cars

            #then get nearest car by performing min() func to the distance of other front cars
            nearest_car = min(distance_to_front_cars) if distance_to_front_cars else None

            if not nearest_car or nearest_car > self.car_collision_observe_distance:
                step = min(self.speed, distance)
                self.pos_x += (dx / distance) * step
                self.pos_y += (dy / distance) * step
                self._move_to(self.pos_x, self.pos_y)
                self.is_moving = True
            else:
                self.is_moving = False

        if not self.is_moving:
            self.wait_time += 1 #registers at ticks which we'll have to convert to seconds
        # This method is called when the distance is below the light observation distance threshold.
        if distance > self.light_observation_distance:
            __move()

        elif distance > 0:

            if "P" in str(self.next_destination_node): # If car is heading to a parking node
                cars_occupied, edge_capacity = tm.manage_parking(self, self.next_destination_node, how="inquire") # Check if parking is available
            else:
                cars_occupied, edge_capacity = tm.get_edge_traffic(self.next_edge)
            
            is_next_destination_available = True if cars_occupied < edge_capacity else False

            if tm.destination_has_intersection(self.next_destination_node):
                if tm.get_intersection_light_state(self.next_destination_node, self.origin_node) == "green" and is_next_destination_available:
                    __move()
            elif is_next_destination_available:
                __move()
        else:
            try:
                if "P" in str(self.next_destination_node): #If the car is heading to a parking node
                    tm.manage_parking(self, self.next_destination_node, how="add") # then add itself to the parking node
                tm.manage_car_from_edge(self, self.origin_node, self.next_destination_node, how="remove")
                
                self.last_origin_node = self.origin_node
                self.origin_node = self.next_destination_node

                # place recomputation of shortest path here
                self.compute_shortest_path()
                logging.info(f"Car {self.index} now heading to {self.next_destination_node} from {self.origin_node}")

                tm.manage_car_from_edge(self, self.origin_node, self.next_destination_node, how="add")
                self.wait_time = 0;

            except StopIteration as e:
                # print(f"StopIteration {self.next_destination_node}, {e}")
                self.arrived = True

                #remove self after execution of final destination
                self.remove_car()
            
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
    def __init__(self, intersection_nodes, edge_list, **kwargs):
        """
        Accepts nodes as intersection points and a list containing the edges.
        edge_list should contain the nodes found in intersection points, else will return error.
        """
        self.G = nx.DiGraph()
        #graph for 
        self.intersection_nodes = intersection_nodes
        self.edge_list = edge_list
        # self.validate_network()
        self.edges = None
        self.intersection_states = {}
        self.intersection_radius = 4
        self.default_intersection_time = 300

        # kwargs
        self.parking_capacities = kwargs['parking_capacities'] if 'parking_capacities' in kwargs.keys() else None
        self.CONST_SYSTEM_DEFAULT_PARKING_CAPACITY = 100
        if self.parking_capacities is None: print("No Parking Capacities found. All parking nodes will default to {self.CONST_SYSTEM_DEFAULT_PARKING_CAPACITY}"); self.parking_capacities = self.CONST_SYSTEM_DEFAULT_PARKING_CAPACITY

        self.disallowed_sequences = kwargs['disallowed_sequences'] if 'disallowed_sequences' in kwargs.keys() else None
        if self.disallowed_sequences is None: print("No Disallowed path sequences found.")

        self.default_edge_capacity = kwargs['default_edge_capacity'] if 'default_edge_capacity' in kwargs.keys() else None
        self.CONST_SYSTEM_DEFAULT_EDGE_CAPACITY = 10 #This will only be used if the user did not defined a default edge capacity of edge entries with no capacity values

        if self.default_edge_capacity is None: print(f"Default Edge Capacity not defined. All undefined edge capacity values will default to {self.CONST_SYSTEM_DEFAULT_EDGE_CAPACITY}. This might emerge wierd behaviors for the entire system."); self.default_edge_capacity = self.CONST_SYSTEM_DEFAULT_EDGE_CAPACITY

        #end kwargs
        self.entry_nodes = []
        self.parking_nodes = {} #dictionary because we need to contain occupied cars and capacity

        self.entry_edges = []

        self.__build_network__()

    def change_light_state(self, intersection_node, neighboring_node, color_state=None, timer=None):
        # Set color state to red if color is red or red
        if color_state is None:
            if self.intersection_states[intersection_node][neighboring_node]["color"] == "red":
                color_state = "green"
            else:
                color_state = "red"
        
        # print(f"Changing the light state of intersection {intersection_node} heading to {neighboring_node} to {color_state}")
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

    def get_edge_traffic(self, orientation=None):
        cars_occupied = -1
        edge_capacity = 0
        if orientation:
            cars_occupied = len(self.edges[orientation]['cars_occupied'])
            edge_capacity = self.edges[orientation]['max_capacity']
        return cars_occupied, edge_capacity
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

        # self.edges = {i: {'cars_occupied': [], 'weight': 0} for i in self.edge_list}
        self.edges = {(i[0], i[1]): {'cars_occupied': [], 'weight': 0, 'max_capacity': i[2] if len(i) > 2 else self.default_edge_capacity} for i in self.edge_list}

        # Add a node to the graph.
        for index, pos in self.intersection_nodes.items():
            # Add index to the list of entries in the entry_nodes list.
            if "E" in str(index):
                self.entry_nodes.append(index)
            elif "P" in str(index):
                # self.parking_nodes.append(index)
                # self.parking_nodes[index] = (0, self.parking_capacities[index]) #Tuple of (cars_occupied, max capacity)
                self.parking_nodes[index] = {
                    "cars_occupied": [],
                    "max_capacity": self.parking_capacities[index]
                }
            self.G.add_node(index, pos=pos)

        # Add edges to the graph.
        for edges in self.edge_list:
            # Add edges to the entry edges list
            self.G.add_edge(edges[0], edges[1])
            if any("E" in str(edge) for edge in edges):
                #for now we will assume that the entry edge is at the first element
                self.entry_edges.append(edges)

                #add the inverse edge of the E's as well
                self.edges[(edges[1], edges[0])] = {'cars_occupied': [], 'weight': 0, 'max_capacity': edges[2] if len(edges) > 2 else self.default_edge_capacity}
                self.G.add_edge(edges[1], edges[0])
            elif any("P" in str(edge) for edge in edges):
                if "P" in str(edges[0]): #this one is more likely to happen for now
                    # self.entry_edges.append((edges[0], edges[1]))

                    #add the inverse edge of the P's as well
                    self.edges[(edges[1], edges[0])] = {'cars_occupied': [], 'weight': 0, 'max_capacity': edges[2] if len(edges) > 2 else self.default_edge_capacity}
                    self.G.add_edge(edges[1], edges[0])

        # Loop all nodes and check which nodes have more than 2 edges, and apply intersection states for each edge
        for n in self.G:
            # Apply intersection light states between nodes
            if self.G.in_degree[n] > 2: #check if node has more than 3 neighbors then apply intersection light states.
                neighbor_nodes = list(self.G.predecessors(n)) #must use G.predecessors instead of neighbors
                self.intersection_states[n] = {}
                # This function is used to generate a dictionary of light states between nodes and neighbors
                for index, neighbor in enumerate(neighbor_nodes): #needed to enumerate so I can use module to alternate values
                    color_state = "green"

                    # TODO: Change this back to 3 once its fixed
                    if index % 3 == 0: #alternate light states between nodes
                        color_state = "red"
                    self.intersection_states[n][neighbor] = {
                        "color": color_state,
                        "timer": self.default_intersection_time
                    }

                    # color_state = "red"
                    # print(f"Setting {(n, neighbor)} to {color_state}")

        # Draw the intersection of all nodes in the intersection_nodes.
        for index, pos in self.intersection_nodes.items():
            self.__draw_intersection__(*pos, index=index)

        # Draw all lines from the edges of the graph.
        for edge in self.edges:
            self.__draw_line_from_edge__(*edge)

    #TODO: Change car_object type annotation to Car,by using from typing import Type
    def manage_car_from_edge(self, car_object: Car, origin: int, destination: int, how: str):
        orientation = (origin, destination)

        if orientation:
            if how == "add":
                self.edges[orientation]['cars_occupied'].append(car_object)
                # print(f"Added {car_object.index} to {orientation}")
            elif how == "remove":
                self.edges[orientation]['cars_occupied'].remove(car_object)
                # print(f"Removing {car_object.index} to {orientation}")
            else: raise Exception("Invalid 'how' value, must be 'add' or 'remove'")

            # Dynamic Weighting mechanism
            cars_occupied, edge_capacity = self.get_edge_traffic(orientation)
            self.edges[orientation]['weight'] = cars_occupied / edge_capacity #supposedly cars occupied / max capacity of edge
            # print(f"Adjusting weight of {orientation} to {self.edges[orientation]['weight']}")
        else:
            raise KeyError(f"Cannot find the edge {(origin, destination)} or {(destination,origin)}")

    def manage_parking(self, car_object: Car, parking_node: int, how: str):
        if how == "add":
            self.parking_nodes[parking_node]["cars_occupied"].append(car_object)
        elif how == "remove":
            self.parking_nodes[parking_node]["cars_occupied"].remove(car_object)
        elif how == "inquire": #Inquire count of cars occupied
            return len(self.parking_nodes[parking_node]["cars_occupied"]), self.parking_nodes[parking_node]["max_capacity"]
        else:
            raise KeyError(f"Invalid 'how' value, must be 'add', 'remove' or 'inquire' (returns an int (number of cars parked), int (max capacity))")
        
    def get_cars_in_edge(self, origin, destination) -> list[Car]:
        orientation = (origin, destination)
        return self.edges[orientation]['cars_occupied'].copy()

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

    def __draw_line_from_edge__(self, a, b, capacity=None):
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


intersection_nodes, edge_list, parking_capacities = bgc_layout()
# intersection_nodes, edge_list = bgc_short_test()
tm = TrafficManager(intersection_nodes, edge_list, 
                    parking_capacities=parking_capacities,
                    # disallowed_sequences=disallowed_sequences, 
                    default_edge_capacity=10)

"""
Draw cars in the grid, and assign their origin and destination
"""
number_of_cars = 2000
cars = []

#create a text canvas widget
canvas_index = 0
logs = {}


for index in range(number_of_cars):
    car = Car(index)
    cars.append(car)

spawn_delay = 5
y_offset = 50

def car_spawn_task(env):
    global canvas_index
    while True:
        for each_car in cars:
            if not each_car.is_spawned:
                canvas_index += 1
                entry_choice = list(random.choice(list(tm.entry_edges)))
                origin = entry_choice[0]
                immediate_destination = entry_choice[1]
                
                # entry_nodes = list(tm.entry_nodes)
                parking_nodes = list(tm.parking_nodes.keys())

                # entry_nodes.remove(origin)

                #final destination can either be entry nodes or parking nodes
                # final_destination = random.choice(entry_nodes)
                final_destination = random.choice(parking_nodes)

                cars_occupied, edge_capacity = tm.get_edge_traffic((origin, immediate_destination))

                if cars_occupied <= edge_capacity:
                    each_car.spawn(origin, final_destination)
                else:
                    logging.info(f"{(origin, immediate_destination)} cannot spawn due to full")

                #generate text widget
                # text_log = child_canvas.create_text(0, y_offset * canvas_index + 10, anchor='nw', text="START")
                # logs[each_car.index] = text_log

            yield env.timeout(spawn_delay)
         
car_task_delay = 1
def car_movement_logic(each_car):
    if not each_car.is_spawned:
        # Wait for spawn
        pass
    else:
        each_car.travel()

def car_task(env):
    while True:
        with ThreadPoolExecutor(max_workers=64) as executor:
            # Execute the car_movement_logic for each car concurrently in multiple threads
            executor.map(car_movement_logic, cars)

        # Remove completed cars
        cars[:] = [each_car for each_car in cars if not each_car.arrived]

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
    env = simpy.rt.RealtimeEnvironment(factor=1/60, strict=False)
    env.process(car_spawn_task(env))
    env.process(car_task(env))
    env.process(traffic_manager_task(env))
    env.run()

thread = threading.Thread(target=run)
thread.start()

root.mainloop()