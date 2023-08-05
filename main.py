from typing import Type
import math
import simpy
import tkinter as tk
import random
import networkx as nx
import threading
import numpy as np
import logging

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
        self.last_origin = None # last recorded origin
        self.next_destination_node = None # next immediate destination
        self.final_destination_node = None # final destination
        self.node_paths = None # collection of node paths to take by the shortest path finding algorithm
        #Attributes
        self.speed = .3 # put range of numbers for variability
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

    def __check_subsequence__(self, paths):
        paths_to_check = paths.copy()

        if self.last_origin is not None:
            paths_to_check.insert(0, self.last_origin)
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
        paths = nx.dijkstra_path(tm.G, self.origin_node, self.final_destination_node, weight='weight')

        is_illegal_path, node_to_remove = self.__check_subsequence__(paths)
        if is_illegal_path:
            temp_G = tm.G.copy()
            temp_G.remove_node(node_to_remove)
            paths = nx.dijkstra_path(temp_G, self.origin_node, self.final_destination_node, weight='weight')

        self.node_paths = iter(paths[1:]) #ommitting first index, since it is already the origin
        self.next_destination_node = next(self.node_paths)
    
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

                # logger.info()
                
                self.last_origin = self.origin_node
                self.origin_node = self.next_destination_node

                # place recomputation of shortest path here
                self.compute_shortest_path()
                logging.info(f"Car {self.index} now heading to {self.next_destination_node} from {self.origin_node}")

                tm.manage_car_from_edge(self, self.origin_node, self.next_destination_node, how="add")
                self.wait_time = 0;

            except StopIteration:
                logging.info(f"StopIteration {self.next_destination_node}")
                self.arrived = True
                logging.info(f"Car {self.index} has arrived to destination")

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

        self.disallowed_sequences = kwargs['disallowed_sequences'] if 'disallowed_sequences' in kwargs.keys() else {}
        if self.disallowed_sequences is None: logging.info("No Disallowed path sequences found.")

        self.default_edge_capacity = kwargs['default_edge_capacity'] if 'default_edge_capacity' in kwargs.keys() else None
        self.CONST_SYSTEM_DEFAULT_EDGE_CAPACITY = 10 #This will only be used if the user did not defined a default edge capacity of edge entries with no capacity values

        if self.default_edge_capacity is None: 
            logging.warning(f"Default Edge Capacity not defined, If there are no edge capacity values found in edge_list. The value will default to {self.CONST_SYSTEM_DEFAULT_EDGE_CAPACITY}. This might emerge a wierd behavior for the entire system.")
            self.default_edge_capacity = self.CONST_SYSTEM_DEFAULT_EDGE_CAPACITY

        self.entry_nodes = []
        self.entry_edges = []

        self.__build_network__()

    def change_light_state(self, intersection_node, neighboring_node, color_state=None, timer=None):
        # Set color state to red if color is red or red
        if color_state is None:
            if self.intersection_states[intersection_node][neighboring_node]["color"] == "red":
                color_state = "green"
            else:
                color_state = "red"
        
        logging.info(f"Changing the light state of intersection {intersection_node} heading to {neighboring_node} to {color_state}")
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

    def get_edge_traffic(self, origin_node, destination_node):
        orientation = (origin_node, destination_node)
        cars_occupied = len(self.edges[orientation]['cars_occupied'])
        edge_capacity = self.edges[orientation]['max_capacity']
        return {
            'cars_occupied': cars_occupied,
            'edge_capacity': edge_capacity
        }
    
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
            if "E" in str(index) or "P" in str(index):
                self.entry_nodes.append(index)
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
                    self.entry_edges.append((edges[0], edges[1]))

                    #add the inverse edge of the P's as well
                    self.edges[(edges[1], edges[0])] = {'cars_occupied': [], 'weight': 0, 'max_capacity': edges[2] if len(edges) > 2 else self.default_edge_capacity}
                    self.G.add_edge(edges[1], edges[0])

        # Loop all nodes and check which nodes have more than 2 edges, and apply intersection states for each edge
        for n in self.G:
            # Apply intersection light states between nodes
            if self.G.in_degree[n] > 2 & self.G.out_degree[n] > 2: #check if node has more than 3 neighbors then apply intersection light states.
                neighbor_nodes = list(self.G.neighbors(n))
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

                    logging.info(f"Setting {n}, Neighbor {neighbor} to {color_state}")

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
            cars_occupied = len(self.edges[orientation]['cars_occupied'])
            edge_capacity = self.edges[orientation]['max_capacity']
            self.edges[orientation]['weight'] = cars_occupied / edge_capacity #supposedly cars occupied / max capacity of edge
            print(f"Adjusting weight of {orientation} to {self.edges[orientation]['weight']}")
        else:
            raise KeyError(f"Cannot find the edge {(origin, destination)} or {(destination,origin)}")
        
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

def bgc_layout():
    intersection_nodes = {
        #ENTRY NODES
        'E1': (600, 80), 
        'E2': (50, 50), 
        'E3': (100, 500),
        'E4': (650, 500),
        #PARKING NODES
        'P1': (150, 125),
        'P2': (75, 275),
        'P3': (450, 375),
        # CONNECTOR NODES (these are used to connect to parking nodes)
        'C1': (200, 125),
        'C2': (100, 275),
        'C3': (400, 375),
        #PROPER NODES
        # 1st BGC parallel nodes
        1: (100, 100),
        2: (200, 100),
        3: (300, 100),
        4: (400, 100),
        5: (500, 100),
        6: (600, 125),
        # 2nd Parallel nodes
        7: (50, 200),
        8: (100, 200),
        9: (200, 150),
        10: (300, 150),
        11: (400, 150),
        12: (500, 150),
        # 3rd Parallel Nodes
        13: (50, 300),
        14: (100, 300),
        15: (200, 350),
        16: (300, 350),
        17: (400, 350),
        18: (500, 350),
        # 4th Parallel Nodes
        19: (100, 400),
        20: (200, 400),
        21: (300, 400),
        22: (400, 400),
        23: (500, 400),
        24: (600, 375)
    }

    #('E2', 1)
    edge_list = [
        # TUPLE SYNTAX: (Node1, Node2, Capacity) -> (1, 2, 30). 30 being the capcity. If none entered, will be using a default value by the TrafficManager class
        # Important: Current rules for placing edges.
        # 1. For Entry (E) nodes, they must be placed first for each of the tuples
        # 2. For Parking (P) nodes, they must be placed first for each of the tuples
        # 3. //TODO something about connectors only connected to one direction
        # 4. If you want to implement edge capacity, add a third value to the tuple ex. (1, 2, 30) 30 -> Capacity
        # Entry / Exit Nodes
        ('E1', 6, 20),('E2', 7, 20),('E3', 19, 20),('E4', 24, 20),
        #Parking and connector nodes,
        ('P1', 'C1'), (2, 'C1') , ('C1', 9),
        ('P2', 'C2'), ('C2', 8), (14, 'C2'),
        ('P3', 'C3'), (22, 'C3'), ('C3', 17), #Gallery Parkade
        # Removed
        # (8, 'C2'),('C2', 14),
        # 1st Parallel Nodes
        (1, 2, 10),(1, 7, 10),
        (2, 1, 10),(2, 3, 10),
        (3, 2, 10),(3, 4, 10),(3, 10, 7),
        (4, 3, 10),(4, 5, 10),(4, 11, 7),
        (5, 4, 10),(5, 6, 10),(5, 12, 7),
        (6, 5, 10),(6, 24, 25),
        (7, 13, 10),
        (8, 1, 10),(8, 7, 7),
        (9, 2, 7),(9, 8, 15),(9, 15, 20),
        (10, 9, 10),(10, 11, 10),(10, 16, 20),
        (11, 4, 7),(11, 10, 10),
        (12, 5, 7),(12, 11, 10),(12, 18, 20),
        (13, 14, 7),(13, 19, 10),
        (14, 15, 15),
        (15, 9, 20),(15, 16, 10),(15, 20, 7),
        (16, 17, 10),(16, 21, 7),
        (17, 11, 20),(17, 18, 10),
        (18, 12, 20),(18, 23, 7),
        (19, 14, 10),(19, 20, 10),
        (20, 15, 7),(20, 19, 10),(20, 21, 10),
        (21, 20, 10),(21, 22, 10),
        (22, 21, 10),(22, 23, 10),
        (23, 18, 7),(23, 22, 10),(23, 24, 10),
        (24, 6, 25),(24, 23, 10)
    ]
    #REMOVED DUE TO ONE WAY
    """
    (1, 8),
    (2, 9),
    (7, 1),
    (7, 8),
    (8, 9),
    (8, 14),
    (9, 10),
    (10, 3),
    (11, 12),
    (11, 17),
    (13, 7),
    (14, 8),
    (14, 13),
    (14, 19),
    (15, 14),
    (16, 10),
    (16, 15),
    (17, 16),
    (17, 22),
    (18, 17),
    (19, 13),
    (21, 16),
    (22, 17),
    """

    disallowed_sequences = {
        ('C1', 9, 2): 2,
        ('C2', 14, 8): 8,
        ('C3', 17, 22): 22,
    }
    
    parking_capacities = {
        "P1": 50,
        "P2": 100,
        "P3": 100
    }

    return intersection_nodes, edge_list, disallowed_sequences

intersection_nodes, edge_list, disallowed_sequences = bgc_layout()
tm = TrafficManager(intersection_nodes, edge_list, 
                    disallowed_sequences=disallowed_sequences, 
                    default_edge_capacity=10)

"""
Draw cars in the grid, and assign their origin and destination
"""
number_of_cars = 200
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
                edge_choice = list(random.choice(list(tm.entry_edges)))
                origin = edge_choice[0]
                # next_immediate_destination = edge_choice[1]
                
                entry_nodes = list(tm.entry_nodes)
                # print(f"Entry nodes {entry_nodes}")
                # print(f"Origin {origin}")

                entry_nodes.remove(origin)

                final_destination = random.choice(entry_nodes)

                each_car.spawn(origin, final_destination)

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
        # Create a ThreadPoolExecutor with the desired number of threads
        with ThreadPoolExecutor(max_workers=64) as executor:  # You can adjust max_workers based on the number of cars and available resources
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