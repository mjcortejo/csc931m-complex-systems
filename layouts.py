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
        ('P1', 'C1', 5), (2, 'C1', 3) , ('C1', 9, 4), (9, 'C1', 4), ('C1', 2, 3), #Three Parkade
        ('P2', 'C2', 5), ('C2', 8, 8), (14, 'C2', 2), #Uptown Mall Parking
        ('P3', 'C3', 5), (22, 'C3', 3), ('C3', 17, 4), #Gallery Parkade
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

def test_layout():
    intersection_nodes = {
        'E1': (100, 100),
        'E2': (200, 300),
        'P1': (300, 100),
        'C1': (300, 200),
        1: (100, 200),
        2: (0, 200),
        3: (100, 300),
        4: (200, 100),
        5: (200, 200),
    }

    edge_list = [
        ('E1', 1, 10), ('E2', 5, 10),
        (1, 2, 10), (1, 3, 10), (1, 5, 10),
        (5, 1, 10), (5, 4, 10),(5, 'C1', 10),
        ('C1', 5, 10), ('P1', 'C1', 10)
    ]

    return intersection_nodes, edge_list