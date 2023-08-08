class Logger():
    def __init__(self, **kwargs):
        self.edge_volume = {}
        self.time_step = 0
        self.time_out = 100

    def setup_edge_logs(self, edges: object):
        """Accepts list of edges generated from the TrafficManager Class.

        Stores edge logs as a tuple of (time_step, cars_occupied, max_capacity)

        Args:
            edges (object): _description_
        """
        for each_edge in edges.keys():
            self.edge_volume[each_edge] = []

    def step_time(self):
        self.time_step += 1