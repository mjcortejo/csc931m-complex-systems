import numpy as np
class Logger():
    def __init__(self, **kwargs):
        self.historical_edge_data = {}
        self.current_edge_data = {}
        self.time_step = 0
        self.time_out = 100
        self.overall_wait_avg_data = []

    def setup_edge_logs(self, edges: object):
        """Accepts list of edges generated from the TrafficManager Class.

        Stores edge logs as a tuple of (time_step, cars_occupied, max_capacity)

        Args:
            edges (object): _description_
        """
        for each_edge in edges.keys():
            self.historical_edge_data[each_edge] = []

    def log(self, edge, log):
        """_summary_

        Args:
            edge (int, int): _description_
            log (tuple(int, int, int, int)): _description_
        """
        self.historical_edge_data[edge].append(log)
        self.current_edge_data[edge] = log

    def compute_overall_wait_avg(self):
        overall_wait_average = np.mean([data[3] for _,data in self.current_edge_data.items()]) if self.current_edge_data else 0.0
        self.overall_wait_avg_data.append(overall_wait_average)

    def step_time(self):
        self.time_step += 1