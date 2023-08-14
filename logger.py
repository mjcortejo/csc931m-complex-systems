import numpy as np
class Logger():
    def __init__(self, **kwargs):
        self.historical_edge_data = {}
        self.current_edge_data = {}
        self.time_step = 0
        self.time_out = 100
        self.overall_wait_avg_data = []
        self.overall_edge_volume_avg_data = []
        
        self.overall_outflow_data = []
        self.car_holding_time_distribution_data = []

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
            log (tuple(int, int, int, int)): Tuple contains the log data for each edge per time step (time step, cars occupied, edge max capacity, average waiting time of cars)
        """
        self.historical_edge_data[edge].append(log)
        self.current_edge_data[edge] = log

    def log_outflow(self, cars_exited):
        self.overall_outflow_data.append(cars_exited)

    def log_holding_time(self, car_holding_time):
        self.car_holding_time_distribution_data.append(car_holding_time)

    def compute_overall_wait_avg(self):
        overall_wait_average = np.mean([data[3] for _,data in self.current_edge_data.items()]) if self.current_edge_data else 0.0
        self.overall_wait_avg_data.append(overall_wait_average)

        overall_edge_volume_average = np.mean([(data[1] / data[2]) for _, data in self.current_edge_data.items()]) if self.current_edge_data else 0.0
        self.overall_edge_volume_avg_data.append(overall_edge_volume_average)


    def step_time(self):
        self.time_step += 1