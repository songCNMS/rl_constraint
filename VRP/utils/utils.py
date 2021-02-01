import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import re
import math

from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

from ortools.algorithms import pywrapknapsack_solver



class KnapsackProblem(object):
    def __init__(self, problem_name):
        self.problem_name = problem_name
        self.values = None
        self.weights = None
        self.capacity = None
        self.opt_value = 0.0
        self.load_problem(problem_name)

    def load_problem(self, problem_name):
        with open(f"data/Knapsack/{problem_name}/{problem_name}_c.txt", 'r') as f:
            self.capacity = []
            for line in f.readlines():
                self.capacity.append(int(line.strip()))
        with open(f"data/Knapsack/{problem_name}/{problem_name}_w.txt", 'r') as f:
            self.weights = []
            for line in f.readlines():
                self.weights.append(int(line.strip()))
            self.weights = [self.weights]
        with open(f"data/Knapsack/{problem_name}/{problem_name}_p.txt", 'r') as f:
            self.values = []
            for line in f.readlines():
                self.values.append(int(line.strip()))
        with open(f"data/Knapsack/{problem_name}/{problem_name}_s.txt", 'r') as f:
            for i, line in enumerate(f.readlines()):
                if line.strip() == '1':
                    self.opt_value += self.values[i]

    def get_opt_value(self):
        return self.opt_value
    
    def get_ortool_value(self):
        # Create the solver.
        data = self.get_ortool_model()
        solver = pywrapknapsack_solver.KnapsackSolver(
            pywrapknapsack_solver.KnapsackSolver.
            KNAPSACK_MULTIDIMENSION_BRANCH_AND_BOUND_SOLVER, 'KnapsackExample')

        values = data['values']
        weights = data['weights']
        capacities = data['capacity']

        solver.Init(values, weights, capacities)
        computed_value = solver.Solve()

        packed_items = []
        packed_weights = []
        total_weight = 0
        print('Total value =', computed_value)
        for i in range(len(values)):
            if solver.BestSolutionContains(i):
                packed_items.append(i)
                packed_weights.append(weights[0][i])
                total_weight += weights[0][i]
        print('Total weight:', total_weight)
        print('Packed items:', packed_items)
        print('Packed_weights:', packed_weights)
        return computed_value

    def get_ortool_model(self):
        data = {"values": self.values, "weights": self.weights, "capacity": self.capacity}
        return data


class VRProblem(object):
    def __init__(self, problem_name):
        self.problem_name = problem_name
        self.num_trucks = 0
        self.optimal_value = 0
        self.num_nodes = 0
        self.vehicle_capacity = 0
        self.node_pos_x = []
        self.node_pos_y = []
        self.node_demand = []
        self.depots = []
        self.ortool_opt_value = None
        self.load_problem(problem_name)

    def load_problem(self, problem_name):
        with open(f"data/VRP/{problem_name}.vrp", 'r') as f:
            lines = f.readlines()
            try:
                self.num_trucks = int(lines[1].split(':')[2].split(',')[0].strip())
            except:
                self.num_trucks = 0
            self.num_nodes = int(lines[3].split(':')[1].strip())
            self.vehicle_capacity = int(lines[5].split(':')[1].strip())
            for i in range(7, 7+self.num_nodes):
                feilds = re.split(' |\t', lines[i].strip())
                x, y = float(feilds[1].strip()), float(feilds[2].strip())
                self.node_pos_x.append(x)
                self.node_pos_y.append(y)
            for i in range(7+self.num_nodes+1, 7+2*self.num_nodes+1):
                feilds = re.split(' |\t', lines[i].strip())
                self.node_demand.append(int(feilds[-1].strip()))
            i = 7+2*self.num_nodes+2
            while(True):
                _depot = int(lines[i].strip())
                if _depot == -1:
                    break
                self.depots.append(_depot)
                i += 1
    
    def get_nodes_coordinates(self):
        distance_matrix = np.zeros((self.num_nodes, self.num_nodes))
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                distance_matrix[i][j] = math.sqrt(math.pow(self.node_pos_x[i]-self.node_pos_x[j], 2) + math.pow(self.node_pos_y[i]-self.node_pos_y[j], 2))        
        return distance_matrix

    def get_ortool_model(self):
        data = {}
        data['distance_matrix'] = self.get_nodes_coordinates()
        # [START demands_capacities]
        data['demands'] = self.node_demand
        data['vehicle_capacities'] = [self.vehicle_capacity] * self.num_trucks
        # [END demands_capacities]
        data['num_vehicles'] = self.num_trucks
        data['depot'] = self.depots[0] - 1
        return data

    def ortool_solve(self):
        """Solve the CVRP problem."""
        # Instantiate the data problem.
        # Create the routing index manager.
        # [START index_manager]
        data = self.get_ortool_model()
        manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']),
                                            data['num_vehicles'], data['depot'])
        # [END index_manager]

        # Create Routing Model.
        # [START routing_model]
        routing = pywrapcp.RoutingModel(manager)

        # [END routing_model]

        # Create and register a transit callback.
        # [START transit_callback]
        def distance_callback(from_index, to_index):
            """Returns the distance between the two nodes."""
            # Convert from routing variable Index to distance matrix NodeIndex.
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return data['distance_matrix'][from_node][to_node]

        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        # [END transit_callback]

        # Define cost of each arc.
        # [START arc_cost]
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        # [END arc_cost]

        # Add Capacity constraint.
        # [START capacity_constraint]
        def demand_callback(from_index):
            """Returns the demand of the node."""
            # Convert from routing variable Index to demands NodeIndex.
            from_node = manager.IndexToNode(from_index)
            return data['demands'][from_node]

        demand_callback_index = routing.RegisterUnaryTransitCallback(
            demand_callback)
        routing.AddDimensionWithVehicleCapacity(
            demand_callback_index,
            0,  # null capacity slack
            data['vehicle_capacities'],  # vehicle maximum capacities
            True,  # start cumul to zero
            'Capacity')
        # [END capacity_constraint]

        # Setting first solution heuristic.
        # [START parameters]
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
        search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
        search_parameters.time_limit.FromSeconds(1)
        # [END parameters]

        # Solve the problem.
        # [START solve]
        solution = routing.SolveWithParameters(search_parameters)
        return data, manager, routing, solution

    def get_ortool_opt_val(self):
        if self.ortool_opt_value is None:
            try:
                data, manager, routing, solution = self.ortool_solve()
                total_distance = 0
                total_load = 0
                for vehicle_id in range(data['num_vehicles']):
                    index = routing.Start(vehicle_id)
                    route_distance = 0
                    route_load = 0
                    while not routing.IsEnd(index):
                        node_index = manager.IndexToNode(index)
                        route_load += data['demands'][node_index]
                        previous_index = index
                        index = solution.Value(routing.NextVar(index))
                        route_distance += routing.GetArcCostForVehicle(
                            previous_index, index, vehicle_id)
                    total_distance += route_distance
                    total_load += route_load
                self.ortool_opt_value = total_distance
            except:
                self.ortool_opt_value = -1.0
        return self.ortool_opt_value


def list_to_figure(results, labels, caption, loc_path, smoothed=False):
    num_result = len(results)
    smooth_steps = 100
    if num_result <= 0:
        return
    num_points = np.max([len(results[i]) for i in range(num_result)])
    for i in range(num_result):
        if len(results[i]) < num_points:
            results[i].extend([results[i][-1]]*(num_points-len(results[i])))

    if smoothed and num_points > smooth_steps:
        smooth_gap = num_points - smooth_steps + 1
        num_points = smooth_steps
        for i in range(num_result):
            new_result = [np.median(results[i][j:j+smooth_gap]) for j in range(smooth_steps)]
            results[i] = new_result

    x = np.linspace(0, num_points, num_points)                                         
    fig = plt.figure(figsize=(10,7))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_facecolor((234/255, 234/255,242/255 ))
    plt.grid(True, color='white')
    plt.title(caption)
    for y_label, y in zip(labels, results):
        plt.plot(x, y, label=y_label )
    plt.legend(loc='best')
    plt.savefig(loc_path)
    plt.close()

if __name__ == '__main__':
    vrp_problem = VRProblem('A-n32-k5')
    print(vrp_problem.get_ortool_model())