import numpy as np
import gurobipy as grb
import random
from matplotlib import pyplot as plt

random.seed(10)


class Input:
    """ Generates the input data randomly based on set parameters
    """

    def __init__(self, n_customers, capacity, min_x, max_x, min_y, max_y, min_demand, max_demand, min_profit,
                 max_profit):
        self.n_customers = n_customers
        self.capacity = capacity
        self.min_x = min_x
        self.max_x = max_x
        self.min_y = min_y
        self.max_y = max_y
        self.min_demand = min_demand
        self.max_demand = max_demand
        self.min_profit = min_profit
        self.max_profit = max_profit

    def _generate_inputs(self):
        """ Generates customer coordinates, demand and profit randomly
            and calculates the distance based on euclidean distance, for each pair of customers
        """
        self.customers = [i for i in range(1, self.n_customers)]
        self.depot = 1
        self.coord_x = {}
        self.coord_y = {}
        self.demand = {}
        self.profit = {}
        for i in self.customers:
            self.coord_x[i] = random.uniform(self.min_x, self.max_x)
            self.coord_y[i] = random.uniform(self.min_y, self.max_y)
            self.demand[i] = int(random.uniform(self.min_demand, self.max_demand))
            self.profit[i] = int(random.uniform(self.min_profit, self.max_profit))
        self.dist_matrix = {}
        for i in self.customers:
            for j in self.customers:
                self.dist_matrix[(i, j)] = np.sqrt(
                    (self.coord_x[i] - self.coord_x[j]) ** 2 + (self.coord_y[i] - self.coord_x[j]) ** 2)


class ModelHierarquical:
    """ Build the mixed integer programming for the prize-collecting vehicle routing problem
        The Prize-Collecting Vehicle Routing can be stated as the following: given a graph (set of nodes and edges),
        build a trip schedule starting and ending at the same depot, making sure to:
            - Flow constraint is respected
            - Capacity of a vehicle is respected
            - If a customer is visited, its demand should be satisfied
        While maximizing the profit of visiting each customer (objective 1) and
              minimizing the total travel time (objective 2)
    """

    def __init__(self, InputData, max_profit_tol, min_tot_dist_tol):
        self.input_data = InputData
        self.max_profit_tol = max_profit_tol
        self.min_tot_dist_tol = min_tot_dist_tol
        self.opt_model = grb.Model(name="MIP Model")

    def _define_variables(self):
        """ Define model variables
        """
        """ Binary variable: 1 there's a link between node i and node j. 0 otherwise """
        self.x = {(i, j): self.opt_model.addVar(vtype=grb.GRB.BINARY, name="x_{0}_{1}".format(i, j))
                  for i in self.input_data.customers for j in self.input_data.customers if i != j}
        """ Integer variable: flow (demand) carried by vehicle while visiting customer i """
        self.y = {
            i: self.opt_model.addVar(vtype=grb.GRB.INTEGER, lb=self.input_data.demand[i], ub=self.input_data.capacity,
                                     name="y_{0}".format(i))
            for i in self.input_data.customers if i != self.input_data.depot}

    def _define_constraints(self):
        """ Define model constraints
        """
        """ Constraint 1 - Flow balance constraint. Number of inbound lanes is the same of outbound lanes, for each node
        """
        c1 = {j: self.opt_model.addLConstr(grb.quicksum(self.x[(i, j)] for i in self.input_data.customers if i != j) ==
                                           grb.quicksum(self.x[(j, i)] for i in self.input_data.customers if i != j),
                                           name="c1_{0}".format(j))
              for j in self.input_data.customers}

        """ Constraint 2 - Each customer must be visited maximum once """
        c2 = {
            j: self.opt_model.addLConstr(grb.quicksum(self.x[(i, j)] for i in self.input_data.customers if i != j) <= 1,
                                         name="c2_{0}".format(j))
            for j in self.input_data.customers if j != self.input_data.depot}

        """ Constraint 3 - Subtour elimination constraint - ensures model does not generate subtours """
        c3 = {(i, j): self.opt_model.addLConstr(
            self.y[j] - self.y[i] >= self.input_data.demand[j] - self.input_data.capacity * (1 - (self.x[i, j])),
            name="c3_{0}_{1}".format(i, j))
            for i in self.input_data.customers for j in self.input_data.customers if
            i != j and i != self.input_data.depot and j != self.input_data.depot}

        """ Constraint 4 - Number of arcs leaving the depot is equal to arcs going back to depot
        """
        c4 = self.opt_model.addLConstr(grb.quicksum(
            self.x[(self.input_data.depot, j)] for j in self.input_data.customers if j != self.input_data.depot) ==
                                       grb.quicksum(
                                           self.x[(i, self.input_data.depot)] for i in self.input_data.customers if
                                           i != self.input_data.depot), name="c4")

        """ Constraint 5 - Vehicles should leave the depot and at least one vehicle must be in the tour
        """
        c5 = self.opt_model.addLConstr(grb.quicksum(
            self.x[(self.input_data.depot, j)] for j in self.input_data.customers if j != self.input_data.depot) >= 1,
                                       name="c5")

    def _define_objective_function(self):
        """ Define objective function
        """
        """ Objective function 1 - maximize the profit of the visited customers """
        self.obj_max_profit = grb.quicksum(
            self.x[(i, j)] * self.input_data.profit[j]
            for i in self.input_data.customers for j in self.input_data.customers
            if i != j and j != self.input_data.depot)

        """ Objective function 2 - minimize total distance of the trip """
        self.obj_min_tot_dist = grb.quicksum(
            self.x[(i, j)] * self.input_data.dist_matrix[(i, j)]
            for i in self.input_data.customers for j in self.input_data.customers if i != j)

    def _set_objective_function(self):
        """ Set objective function - minimization
            1 - Maximize total profit
            2 - Minimize total travel time
        """
        """ Set of objectives 
            - objective: objective function variable name
            - priority: priority of each objective function - higher is first
            - relative tolerance: percentage of each objective function that can be degraded
            - weight: weight of each objective. < 1 is for maximization and > 1 for minimization
        """
        objs_dict = {1: {'objective': self.obj_max_profit, 'priority': 2, 'relative tolerance': self.max_profit_tol,
                         'weight': -1},
                     2: {'objective': self.obj_min_tot_dist, 'priority': 1, 'relative tolerance': self.min_tot_dist_tol,
                         'weight': 1}}
        self.opt_model.ModelSense = grb.GRB.MINIMIZE
        for i in objs_dict:
            self.opt_model.setObjectiveN(objs_dict[i]['objective'], index=i, priority=objs_dict[i]['priority'],
                                         reltol=objs_dict[i]['relative tolerance'], weight=objs_dict[i]['weight'])

    def _solve_model(self):
        """ Call routine classes
        """
        self._define_variables()
        self._define_constraints()
        self._define_objective_function()
        self._set_objective_function()
        self.opt_model.Params.LogToConsole = 0
        self.opt_model.optimize()
        self._get_objective_function_values()

    def _get_objective_function_values(self):
        """ Get objective function values
        """
        self.output_obj_min_tot_dist = 0
        self.output_obj_max_profit = 0
        for i in self.input_data.customers:
            for j in self.input_data.customers:
                if i != j:
                    self.output_obj_min_tot_dist += self.x[(i, j)].X * self.input_data.dist_matrix[(i, j)]
                    if j != self.input_data.depot:
                        self.output_obj_max_profit += self.x[(i, j)].X * self.input_data.profit[j]

    def _print_outputs(self):
        """ Display model outputs
        """

        visited = {}
        for i in self.input_data.customers:
            if i != self.input_data.depot:
                print(model.y[i].X)
        print("")
        for i in self.input_data.customers:
            for j in self.input_data.customers:
                if i != j:
                    if self.x[i, j].X > 0.1:
                        visited[j] = 1
                        print(i, j)
        print("")
        for i in self.input_data.customers:
            if i in visited:
                print(i, "is visited")
            else:
                print(i, "is not visited")


class ModelHierarquicalManual:
    """ Build the mixed integer programming for the prize-collecting vehicle routing problem
        The Prize-Collecting Vehicle Routing can be stated as the following: given a graph (set of nodes and edges),
        build a trip schedule starting and ending at the same depot, making sure to:
            - Flow constraint is respected
            - Capacity of a vehicle is respected
            - If a customer is visited, its demand should be satisfied
        While maximizing the profit of visiting each customer (objective 1) and
              minimizing the total travel time (objective 2)
    """

    def __init__(self, InputData, max_profit_tol, first_objective_value):
        self.input_data = InputData
        self.max_profit_tol = max_profit_tol
        self.first_objective_value = first_objective_value
        self.opt_model = grb.Model(name="MIP Model")

    def _define_variables(self):
        """ Define model variables
        """
        """ Binary variable: 1 there's a link between node i and node j. 0 otherwise """
        self.x = {(i, j): self.opt_model.addVar(vtype=grb.GRB.BINARY, name="x_{0}_{1}".format(i, j))
                  for i in self.input_data.customers for j in self.input_data.customers if i != j}
        """ Integer variable: flow (demand) carried by vehicle while visiting customer i """
        self.y = {
            i: self.opt_model.addVar(vtype=grb.GRB.INTEGER, lb=self.input_data.demand[i], ub=self.input_data.capacity,
                                     name="y_{0}".format(i))
            for i in self.input_data.customers if i != self.input_data.depot}

    def _define_constraints(self):
        """ Define model constraints
        """
        """ Constraint 1 - Flow balance constraint. Number of inbound lanes is the same of outbound lanes, for each node
        """
        c1 = {j: self.opt_model.addLConstr(grb.quicksum(self.x[(i, j)] for i in self.input_data.customers if i != j) ==
                                           grb.quicksum(self.x[(j, i)] for i in self.input_data.customers if i != j),
                                           name="c1_{0}".format(j))
              for j in self.input_data.customers}

        """ Constraint 2 - Each customer must be visited maximum once """
        c2 = {
            j: self.opt_model.addLConstr(grb.quicksum(self.x[(i, j)] for i in self.input_data.customers if i != j) <= 1,
                                         name="c2_{0}".format(j))
            for j in self.input_data.customers if j != self.input_data.depot}

        """ Constraint 3 - Subtour elimination constraint - ensures model does not generate subtours """
        c3 = {(i, j): self.opt_model.addLConstr(
            self.y[j] - self.y[i] >= self.input_data.demand[j] - self.input_data.capacity * (1 - (self.x[i, j])),
            name="c3_{0}_{1}".format(i, j))
            for i in self.input_data.customers for j in self.input_data.customers if
            i != j and i != self.input_data.depot and j != self.input_data.depot}

        """ Constraint 4 - Flow balance constraint. Number of inbound lanes is the same of outbound lanes, for each node
        """
        c4 = self.opt_model.addLConstr(grb.quicksum(
            self.x[(self.input_data.depot, j)] for j in self.input_data.customers if j != self.input_data.depot) ==
                                       grb.quicksum(
                                           self.x[(i, self.input_data.depot)] for i in self.input_data.customers if
                                           i != self.input_data.depot), name="c4")

        """ Constraint 5 - Vehicles should leave the depot and at least one vehicle must be in the tour
        """
        c5 = self.opt_model.addLConstr(grb.quicksum(
            self.x[(self.input_data.depot, j)] for j in self.input_data.customers if j != self.input_data.depot) >= 1,
                                       name="c5")

        """ Constraint 6 - Vehicles should leave the depot and at least one vehicle must be in the tour
        """
        c6 = self.opt_model.addLConstr(grb.quicksum(
            self.x[(i, j)] * self.input_data.profit[j] for i in self.input_data.customers for j in
            self.input_data.customers if i != j and j != self.input_data.depot) >=
                                       (1 - self.max_profit_tol) * self.first_objective_value,
                                       name="c5")

    def _define_objective_function(self):
        """ Define objective function
        """
        """ Objective function 2 - minimize total distance of the trip """
        self.obj_min_tot_dist = grb.quicksum(
            self.x[(i, j)] * self.input_data.dist_matrix[(i, j)]
            for i in self.input_data.customers for j in self.input_data.customers if i != j)

    def _set_objective_function(self):
        """ Set objective function - minimization
            1 - Minimize total travel time
        """
        self.opt_model.ModelSense = grb.GRB.MINIMIZE
        self.opt_model.setObjective(self.obj_min_tot_dist)

    def _solve_model(self):
        """ Call routine classes
        """
        self._define_variables()
        self._define_constraints()
        self._define_objective_function()
        self._set_objective_function()
        self.opt_model.Params.LogToConsole = 0
        self.opt_model.optimize()
        self._get_objective_function_values()

    def _get_objective_function_values(self):
        """ Get objective function values
        """
        self.output_obj_min_tot_dist = 0
        self.output_obj_max_profit = 0
        for i in self.input_data.customers:
            for j in self.input_data.customers:
                if i != j:
                    self.output_obj_min_tot_dist += self.x[(i, j)].X * self.input_data.dist_matrix[(i, j)]
                    if j != self.input_data.depot:
                        self.output_obj_max_profit += self.x[(i, j)].X * self.input_data.profit[j]

    def _print_outputs(self):
        """ Display model outputs
        """

        visited = {}
        for i in self.input_data.customers:
            if i != self.input_data.depot:
                print(model.y[i].X)
        print("")
        for i in self.input_data.customers:
            for j in self.input_data.customers:
                if i != j:
                    if self.x[i, j].X > 0.1:
                        visited[j] = 1
                        print(i, j)
        print("")
        for i in self.input_data.customers:
            if i in visited:
                print(i, "is visited")
            else:
                print(i, "is not visited")


class ModelSingleObj:
    """ Build the mixed integer programming for the prize-collecting vehicle routing problem
        The Prize-Collecting Vehicle Routing can be stated as the following: given a graph (set of nodes and edges),
        build a trip schedule starting and ending at the same depot, making sure to:
            - Flow constraint is respected
            - Capacity of a vehicle is respected
            - If a customer is visited, its demand should be satisfied
        While maximizing the profit of visiting each customer (objective 1) and
              minimizing the total travel time (objective 2)
    """

    def __init__(self, InputData):
        self.input_data = InputData
        self.opt_model = grb.Model(name="MIP Model")

    def _define_variables(self):
        """ Define model variables
        """
        """ Binary variable: 1 there's a link between node i and node j. 0 otherwise """
        self.x = {(i, j): self.opt_model.addVar(vtype=grb.GRB.BINARY, name="x_{0}_{1}".format(i, j))
                  for i in self.input_data.customers for j in self.input_data.customers if i != j}
        """ Integer variable: flow (demand) carried by vehicle while visiting customer i """
        self.y = {
            i: self.opt_model.addVar(vtype=grb.GRB.INTEGER, lb=self.input_data.demand[i], ub=self.input_data.capacity,
                                     name="y_{0}".format(i))
            for i in self.input_data.customers if i != self.input_data.depot}

    def _define_constraints(self):
        """ Define model constraints
        """
        """ Constraint 1 - Flow balance constraint. Number of inbound lanes is the same of outbound lanes, for each node
        """
        c1 = {j: self.opt_model.addLConstr(grb.quicksum(self.x[(i, j)] for i in self.input_data.customers if i != j) ==
                                           grb.quicksum(self.x[(j, i)] for i in self.input_data.customers if i != j),
                                           name="c1_{0}".format(j))
              for j in self.input_data.customers}

        """ Constraint 2 - Each customer must be visited maximum once """
        c2 = {
            j: self.opt_model.addLConstr(grb.quicksum(self.x[(i, j)] for i in self.input_data.customers if i != j) <= 1,
                                         name="c2_{0}".format(j))
            for j in self.input_data.customers if j != self.input_data.depot}

        """ Constraint 3 - Subtour elimination constraint - ensures model does not generate subtours """
        c3 = {(i, j): self.opt_model.addLConstr(
            self.y[j] - self.y[i] >= self.input_data.demand[j] - self.input_data.capacity * (1 - (self.x[i, j])),
            name="c3_{0}_{1}".format(i, j))
            for i in self.input_data.customers for j in self.input_data.customers if
            i != j and i != self.input_data.depot and j != self.input_data.depot}

        """ Constraint 4 - Flow balance constraint. Number of inbound lanes is the same of outbound lanes, for each node
        """
        c4 = self.opt_model.addLConstr(grb.quicksum(
            self.x[(self.input_data.depot, j)] for j in self.input_data.customers if j != self.input_data.depot) ==
                                       grb.quicksum(
                                           self.x[(i, self.input_data.depot)] for i in self.input_data.customers if
                                           i != self.input_data.depot), name="c4")

        """ Constraint 5 - Vehicles should leave the depot and at least one vehicle must be in the tour
        """
        c5 = self.opt_model.addLConstr(grb.quicksum(
            self.x[(self.input_data.depot, j)] for j in self.input_data.customers if j != self.input_data.depot) >= 1,
                                       name="c5")

    def _define_objective_function(self):
        """ Define objective function
        """
        """ Objective function 1 - maximize the profit of the visited customers """
        self.obj_max_profit = grb.quicksum(
            self.x[(i, j)] * self.input_data.profit[j]
            for i in self.input_data.customers for j in self.input_data.customers
            if i != j and j != self.input_data.depot)

    def _set_objective_function(self):
        """ Set objective function - minimization
            1 - Maximize total profit
        """
        self.opt_model.ModelSense = grb.GRB.MAXIMIZE
        self.opt_model.setObjective(self.obj_max_profit)

    def _solve_model(self):
        """ Call routine classes
        """
        self._define_variables()
        self._define_constraints()
        self._define_objective_function()
        self._set_objective_function()
        self.opt_model.Params.LogToConsole = 0
        self.opt_model.optimize()
        self._get_objective_function_values()

    def _get_objective_function_values(self):
        """ Get objective function values
        """
        self.output_obj_min_tot_dist = 0
        self.output_obj_max_profit = 0
        for i in self.input_data.customers:
            for j in self.input_data.customers:
                if i != j:
                    self.output_obj_min_tot_dist += self.x[(i, j)].X * self.input_data.dist_matrix[(i, j)]
                    if j != self.input_data.depot:
                        self.output_obj_max_profit += self.x[(i, j)].X * self.input_data.profit[j]

    def _print_outputs(self):
        """ Display model outputs
        """

        visited = {}
        for i in self.input_data.customers:
            if i != self.input_data.depot:
                print(model.y[i].X)
        print("")
        for i in self.input_data.customers:
            for j in self.input_data.customers:
                if i != j:
                    if self.x[i, j].X > 0.1:
                        visited[j] = 1
                        print(i, j)
        print("")
        for i in self.input_data.customers:
            if i in visited:
                print(i, "is visited")
            else:
                print(i, "is not visited")


if __name__ == '__main__':

    n_customers, capacity, min_x, max_x, min_y, max_y, min_demand, max_demand, min_profit, max_profit = \
        40, 1000, -10, 10, -10, 10, 1, 20, 1, 100
    input_data = Input(n_customers, capacity, min_x, max_x, min_y, max_y, min_demand, max_demand, min_profit,
                       max_profit)
    input_data._generate_inputs()

    model_single = ModelSingleObj(input_data)
    model_single._solve_model()
    initial_max_profit = model_single.output_obj_max_profit
    for tol in range(0, 10):
        model_hierarquical = ModelHierarquicalManual(input_data, tol / 10, initial_max_profit)
        model_hierarquical._solve_model()
        print(model_hierarquical.output_obj_max_profit, model_hierarquical.output_obj_min_tot_dist)

    print("")
    max_profit_tol, min_tot_dist_tol = 0.0, 0.1
    f1_vals = []
    f2_vals = []
    for tol in range(0, 100, 5):
        model = ModelHierarquical(input_data, tol / 100, min_tot_dist_tol)
        model._solve_model()
        f1_vals.append(model.output_obj_max_profit)
        f2_vals.append(model.output_obj_min_tot_dist)
        print(model.output_obj_max_profit, model.output_obj_min_tot_dist)

plt.xlabel("Distance")
plt.ylabel("Profit")
plt.title("Pareto Front")
plt.plot(f2_vals, f1_vals, marker='o')
plt.legend()
plt.show()