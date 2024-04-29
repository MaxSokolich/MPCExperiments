from gurobipy import Model, GRB, quicksum

# Create a new model
m = Model("quadratic")

# Create variables
x = m.addVar(name="x")
y = m.addVar(name="y")

# Set the objective
m.setObjective(x*x + x*y + y*y, GRB.MINIMIZE)

# Add constraints
m.addConstr(x + 2*y >= 2, "c0")
m.addConstr(x + y >= 1, "c1")

# Optimize model
m.optimize()

# Print the optimal solutions for x and y
if m.status == GRB.OPTIMAL:
    print('Optimal x:', x.X)
    print('Optimal y:', y.X)
    print('Minimum objective value:', m.objVal)
else:
    print('No optimal solution found.')