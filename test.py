from gurobipy import Model, GRB

# Create a new model
m = Model("quadratic")

# Create variables
x = m.addVar(name="x")
y = m.addVar(name="y")

# Set objective
m.setObjective(x*x + x*y + y*y + 3*x + 4*y, GRB.MINIMIZE)

# Add constraints
m.addConstr(x + 2*y >= 2, "c0")
m.addConstr(x + y >= 3, "c1")

# Optimize model
m.optimize()

# Print results
if m.status == GRB.OPTIMAL:
    print(f'Optimal solution found: x = {x.X}, y = {y.X}')
else:
    print("No optimal solution found.")
