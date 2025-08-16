import pulp

# Define the problem
model = pulp.LpProblem("Product_Mix_Optimization", pulp.LpMaximize)

# Decision variables
xA = pulp.LpVariable('Product_A', lowBound=0, cat='Continuous')
xB = pulp.LpVariable('Product_B', lowBound=0, cat='Continuous')

# Objective function: Maximize profit
model += 40 * xA + 50 * xB, "Total_Profit"

# Constraints
model += 3 * xA + 4 * xB <= 120, "Machine_Hours"
model += 2 * xA + 3 * xB <= 90, "Labor_Hours"

# Solve
model.solve()

# Output results
print(f"Status: {pulp.LpStatus[model.status]}")
print(f"Optimal units of Product A: {xA.varValue}")
print(f"Optimal units of Product B: {xB.varValue}")
print(f"Maximum Profit: ${pulp.value(model.objective)}")