import gurobipy as gp
from gurobipy import GRB

m = gp.Model("lp_example")
# Create variables
X1 = m.addVar(name="x1")
X2 = m.addVar(name="x2")
# Set objective
m.setObjective(2 * X1 + X2, GRB.MAXIMIZE)
# Add constraints
m.addConstr(5*X2<=15, "c0")
m.addConstr(6*X1+2*X2<=24, "c1")
m.addConstr(X1+X2<=5, "c2")
m.addConstr(X1>=0, "c3")
m.addConstr(X2>=0, "c4") 
# Optimize model
m.optimize()

if m.status == GRB.OPTIMAL:
    print(f"Optimal objective value: {m.objVal}")
    for v in m.getVars():
        print(f"{v.varName}: {v.x}")