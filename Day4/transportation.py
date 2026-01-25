import gurobipy as gp
from gurobipy import GRB

T = [1, 2, 3, 4]
cap = {1: 25, 2: 35, 3: 30, 4: 20}          # production capacity
d   = {1: 15, 2: 20, 3: 25, 4: 20}          # delivery demand (must meet by quarter end)
c   = {1: 12.0, 2: 11.0, 3: 11.5, 4: 12.5}  # unit production cost (10k RMB)
h = 0.1                                     # holding cost per unit per quarter (10k RMB)


m = gp.Model("quarterly_production_plan")

# Decision variables (integer, typical for "units")
p = m.addVars(T, lb=0, vtype=GRB.INTEGER, name="p")     # production
I = m.addVars(T, lb=0, vtype=GRB.INTEGER, name="I")     # end-of-quarter inventory

# Objective: production cost + holding cost
m.setObjective(
    gp.quicksum(c[t] * p[t] + h * I[t] for t in T),
    GRB.MINIMIZE
)

# Capacity constraints
for t in T:
    m.addConstr(p[t] <= cap[t], name=f"cap_{t}")

# Inventory balance (I0 = 0)
m.addConstr(I[1] == p[1] - d[1], name="bal_1")
for t in T[1:]:
    m.addConstr(I[t] == I[t-1] + p[t] - d[t], name=f"bal_{t}")

m.addConstr(I[4] == 0, name="end_inventory")

m.optimize()

# -------------------------
# Output
# -------------------------
if m.status == GRB.OPTIMAL:
    print("\nOptimal Plan (units):")
    prod_cost = 0.0
    hold_cost = 0.0
    inv_prev = 0

    for t in T:
        pt = p[t].X
        It = I[t].X
        prod_cost += c[t] * pt
        hold_cost += h * It

        # delivered in quarter t is fixed as d[t]
        print(f"Q{t}: produce={pt:>5.0f}, deliver={d[t]:>5d}, end_inv={It:>5.0f}")

    print("\nCost Breakdown (10k RMB):")
    print(f"  Production cost = {prod_cost:.2f}")
    print(f"  Holding cost    = {hold_cost:.2f}")
    print(f"  Total cost      = {m.ObjVal:.2f}")
else:
    print("No optimal solution found. Status:", m.status)
