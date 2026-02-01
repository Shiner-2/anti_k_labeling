import glob
import os
import time
import logging
import pandas as pd
from typing import Optional
import gurobipy as gp
from gurobipy import GRB

# ================= CONFIG =================
name = "Gurobi"
LOG_FILE = f"logs/Gurobi/log_{name}.txt"
EXCEL_FILE = f"output/Gurobi/output_{name}.xlsx"

TIME_LIMIT_DEFAULT = 1800
# ==========================================

# ================= LOGGING =================
def setup_logger(name="gurobi", log_file=LOG_FILE):
    logger = logging.getLogger(name)
    if getattr(logger, "_configured", False):
        return logger

    logger.setLevel(logging.INFO)
    logger.propagate = False

    for h in list(logger.handlers):
        logger.removeHandler(h)

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    logger._configured = True
    return logger
# ==========================================


# ================= INPUT ===================
def read_input(file_path):
    graph = {}
    with open(file_path, 'r') as f:
        n, e, k, lb, ub = map(int, f.readline().split())
        for i in range(1, n + 1):
            graph[i] = []
        for _ in range(e):
            u, v = map(int, f.readline().split())
            graph[u].append(v)
    return graph, k, lb, ub
# ==========================================


# ================= GUROBI SOLVER ===========
def solve_gurobi(graph, k, lb, ub, timeout_sec):
    n = len(graph)
    M = k*2  # Big-M

    model = gp.Model("Gurobi")
    model.Params.TimeLimit = timeout_sec

    # -------- variables --------
    label = model.addVars(
        range(1, n + 1),
        vtype=GRB.INTEGER,
        lb=1,
        ub=k,
        name="label"
    )

    width = model.addVar(
        vtype=GRB.INTEGER,
        lb=lb,
        ub=ub,
        name="width"
    )

    # y[i,l] = 1 if node i uses label l
    y = model.addVars(
        range(1, n + 1),
        range(1, k + 1),
        vtype=GRB.BINARY,
        name="y"
    )

    # -------- link label & y --------
    for i in range(1, n + 1):
        model.addConstr(
            gp.quicksum(l * y[i, l] for l in range(1, k + 1)) == label[i]
        )
        model.addConstr(
            gp.quicksum(y[i, l] for l in range(1, k + 1)) == 1
        )

    # -------- no-hole --------
    for l in range(1, k + 1):
        model.addConstr(
            gp.quicksum(y[i, l] for i in range(1, n + 1)) >= 1
        )

    # -------- anti-k-labeling --------
    added = set()

    for u in graph:
        for v in graph[u]:
            if (v, u) in added:
                continue
            added.add((u, v))

            b = model.addVar(vtype=GRB.BINARY)
            model.addConstr(label[u] - label[v] >= width - M * (1 - b))
            model.addConstr(label[v] - label[u] >= width - M * b)

    # # -------- symmetry breaking --------
    deg = {i: 0 for i in range(1, n + 1)}
    for u in graph:
        for v in graph[u]:
            deg[u] += 1
            deg[v] += 1
    node = min(deg, key=lambda x: deg[x])
    model.addConstr(label[node] <= k // 2)

    # -------- objective --------
    model.setObjective(width, GRB.MAXIMIZE)

    model.optimize()

    if model.SolCount > 0:
        return True, int(width.X)
    return False, None
# ==========================================


# ================= PIPELINE ================
res = [["filename", "n", "k", "lb", "ub",
        "best_width", "verdict", "time"]]


def write_to_excel(data, mode='write'):
    logger = setup_logger()
    df = pd.DataFrame(data)
    if mode == 'append' and os.path.exists(EXCEL_FILE):
        old = pd.read_excel(EXCEL_FILE)
        df = pd.concat([old, df], ignore_index=True)
    df.to_excel(EXCEL_FILE, index=False)
    logger.info("Write Excel")


def solve_for_ans(graph, k, lb, ub, filename, time_limit):
    logger = setup_logger()

    t0 = time.time()
    ok, best_width = solve_gurobi(graph, k, lb, ub, time_limit)
    elapsed = round(time.time() - t0, 2)

    res.append([
        filename,
        len(graph),
        k,
        lb,
        ub,
        best_width if ok else None,
        ok,
        elapsed
    ])

    write_to_excel(
        res if len(res) == 2 else [res[-1]],
        mode='write' if len(res) == 2 else 'append'
    )

    return best_width if ok else -9999
# ==========================================


# ================= MAIN ====================
def solve():
    logger = setup_logger()

    open(LOG_FILE, "w").close()
    if os.path.exists(EXCEL_FILE):
        os.remove(EXCEL_FILE)

    folder_path = "data/hb"
    files = glob.glob(f"{folder_path}/*")

    for file_path in files:
        print("solving ", file_path)
        t0 = time.time()

        graph, k, lb, ub = read_input(file_path)
        fname = os.path.basename(file_path)

        ans = solve_for_ans(
            graph, k, lb, ub, fname, TIME_LIMIT_DEFAULT
        )

        logger.info("$$$$")
        logger.info(f"{fname} → best width = {ans}")
        logger.info("$$$$")
        logger.info(f"Time total: {round(time.time() - t0, 2)}s")


if __name__ == "__main__":
    solve()
