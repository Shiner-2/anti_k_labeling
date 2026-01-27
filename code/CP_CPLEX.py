import glob
import os
import time
import logging
import pandas as pd
from typing import Optional
from docplex.cp.model import CpoModel


# ================= CONFIG =================
name = "DE_CP_4cores"
LOG_FILE = f"logs/DE/log_{name}.txt"
EXCEL_FILE = f"output/DE/output_{name}.xlsx"

TIME_LIMIT_DEFAULT = 1800
MAX_WORKERS = 4
# ==========================================


# ================= LOGGING =================
def setup_logger(name="cp", log_file=LOG_FILE):
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


# ================= CP SOLVER ===============
def solve_cp(graph, k, lb, ub, timeout_sec):
    n = len(graph)
    mdl = CpoModel(name="AntiKLabeling_MaxWidth")

    # -------- variables --------
    label = {
        i: mdl.integer_var(1, k, name=f"label_{i}")
        for i in range(1, n + 1)
    }

    width = mdl.integer_var(lb, ub, name="width")

    # -------- no-hole --------
    for l in range(1, k + 1):
        mdl.add(mdl.count(label.values(), l) >= 1)

    # -------- anti-k-labeling --------
    for u in graph:
        for v in graph[u]:
            mdl.add(mdl.abs(label[u] - label[v]) >= width)

    # -------- symmetry breaking --------
    cnt = {i: 0 for i in range(1, n + 1)}
    for u in graph:
        for v in graph[u]:
            cnt[u] += 1
            cnt[v] += 1

    node = min(cnt, key=lambda x: cnt[x])
    mdl.add(label[node] <= k // 2)

    # -------- objective --------
    mdl.maximize(width)

    # -------- solve --------
    sol = mdl.solve(
        TimeLimit=timeout_sec,
        Workers=MAX_WORKERS,
        LogVerbosity="Quiet"
    )

    if sol:
        return True, sol[width]
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
    ok, best_width = solve_cp(graph, k, lb, ub, time_limit)
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

    if ok:
        return best_width
    return -9999
# ==========================================


# ================= MAIN ====================
def solve():
    logger = setup_logger()

    # clear files
    open(LOG_FILE, "w").close()
    if os.path.exists(EXCEL_FILE):
        os.remove(EXCEL_FILE)

    folder_path = "data/hb"
    files = glob.glob(f"{folder_path}/*")

    for file_path in files:
        t0 = time.time()

        graph, k, lb, ub = read_input(file_path)
        fname = os.path.basename(file_path)

        ans = solve_for_ans(
            graph,
            k,
            lb,
            ub,
            fname,
            TIME_LIMIT_DEFAULT
        )

        logger.info("$$$$")
        logger.info(f"{fname} → best width = {ans}")
        logger.info("$$$$")
        logger.info(
            f"Time total: {round(time.time() - t0, 2)}s"
        )


if __name__ == "__main__":
    solve()
