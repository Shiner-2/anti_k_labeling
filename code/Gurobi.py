# -*- coding: utf-8 -*-
"""
AKL feasibility (No-hole) — Gurobi port
- Tự động dùng license gurobi.lic trong cùng thư mục nếu GRB_LICENSE_FILE chưa được set
- Mirror lại khung CP: x[i,l], L/R aux, SCL_AMO, ExactlyOne endpoints, No-hole, Anti-width
- Giữ nguyên driver: tuantu_for_ans / binary_search_for_ans + Excel output
"""

import os, sys, time, glob, multiprocessing, pandas as pd
import gurobipy as gp
from gurobipy import GRB

# =========================
# License: dùng gurobi.lic trong cùng thư mục (nếu chưa có GRB_LICENSE_FILE)
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LIC_PATH = os.path.join(BASE_DIR, "gurobi.lic")
# Không ghi đè nếu người dùng đã set GRB_LICENSE_FILE ở ngoài
if "GRB_LICENSE_FILE" not in os.environ:
    if os.path.isfile(LIC_PATH):
        os.environ["GRB_LICENSE_FILE"] = LIC_PATH
    # else: để Gurobi tự tìm theo mặc định

# =========================
# Paths & Logs
# =========================
LOG_FILE = "logs/gurobi.txt"                 # giữ nguyên tên file log text như bản cũ
SOLVER_LOG_FILE = "logs/gurobi_AKL_solver.log" # log nội bộ của solver Gurobi
EXCEL_OUTPUT = "output/output_gurobi_1800.xlsx"

# =========================
# Core solver (Gurobi)
# =========================
def solve_no_hole_anti_k_labeling_gurobi(graph, k, width, queue, timelimit=None, threads=None):
    """
    Gurobi bám sát SAT/CP:
      - x[i,l] ∈ {0,1}
      - L_aux / R_aux ∈ {0,1}
      - SCL_AMO (9)–(13) tuyến tính trên 0/1
      - ExactlyOne endpoints
      - No-hole: sum_i x[i,l] ≥ 1  (∀ l=1..k)
      - Anti-width bằng bất đẳng thức kiểu a + b ≤ 1

    Trả về queue: (num_vars, num_cons, verdict)
    """
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    os.makedirs(os.path.dirname(SOLVER_LOG_FILE), exist_ok=True)
    log_file = open(LOG_FILE, "a", encoding="utf-8", buffering=1)
    sys.stdout = log_file

    n = len(graph)
    if width <= 1:
        print("Width must be greater than 1!!!!!!!!!!!!", flush=True)
        if queue is not None:
            queue.put(0); queue.put(0); queue.put(False)
        return False

    start = time.time()

    # --- Env & Model ---
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0) 
    env.setParam("LogFile", SOLVER_LOG_FILE)
    if threads is not None:
        try:
            env.setParam("Threads", int(threads))
        except Exception:
            pass
    env.start()

    mdl = gp.Model(name=f"NH-AKL_GUROBI_k{k}_w{width}", env=env)
    if timelimit is not None:
        mdl.setParam("TimeLimit", max(0.001, float(timelimit)))
    # Tùy chọn thêm (mở nếu cần):
    # mdl.setParam("MIPFocus", 1)
    # mdl.setParam("Heuristics", 0.2)
    # mdl.setParam("MemLimit", 64)         # GB
    # mdl.setParam("NodefileStart", 0.5)   # GB

    # --- Index ---
    V = range(1, n + 1)
    LBL = range(1, k + 1)

    # --- Variables x[i,l] ---
    x = {}
    for i in V:
        for l in LBL:
            x[(i, l)] = mdl.addVar(vtype=GRB.BINARY, name=f"x_{i}_{l}")

    # --- Block partition as SAT ---
    block = (k - 1) // width
    last_block_size = k - block * width
    block = max(1, block)
    last_block_size = max(1, last_block_size)

    # --- L_aux / R_aux ---
    L_aux = {}
    for i in V:
        for b in range(1, block + 1):
            wlen = width if b == 1 else max(1, width - 1)
            for idx in range(1, wlen + 1):
                L_aux[(i, b, idx)] = mdl.addVar(vtype=GRB.BINARY, name=f"L_{i}_{b}_{idx}")

    R_aux = {}
    for i in V:
        for b in range(1, block + 1):
            rlen = last_block_size if b == block else width
            for idx in range(1, rlen + 1):
                R_aux[(i, b, idx)] = mdl.addVar(vtype=GRB.BINARY, name=f"R_{i}_{b}_{idx}")

    # --- Safe getters (not exist -> constant 0) ---
    def L_get(ii, bb, idx):
        return L_aux[(ii, bb, idx)] if (ii, bb, idx) in L_aux else 0
    def R_get(ii, bb, idx):
        return R_aux[(ii, bb, idx)] if (ii, bb, idx) in R_aux else 0

    cons_count = 0

    # --- SCL_AMO (9)-(13) ---
    def add_SCL_AMO(order_vars, i, b, wlen, side='R'):
        nonlocal cons_count
        # (9): T[idx] >= order[idx]
        for idx in range(1, wlen + 1):
            if side == 'L':
                mdl.addConstr(L_get(i, b, idx) >= order_vars[idx]); cons_count += 1
            else:
                mdl.addConstr(R_get(i, b, idx) >= order_vars[idx]); cons_count += 1
        # (10): T[idx+1] >= T[idx]
        for idx in range(1, wlen):
            if side == 'L':
                mdl.addConstr(L_get(i, b, idx + 1) >= L_get(i, b, idx)); cons_count += 1
            else:
                mdl.addConstr(R_get(i, b, idx + 1) >= R_get(i, b, idx)); cons_count += 1
        # (11): T[1] <= order[1]
        if side == 'L':
            mdl.addConstr(L_get(i, b, 1) <= order_vars[1]); cons_count += 1
        else:
            mdl.addConstr(R_get(i, b, 1) <= order_vars[1]); cons_count += 1
        # (12): T[j] <= order[j] + T[j-1], j>=2
        for j in range(2, wlen + 1):
            if side == 'L':
                mdl.addConstr(L_get(i, b, j) <= order_vars[j] + L_get(i, b, j - 1)); cons_count += 1
            else:
                mdl.addConstr(R_get(i, b, j) <= order_vars[j] + R_get(i, b, j - 1)); cons_count += 1
        # (13): T[j-1] <= 1 - order[j], j>=2
        for j in range(2, wlen + 1):
            if side == 'L':
                mdl.addConstr(L_get(i, b, j - 1) <= 1 - order_vars[j]); cons_count += 1
            else:
                mdl.addConstr(R_get(i, b, j - 1) <= 1 - order_vars[j]); cons_count += 1

    # --- SCL_AMO for each vertex/block ---
    for i in V:
        # Left side (L)
        for b in range(1, block + 1):
            if b == 1:
                order = {}
                for idx in range(1, width + 1):
                    label = b * width - idx + 1
                    order[idx] = x[(i, label)] if 1 <= label <= k else 0
                add_SCL_AMO(order, i, b, width, side='L')
            else:
                wlen = max(1, width - 1)
                order = {}
                for idx in range(1, wlen + 1):
                    label = b * width - idx + 1
                    order[idx] = x[(i, label)] if 1 <= label <= k else 0
                add_SCL_AMO(order, i, b, wlen, side='L')

        # Right side (R)
        for b in range(1, block + 1):
            if b == block:
                rlen = last_block_size
                order = {}
                for idx in range(1, rlen + 1):
                    label = b * width + idx
                    order[idx] = x[(i, label)] if 1 <= label <= k else 0
                add_SCL_AMO(order, i, b, rlen, side='R')
            else:
                order = {}
                for idx in range(1, width + 1):
                    label = b * width + idx
                    order[idx] = x[(i, label)] if 1 <= label <= k else 0
                add_SCL_AMO(order, i, b, width, side='R')

    # --- ExactlyOne trên “endpoints” ---
    for i in V:
        endpoints = [L_get(i, 1, width)]
        for b in range(1, block):
            endpoints.append(R_get(i, b, width))
        endpoints.append(R_get(i, block, last_block_size))
        mdl.addConstr(gp.quicksum(endpoints) == 1); cons_count += 1

    # --- No-hole trên 1..k ---
    for l in LBL:
        mdl.addConstr(gp.quicksum(x[(i, l)] for i in V) >= 1); cons_count += 1

    # --- Symmetry breaking (chọn đỉnh bậc nhỏ nhất cấm nửa thấp) ---
    deg = {i: 0 for i in V}
    for u in graph:
        deg[u] += len(graph[u])
    node = min(V, key=lambda t: deg[t]) if n > 0 else 1
    for l in range(1, n // 2 + 1):
        if (node, l) in x:
            mdl.addConstr(x[(node, l)] == 0); cons_count += 1

    # --- Anti-width bằng L/R như SAT ---
    added = 0
    Eset = set()
    for u in graph:
        for v in graph[u]:
            if u < v:
                Eset.add((u, v))

    for (u, v) in Eset:
        # 1) [-L[u][1][width], -L[v][1][width]]
        mdl.addConstr(L_get(u, 1, width) + L_get(v, 1, width) <= 1); added += 1
        # 2) phần còn lại
        for label in range(2, k - width + 2):
            block_id = (label - 1) // width + 1
            if label % width == 1:
                if block_id - 1 >= 1:
                    mdl.addConstr(R_get(u, block_id - 1, width) + R_get(v, block_id - 1, width) <= 1); added += 1
            else:
                wmod = (label - 1) % width  # 1..width-1
                lu = L_get(u, block_id, width - wmod)
                ru = R_get(u, block_id, wmod)
                lv = L_get(v, block_id, width - wmod)
                rv = R_get(v, block_id, wmod)

                mdl.addConstr(lu + lv <= 1); added += 1
                mdl.addConstr(lu + rv <= 1); added += 1
                mdl.addConstr(ru + lv <= 1); added += 1
                mdl.addConstr(ru + rv <= 1); added += 1

    num_vars = len(x) + len(L_aux) + len(R_aux)
    num_cons = cons_count + added

    print(f"Gurobi build done — vars≈{num_vars}, cons≈{num_cons} (+anti={added})", flush=True)

    # --- Feasibility objective (0) ---
    mdl.setObjective(0.0, GRB.MINIMIZE)

    # --- Solve ---
    verdict = False
    try:
        mdl.optimize()
        verdict = (mdl.SolCount is not None and mdl.SolCount > 0)
    except Exception as e:
        print(f"[solve error] {e}", flush=True)
        verdict = False

    if queue is not None:
        queue.put(num_vars)
        queue.put(num_cons)
        queue.put(bool(verdict))

    if verdict:
        print(f"Solution found: width={width}", flush=True)
    else:
        print("No solution exists (or no feasible solution found under timelimit).", flush=True)

    end = time.time()
    print(f"Time taken: {round(end - start, 3)} seconds", flush=True)

    # Dọn tài nguyên
    try:
        mdl.dispose()
    except:
        pass
    try:
        env.dispose()
    except:
        pass

    return verdict

# =========================
# Driver (giữ nguyên API cũ)
# =========================
res = [["filename", "n", "k", "proportion", "lower_bound", "upper_bound", "width",
        "num_vars", "num_constraints", "verdict", "time"]]
res2 = []

def read_input(file_path):
    graph = {}
    with open(file_path, 'r') as f:
        n, e = map(int, f.readline().split())
        for i in range(1, n + 1):
            graph[i] = []
        for _ in range(e):
            u, v = map(int, f.readline().split())
            graph[u].append(v)
            graph[v].append(u)
    return graph

def run_test_with_timeout(graph, k, width, time_left_sec=3600, threads=None):
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    log_file = open(LOG_FILE, "a", encoding="utf-8", buffering=1)
    sys.stdout = log_file
    start = time.time()
    queue = multiprocessing.Queue()

    p = multiprocessing.Process(
        target=solve_no_hole_anti_k_labeling_gurobi,
        args=(graph, k, width, queue, time_left_sec, threads)
    )
    p.start()
    p.join(timeout=time_left_sec)
    if p.is_alive():
        p.terminate()
        p.join()

    num_var = queue.get() if not queue.empty() else None
    num_cons = queue.get() if not queue.empty() else None
    verdict = queue.get() if not queue.empty() else None

    res2.extend([num_var, num_cons, verdict])
    elapsed = round((time.time() - start), 2)
    print(f"[Test k={k}, w={width}] Time: {elapsed} seconds", flush=True)
    return bool(verdict)

def binary_search_for_ans(graph, k, left, right, file, timeout_sec=1800, threads=None):
    global res, res2
    res.append([file, None, None, None, None, None, None])
    time_left = timeout_sec
    ans = -9999
    while left <= right:
        width = (left + right) // 2
        t0 = time.time()
        res2.extend([file, k, width])
        ok = run_test_with_timeout(graph, k, width, time_left_sec=time_left, threads=threads)
        res2.append(round(time.time() - t0, 2))
        res.append(res2); res2 = []
        time_left -= time.time() - t0
        if ok:
            ans = width; left = width + 1
        else:
            right = width - 1
        if time_left <= 0.5:
            return -ans if ans != -9999 else -9999
    return ans

def tuantu_for_ans(graph, k, rand, lower_bound, upper_bound, file, timeout_sec=3600, threads=None):
    global res, res2
    res.append([None, None, None, None, None, None, None, None, None, None, None])
    time_left = timeout_sec
    ans = -9999
    width = lower_bound
    while True:
        t0 = time.time()
        res2.extend([file, len(graph), k, rand, lower_bound, upper_bound, width])
        ok = run_test_with_timeout(graph, k, width, time_left_sec=time_left, threads=threads)
        res2.append(round(time.time() - t0, 2))
        res.append(res2); res2 = []
        time_left -= time.time() - t0
        if ok:
            ans = width
            width += 1
            if time_left <= 0.5 or ans == upper_bound:
                return -ans if ans != -9999 else -9999
        else:
            if time_left <= 0.5:
                return -ans if ans != -9999 else -9999
            break
    return ans

def write_to_excel(data, output_file=EXCEL_OUTPUT):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    log_file = open(LOG_FILE, "a", encoding="utf-8", buffering=1)
    sys.stdout = log_file
    try:
        df = pd.DataFrame(data)
        if df.shape[1] >= 11:
            df.columns = ["filename","n","k","proportion","lower_bound","upper_bound","width",
                          "num_vars","num_constraints","verdict","time"]
        df.to_excel(output_file, index=False)
        print(f"Data written to {output_file}", flush=True)
    except Exception as e:
        print(f"Error writing to Excel: {e}", flush=True)

def cnf():  # runner
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    log_file = open(LOG_FILE, "w", encoding="utf-8", buffering=1)
    sys.stdout = log_file

    folder_path = "data/11. hb"
    files = glob.glob(f"{folder_path}/*")
    lst, filename = [], []
    upper_bound = [7,9,17,9,22,13,14,8,24,36,51,39,35,102,79,220,64,256,104,220,326,136,113]
    lower_bound = [6,9,16,8,21,12,12,8,19,32,46,39,28, 91,78,219,46,256,103,219,326,136,112]
    proportion  = [77,57,56,62,72,62,56,64,79,56,69,53,77,52,75,66,69,59,64,78,60,58,70]
    for file in files:
        lst.append(folder_path + "/" + os.path.basename(file))
        filename.append(os.path.basename(file))

    threads = None  # ví dụ: 4

    for i in range(0, len(lst)):
        t0 = time.time()
        graph = read_input(lst[i])
        rand = proportion[i]
        k = len(graph) * rand // 100
        file = filename[i]
        time_limit = 1800
        print(f"Processing file: {file} with k = {k}", flush=True)
        # ans = binary_search_for_ans(graph, k, 2, k-1, file, time_limit, threads=threads)
        ans = tuantu_for_ans(graph, k, rand, lower_bound[i]*rand//100, upper_bound[i], file, time_limit, threads=threads)
        print("$$$$"); print(ans); print("$$$$")
        t1 = time.time()
        print(f"Time taken for {file} with k = {k}: {round(t1 - t0, 2)} seconds", flush=True)
        if ans >= 0:
            print(f"Maximum width for {file} is {ans}", flush=True)
        else:
            if ans == -9999:
                print(f"No answer before timeout for {file}", flush=True)
            else:
                print(f"Maximum width before timeout for {file} is {-ans}", flush=True)
            print("time out", flush=True)

    write_to_excel(res)

if __name__ == "__main__":
    # Quan trọng trên Windows (spawn) khi dùng multiprocessing
    cnf()
