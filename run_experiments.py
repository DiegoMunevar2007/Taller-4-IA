#!/usr/bin/env python3
import subprocess
import sys
import re
import os
from collections import OrderedDict

os.chdir(os.path.dirname(os.path.abspath(__file__)))

PYTHON = sys.executable

def run_experiment(problem, planner, layout, heuristic=None, htn=False, timeout=180):
    cmd = [PYTHON, "main.py", "-p", problem, "-l", layout, "-q"]
    if htn:
        cmd += ["-m"]
    else:
        cmd += ["-f", planner]
    if heuristic:
        cmd += ["-h", heuristic]

    time_pattern = re.compile(r"Tiempo de planificación:\s+([\d.]+)s")
    expanded_pattern = re.compile(r"Estados expandidos:\s+(\d+)")
    planlen_pattern = re.compile(r"Longitud del plan:\s+(\d+)")
    fail_pattern = re.compile(r"\[FALLA\]")
    error_pattern = re.compile(r"\[ERROR\]")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        output = result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return None, None, None

    time_m = time_pattern.search(output)
    expanded_m = expanded_pattern.search(output)
    planlen_m = planlen_pattern.search(output)
    is_fail = fail_pattern.search(output)
    is_error = error_pattern.search(output)

    t = float(time_m.group(1)) if time_m else None
    e = int(expanded_m.group(1)) if expanded_m else None
    p = int(planlen_m.group(1)) if planlen_m else None

    if is_fail or is_error or p is None:
        return None, None, None

    return t, e, p


results = OrderedDict()

# ========== SINGLE PATIENT ==========
print("=" * 60)
print("SINGLE PATIENT EXPERIMENTS")
print("=" * 60)

single_configs = [
    # (layout_label, layout_name, problem, planner, heuristic, htn)
    ("tinyBase", "tinyBase", "SimpleRescueProblem", "forwardBFS", None, False),
    ("tinyBase", "tinyBase", "SimpleRescueProblem", "backwardSearch", None, False),
    ("tinyBase", "tinyBase", "SimpleRescueProblem", "aStarPlanner", "ignorePreconditions", False),
    ("tinyBase", "tinyBase", "SimpleRescueProblem", "aStarPlanner", "ignoreDeleteLists", False),
    ("tinyBase", "tinyBase", "SimpleRescueProblem", None, None, True),
    ("smallRescue", "smallRescue", "SimpleRescueProblem", "forwardBFS", None, False),
    ("smallRescue", "smallRescue", "SimpleRescueProblem", "backwardSearch", None, False),
    ("smallRescue", "smallRescue", "SimpleRescueProblem", "aStarPlanner", "ignorePreconditions", False),
    ("smallRescue", "smallRescue", "SimpleRescueProblem", "aStarPlanner", "ignoreDeleteLists", False),
    ("cornerRescue", "cornerRescue", "SimpleRescueProblem", "forwardBFS", None, False),
    ("cornerRescue", "cornerRescue", "SimpleRescueProblem", "backwardSearch", None, False),
    ("cornerRescue", "cornerRescue", "SimpleRescueProblem", "aStarPlanner", "ignorePreconditions", False),
    ("cornerRescue", "cornerRescue", "SimpleRescueProblem", "aStarPlanner", "ignoreDeleteLists", False),
    ("mediumRescue", "mediumRescue", "SimpleRescueProblem", "forwardBFS", None, False),
    ("openRescue", "openRescue", "SimpleRescueProblem", "forwardBFS", None, False),
    ("warehouseRescue", "warehouseRescue", "SimpleRescueProblem", "forwardBFS", None, False),
    ("htnBase", "htnBase", "SimpleRescueProblem", None, None, True),
]

for label, layout, problem, planner, heuristic, htn in single_configs:
    planner_name = "HTN hierarchicalSearch" if htn else planner
    if heuristic:
        planner_name += f" + {heuristic}"
    key = (label, problem, planner_name)
    print(f"  Running: {layout} / {planner_name} ... ", end="", flush=True)
    t, e, p = run_experiment(problem, planner, layout, heuristic, htn)
    results[key] = (t, e, p)
    if t is None:
        print("FAIL/TIMEOUT")
    else:
        print(f"OK  ({t:.3f}s, expanded={e}, len={p})")

# ========== MULTI PATIENT ==========
print()
print("=" * 60)
print("MULTI PATIENT EXPERIMENTS")
print("=" * 60)

multi_configs = [
    ("tinyBase", "tinyBaseMulti", "MultiRescueProblem", "forwardBFS", None, False),
    ("tinyBase", "tinyBaseMulti", "MultiRescueProblem", "backwardSearch", None, False),
    ("tinyBase", "tinyBaseMulti", "MultiRescueProblem", "aStarPlanner", "ignorePreconditions", False),
    ("tinyBase", "tinyBaseMulti", "MultiRescueProblem", "aStarPlanner", "ignoreDeleteLists", False),
    ("smallRescue", "smallRescueMulti", "MultiRescueProblem", "forwardBFS", None, False),
    ("smallRescue", "smallRescueMulti", "MultiRescueProblem", "backwardSearch", None, False),
    ("smallRescue", "smallRescueMulti", "MultiRescueProblem", "aStarPlanner", "ignorePreconditions", False),
    ("smallRescue", "smallRescueMulti", "MultiRescueProblem", "aStarPlanner", "ignoreDeleteLists", False),
    ("cornerRescue", "cornerRescueMulti", "MultiRescueProblem", "forwardBFS", None, False),
    ("cornerRescue", "cornerRescueMulti", "MultiRescueProblem", "backwardSearch", None, False),
    ("cornerRescue", "cornerRescueMulti", "MultiRescueProblem", "aStarPlanner", "ignorePreconditions", False),
    ("cornerRescue", "cornerRescueMulti", "MultiRescueProblem", "aStarPlanner", "ignoreDeleteLists", False),
    ("mediumRescue", "mediumRescueMulti", "MultiRescueProblem", "forwardBFS", None, False),
    ("openRescue", "openRescueMulti", "MultiRescueProblem", "forwardBFS", None, False),
    ("warehouseRescue", "warehouseRescueMulti", "MultiRescueProblem", "forwardBFS", None, False),
]

for label, layout, problem, planner, heuristic, htn in multi_configs:
    planner_name = "HTN hierarchicalSearch" if htn else planner
    if heuristic:
        planner_name += f" + {heuristic}"
    key = (label, problem, planner_name)
    print(f"  Running: {layout} / {planner_name} ... ", end="", flush=True)
    t, e, p = run_experiment(problem, planner, layout, heuristic, htn)
    results[key] = (t, e, p)
    if t is None:
        print("FAIL/TIMEOUT")
    else:
        print(f"OK  ({t:.3f}s, expanded={e}, len={p})")


# ========== GENERATE LATEX TABLE ==========
print()
print("=" * 60)
print("GENERATING LATEX TABLE")
print("=" * 60)

def fmt_time(t):
    if t is None:
        return "---"
    return f"{t:.3f}"

def fmt_expanded(e, planner_name):
    if e is None or e == 0:
        return "---"
    return f"{e:,}"

def fmt_planlen(p):
    if p is None:
        return "---"
    return str(p)

def fmt_planner_name(raw):
    if raw == "backwardSearch":
        return "backwardSearch"
    elif raw == "forwardBFS":
        return "forwardBFS"
    elif raw == "aStarPlanner + ignorePreconditions":
        return "A* + ignorePreconditions"
    elif raw == "aStarPlanner + ignoreDeleteLists":
        return "A* + ignoreDeleteLists"
    elif raw == "HTN hierarchicalSearch":
        return "HTN hierarchicalSearch"
    return raw

sections = [
    ("tinyBase", "SimpleRescueProblem", "tinyBase (5$\\times$7, 1 paciente)", [
        "forwardBFS", "backwardSearch",
        "aStarPlanner + ignorePreconditions", "aStarPlanner + ignoreDeleteLists",
        "HTN hierarchicalSearch",
    ]),
    ("smallRescue", "SimpleRescueProblem", "smallRescue (10$\\times$10, 1 paciente)", [
        "forwardBFS", "backwardSearch",
        "aStarPlanner + ignorePreconditions", "aStarPlanner + ignoreDeleteLists",
    ]),
    ("cornerRescue", "SimpleRescueProblem", "cornerRescue (9$\\times$8, 1 paciente)", [
        "forwardBFS", "backwardSearch",
        "aStarPlanner + ignorePreconditions", "aStarPlanner + ignoreDeleteLists",
    ]),
    ("mediumRescue", "SimpleRescueProblem", "mediumRescue (12$\\times$12, 1 paciente)", [
        "forwardBFS",
    ]),
    ("openRescue", "SimpleRescueProblem", "openRescue (10$\\times$12, 1 paciente)", [
        "forwardBFS",
    ]),
    ("warehouseRescue", "SimpleRescueProblem", "warehouseRescue (12$\\times$14, 1 paciente)", [
        "forwardBFS",
    ]),
    ("htnBase", "SimpleRescueProblem", "htnBase (9$\\times$8, 1 paciente)", [
        "HTN hierarchicalSearch",
    ]),
]

multi_sections = [
    ("tinyBase", "MultiRescueProblem", "tinyBase (5$\\times$7, 2 pacientes)", [
        "forwardBFS", "backwardSearch",
        "aStarPlanner + ignorePreconditions", "aStarPlanner + ignoreDeleteLists",
    ]),
    ("smallRescue", "MultiRescueProblem", "smallRescue (10$\\times$10, 2 pacientes)", [
        "forwardBFS", "backwardSearch",
        "aStarPlanner + ignorePreconditions", "aStarPlanner + ignoreDeleteLists",
    ]),
    ("cornerRescue", "MultiRescueProblem", "cornerRescue (9$\\times$8, 2 pacientes)", [
        "forwardBFS", "backwardSearch",
        "aStarPlanner + ignorePreconditions", "aStarPlanner + ignoreDeleteLists",
    ]),
    ("mediumRescue", "MultiRescueProblem", "mediumRescue (12$\\times$12, 2 pacientes)", [
        "forwardBFS",
    ]),
    ("openRescue", "MultiRescueProblem", "openRescue (10$\\times$12, 2 pacientes)", [
        "forwardBFS",
    ]),
    ("warehouseRescue", "MultiRescueProblem", "warehouseRescue (12$\\times$14, 2 pacientes)", [
        "forwardBFS",
    ]),
]

lines = []
lines.append("\\begin{table}[H]")
lines.append("\\begin{center}")
lines.append("\\label{tab:comparativa}")
lines.append("\\renewcommand{\\arraystretch}{1.4}")
lines.append("\\begin{tabular}{|l|c|c|c|}")
lines.append("\\hline")
lines.append("\\textbf{Algoritmo} & \\textbf{Tiempo (s)} & \\textbf{Estados expandidos} & \\textbf{Longitud del plan} \\\\")
lines.append("\\hline")

all_sections = sections + multi_sections

for label, problem, header, planners in all_sections:
    lines.append(f"\\multicolumn{{4}}{{|c|}}{{\\textbf{{{header}}}}} \\\\")
    lines.append("\\hline")
    for pname in planners:
        key = (label, problem, pname)
        t, e, p = results.get(key, (None, None, None))
        time_str = fmt_time(t)
        exp_str = fmt_expanded(e, pname)
        plan_str = fmt_planlen(p)
        display_name = fmt_planner_name(pname)
        lines.append(f"{display_name} & {time_str} & {exp_str} & {plan_str} \\\\")
    lines.append("\\hline")

lines.append("\\end{tabular}")
lines.append("\\end{center}")
lines.append("\\end{table}")

table = "\n".join(lines)
print(table)

with open("tabla_resultados.tex", "w") as f:
    f.write(table)
print("\nTabla guardada en 'tabla_resultados.tex'")
