from __future__ import annotations

from itertools import product
from planning.pddl import ActionSchema, Action, State, Objects


def _ground_goal_relevant(domain: list[ActionSchema], objects: Objects,
                          goal: State) -> list[Action]:
    """Ground only schemas whose add_list can produce goal predicates."""
    goal_predicates = {f[0] for f in goal}
    type_map = {
        "r": objects["robots"], "loc": objects["cells"],
        "from_cell": objects["cells"], "to_cell": objects["cells"],
        "obj": objects["objects"], "s": objects["supplies"],
        "p": objects["patients"],
    }
    groundings = []
    for schema in domain:
        if not any(f[0] in goal_predicates for f in schema.add_list):
            continue
        domains = [type_map.get(p, []) for p in schema.parameters]
        if any(len(d) == 0 for d in domains):
            continue
        for values in product(*domains):
            if schema.name == "Move" and len(set(values)) < len(values):
                continue
            binding = dict(zip(schema.parameters, values))
            groundings.append(schema.ground(binding))
    return groundings


def nullHeuristic(
    state: State,
    goal: State,
    domain: list[ActionSchema],
    objects: Objects,
) -> float:
    """Trivial heuristic — always returns 0 (equivalent to uniform-cost search)."""
    return 0


# ---------------------------------------------------------------------------
# Punto 4a – Ignore-Preconditions Heuristic
# ---------------------------------------------------------------------------


def ignorePreconditionsHeuristic(
    state: State,
    goal: State,
    domain: list[ActionSchema],
    objects: Objects,
) -> float:
    """
    Estimate the number of actions needed to satisfy all goal fluents,
    ignoring all action preconditions.

    With no preconditions, any action can be applied at any time.
    Each action can satisfy all goal fluents in its add_list in one step.
    The minimum number of actions to cover all unsatisfied goal fluents is
    a lower bound on the true plan length → this heuristic is admissible.

    Algorithm (greedy set cover):
      1. Compute unsatisfied = goal - state  (fluents still needed).
      2. Ground all actions ignoring preconditions and collect their add_lists.
      3. Greedily pick the action whose add_list covers the most unsatisfied fluents.
      4. Repeat until all fluents are covered; count the actions used.
    """
    insatisfechos = goal - state
    if not insatisfechos:
        return 0

    todas_las_acciones = _ground_goal_relevant(domain, objects, goal)

    fluente_a_acciones = {}
    for accion in todas_las_acciones:
        for fluente in accion.add_list:
            if fluente in insatisfechos:
                fluente_a_acciones.setdefault(fluente, []).append(accion)

    pasos = 0
    while insatisfechos:
        acciones_candidatas = set()
        for fluente in insatisfechos:
            acciones_candidatas.update(fluente_a_acciones.get(fluente, []))

        mejor_accion = None
        mejor_cuenta = -1
        for accion in acciones_candidatas:
            cuenta = len(accion.add_list & insatisfechos)
            if cuenta > mejor_cuenta:
                mejor_cuenta = cuenta
                mejor_accion = accion

        if mejor_cuenta <= 0:
            break

        insatisfechos -= mejor_accion.add_list
        pasos += 1

    return pasos


# ---------------------------------------------------------------------------
# Punto 4b – Ignore-Delete-Lists Heuristic
# ---------------------------------------------------------------------------


def ignoreDeleteListsHeuristic(
    state: State,
    goal: State,
    domain: list[ActionSchema],
    objects: Objects,
) -> float:
    """
    Estimate the plan cost by solving a relaxed problem where no action
    has a delete list (effects never remove fluents from the state).

    In this monotone relaxation, the state only grows over time (fluents are
    never removed), so hill-climbing always makes progress and cannot loop.

    Algorithm (hill-climbing on the relaxed problem):
      1. Start from the current state with a relaxed (monotone) apply function.
      2. At each step, pick the grounded action that adds the most unsatisfied
         goal fluents (greedy hill-climbing).
      3. Count steps until all goal fluents are satisfied (or until no progress).
    """
    todas_las_acciones = _ground_goal_relevant(domain, objects, goal)

    insatisfechos = goal - state
    if not insatisfechos:
        return 0

    fluente_a_acciones = {}
    for accion in todas_las_acciones:
        for fluente in accion.add_list:
            if fluente in goal:
                fluente_a_acciones.setdefault(fluente, []).append(accion)

    estado_relajado = state
    pasos = 0

    while insatisfechos:
        acciones_candidatas = set()
        for fluente in insatisfechos:
            acciones_candidatas.update(fluente_a_acciones.get(fluente, []))

        mejor_accion = None
        mejor_cuenta = -1
        for accion in acciones_candidatas:
            if not accion.precond_pos.issubset(estado_relajado):
                continue
            if not accion.precond_neg.isdisjoint(estado_relajado):
                continue
            cuenta = len(accion.add_list & insatisfechos)
            if cuenta > mejor_cuenta:
                mejor_cuenta = cuenta
                mejor_accion = accion

        if mejor_cuenta <= 0:
            break

        estado_relajado |= mejor_accion.add_list
        insatisfechos -= mejor_accion.add_list
        pasos += 1

    return pasos
