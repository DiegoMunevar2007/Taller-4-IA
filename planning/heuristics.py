from __future__ import annotations

from planning.pddl import ActionSchema, State, Objects, get_all_groundings, get_applicable_actions


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
      1. Compute unsatisfied = goal − state  (fluents still needed).
      2. Ground all actions ignoring preconditions and collect their add_lists.
      3. Greedily pick the action whose add_list covers the most unsatisfied fluents.
      4. Repeat until all fluents are covered; count the actions used.

    Tip: frozenset supports set difference (-) and intersection (&).
         You only need to ground actions once per call (use get_applicable_actions
         with the initial state, or generate all groundings regardless of state).
         Remember: with no preconditions, every grounding is "applicable".
    """
    # Calcular los fluentes de la meta que aun no estan en el estado
    insatisfechos = goal - state
    if len(insatisfechos) == 0:
        return 0

    # Obtener todas las acciones grounded (sin precondiciones, todas son aplicables)
    todas_las_acciones = get_all_groundings(domain, objects)

    pasos = 0

    while len(insatisfechos) > 0:
        mejor_accion = None
        mejor_cuenta = -1

        # Buscar la accion que cubra mas fluentes insatisfechos
        for accion in todas_las_acciones:
            cuenta = 0
            for fluente in accion.add_list:
                if fluente in insatisfechos:
                    cuenta = cuenta + 1
            if cuenta > mejor_cuenta:
                mejor_cuenta = cuenta
                mejor_accion = accion

        if mejor_cuenta <= 0:
            break

        # Remover los fluentes cubiertos por la mejor accion
        insatisfechos = insatisfechos - mejor_accion.add_list

        pasos = pasos + 1

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

    Tip: In the relaxed problem, apply_action never removes fluents.
         You can implement this by treating del_list as empty for all actions.
         Use get_applicable_actions to enumerate applicable grounded actions at
         each step (preconditions still apply in the relaxed model).
    """
    # Empezar desde el estado actual
    estado_relajado = state
    pasos = 0

    while not goal.issubset(estado_relajado):

        # Obtener acciones aplicables en el estado actual
        acciones_aplicables = get_applicable_actions(estado_relajado, domain, objects)

        if len(acciones_aplicables) == 0:
            break

        mejor_accion = None
        mejor_cuenta = -1

        # Buscar la accion que agregue mas fluentes de la meta aun no satisfechos
        for accion in acciones_aplicables:
            cuenta = 0
            for fluente in accion.add_list:
                if fluente in goal and fluente not in estado_relajado:
                    cuenta = cuenta + 1
            if cuenta > mejor_cuenta:
                mejor_cuenta = cuenta
                mejor_accion = accion

        if mejor_cuenta <= 0:
            break

        # Aplicar la accion sin borrar (solo agregar, ignorar del_list)
        estado_relajado = estado_relajado | mejor_accion.add_list
        pasos = pasos + 1

    return pasos
