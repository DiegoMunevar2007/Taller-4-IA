from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Callable

from planning.pddl import (
    Action,
    ActionSchema,
    Problem,
    State,
    Objects,
    get_all_groundings,
)
from planning.utils import Queue, PriorityQueue
from planning.heuristics import nullHeuristic


# ---------------------------------------------------------------------------
# Reference implementation – read and understand before coding the rest.
# ---------------------------------------------------------------------------


def tinyBaseSearch(problem: Problem) -> list[Action]:
    """
    Hardcoded plan for the tinyBase layout.
    The robot at (1,4) must: pick up supplies at (1,3), set them up at (1,2),
    pick up the patient at (1,1), bring them to (1,2), and execute Rescue.

    Useful to understand the Action object format and plan structure.
    """
    robot = "robot"
    supplies = "supplies_0"
    patient = "patient_0"

    c14 = (1, 4)  # robot start
    c13 = (1, 3)  # supplies
    c12 = (1, 2)  # medical post
    c11 = (1, 1)  # patient

    plan = [
        Action(
            "Move(robot,(1,4),(1,3))",
            [("At", robot, c14), ("Adjacent", c14, c13), ("Free", c13)],
            [],
            [("At", robot, c13), ("Free", c14)],
            [("At", robot, c14), ("Free", c13)],
        ),
        Action(
            "PickUp(robot,supplies_0,(1,3))",
            [
                ("At", robot, c13),
                ("At", supplies, c13),
                ("HandsFree", robot),
                ("Pickable", supplies),
            ],
            [],
            [("Holding", robot, supplies)],
            [("At", supplies, c13), ("HandsFree", robot)],
        ),
        Action(
            "Move(robot,(1,3),(1,2))",
            [("At", robot, c13), ("Adjacent", c13, c12), ("Free", c12)],
            [],
            [("At", robot, c12), ("Free", c13)],
            [("At", robot, c13), ("Free", c12)],
        ),
        Action(
            "SetupSupplies(robot,supplies_0,(1,2))",
            [("At", robot, c12), ("MedicalPost", c12), ("Holding", robot, supplies)],
            [("SuppliesReady", c12)],
            [("SuppliesReady", c12), ("HandsFree", robot)],
            [("Holding", robot, supplies)],
        ),
        Action(
            "Move(robot,(1,2),(1,1))",
            [("At", robot, c12), ("Adjacent", c12, c11), ("Free", c11)],
            [],
            [("At", robot, c11), ("Free", c12)],
            [("At", robot, c12), ("Free", c11)],
        ),
        Action(
            "PickUp(robot,patient_0,(1,1))",
            [
                ("At", robot, c11),
                ("At", patient, c11),
                ("HandsFree", robot),
                ("Pickable", patient),
            ],
            [],
            [("Holding", robot, patient)],
            [("At", patient, c11), ("HandsFree", robot)],
        ),
        Action(
            "Move(robot,(1,1),(1,2))",
            [("At", robot, c11), ("Adjacent", c11, c12), ("Free", c12)],
            [],
            [("At", robot, c12), ("Free", c11)],
            [("At", robot, c11), ("Free", c12)],
        ),
        Action(
            "PutDown(robot,patient_0,(1,2))",
            [("At", robot, c12), ("Holding", robot, patient)],
            [],
            [("At", patient, c12), ("HandsFree", robot)],
            [("Holding", robot, patient)],
        ),
        Action(
            "Rescue(robot,patient_0,(1,2))",
            [
                ("At", robot, c12),
                ("At", patient, c12),
                ("MedicalPost", c12),
                ("SuppliesReady", c12),
            ],
            [],
            [("Rescued", patient)],
            [("At", patient, c12)],
        ),
    ]
    return plan


# ---------------------------------------------------------------------------
# Punto 2 – Forward Planning
# ---------------------------------------------------------------------------


def forwardBFS(problem: Problem) -> list[Action]:
    """
    Forward BFS in state space.

    Explore states reachable from the initial state by applying actions,
    in breadth-first order, until a goal state is found.

    Returns a list of Action objects forming a valid plan, or [] if no plan exists.

    Tip: The state is a frozenset of fluents. Use problem.getSuccessors(state)
         to get (next_state, action, cost) triples. Track visited states to
         avoid revisiting the same state twice (graph search, not tree search).
         
    Implementación original:
    cola = Queue()
    estado_inicial = problem.getStartState()
    cola.push(estado_inicial)
    
    # Mantener registro de estados visitados
    visitados = set()
    visitados.add(estado_inicial)
    
    # Mantener registro padre-acción para reconstruir el plan
    padre = {}
    accion_desde_padre = {}
    padre[estado_inicial] = None
    
    # BFS
    while not cola.isEmpty():
        estado_actual = cola.pop()
        
        # Verificar si es estado objetivo
        if problem.isGoalState(estado_actual):
            # Reconstruir el plan
            plan = []
            estado_temp = estado_actual
            while padre[estado_temp] is not None:
                accion = accion_desde_padre[estado_temp]
                plan.append(accion)
                estado_temp = padre[estado_temp]
            plan.reverse()
            return plan
        
        # Explorar sucesores
        sucesores = problem.getSuccessors(estado_actual)
        for siguiente_estado, accion, costo in sucesores:
            if siguiente_estado not in visitados:
                visitados.add(siguiente_estado)
                cola.push(siguiente_estado)
                padre[siguiente_estado] = estado_actual
                accion_desde_padre[siguiente_estado] = accion
    
    # No se encontró solución
    return []
    
    Se hizo uso de IA para mejorar el rendimiento, utilizando un diccionario para almacenar el padre y la acción que llevó a cada estado, 
    al igual que usar deque para la cola en lugar de la implementación original con Queue.
    """
    # Usar una cola FIFO para BFS
    cola = deque()
    estado_inicial = problem.getStartState()
    cola.append(estado_inicial)

    # Diccionario para rastrear el camino: estado -> (estado_anterior, accion)
    padres = {}
    visitados = set()
    visitados.add(estado_inicial)

    while cola:
        estado_actual = cola.popleft()

        # Revisar si ya llegamos a la meta
        if problem.isGoalState(estado_actual):

            # Reconstruir el plan desde la meta hasta el inicio
            plan = []
            estado = estado_actual
            while estado in padres:
                estado_anterior, accion = padres[estado]
                plan.append(accion)
                estado = estado_anterior
            plan.reverse()
            return plan

        # Obtener los sucesores del estado actual
        sucesores = problem.getSuccessors(estado_actual)
        for par in sucesores:
            siguiente_estado, accion, costo = par

            if siguiente_estado not in visitados:
                visitados.add(siguiente_estado)
                padres[siguiente_estado] = (estado_actual, accion)
                cola.append(siguiente_estado)

    # No se encontro plan
    return []


def regress(conjunto_meta: State, accion: Action) -> State | None:
    """
    Compute the regression of goal_set through action.

    Given a goal description (set of fluents that must be true) and an action,
    return the new goal description that, if satisfied, guarantees the original
    goal is satisfied after executing action.

    REGRESS(g, a) = (g − ADD(a)) ∪ PRECOND_pos(a)
        IF:  ADD(a) ∩ g ≠ ∅   (action is relevant: contributes to the goal)
        AND: DEL(a) ∩ g = ∅   (action does not undo any goal fluent)
    Returns None if the action is not relevant or creates a contradiction.

    Tip: Use frozenset operations: intersection (&), difference (-), union (|).
         Check relevance first, then check for contradictions, then compute.
    """
    # Verificar relevancia con interseccion de conjuntos
    if not (accion.add_list & conjunto_meta):
        return None

    # Verificar que la accion no borre ningun fluente de la meta
    if accion.del_list & conjunto_meta:
        return None

    # REGRESS(g, a) = (g - ADD(a)) ∪ PRECOND_pos(a)
    nuevo_conjunto = (conjunto_meta - accion.add_list) | accion.precond_pos
    return nuevo_conjunto

def backwardSearch(problem: Problem) -> list[Action]:
    """
    Backward search (regression search) from the goal.

    Start from the goal description and apply action regressions until
    the resulting goal is satisfied by the initial state.

    Returns a list of Action objects forming a valid plan (in forward order),
    or [] if no plan exists.

    Tip: The "state" in backward search is a frozenset of fluents that must
         be true (a partial goal description). The initial state is reached
         when all fluents in the current goal are satisfied by problem.initial_state.
         Only consider actions whose add_list has at least one unsatisfied goal fluent
         (relevant actions). Use regress() to compute the new subgoal.
         Skip subgoals that contain static predicates (MedicalPost, Adjacent,
         Pickable) that are false in the initial state — these are dead ends.
         
    Implementación original:
      # Obtener todas las acciones grounded posibles
    todas_las_acciones = get_all_groundings(problem.domain, problem.objects)

    # Predicados estaticos que no cambian durante la ejecucion
    predicados_estaticos = ["MedicalPost", "Adjacent", "Pickable"]

    # Indice: para cada fluente especifico, que acciones lo agregan
    # Esto permite obtener solo las acciones relevantes para cada subgoal
    fluente_a_acciones = {}
    for accion in todas_las_acciones:
        for fluente in accion.add_list:
            if fluente not in fluente_a_acciones:
                fluente_a_acciones[fluente] = []
            fluente_a_acciones[fluente].append(accion)

    # Usar cola (BFS) para encontrar el plan mas corto
    cola = Queue()
    cola.push((problem.goal, []))
    visitados = set()
    visitados.add(problem.goal)

    while not cola.isEmpty():
        subgoal, plan = cola.pop()

        # Verificar si el subgoal se cumple desde el estado inicial
        se_cumple = True
        for fluente in subgoal:
            if fluente not in problem.initial_state:
                se_cumple = False
                break

        if se_cumple:
            return plan

        # Usar el indice por fluente especifico para obtener acciones relevantes
        acciones_vistas = set()
        for fluente in subgoal:
            if fluente in fluente_a_acciones:
                for accion in fluente_a_acciones[fluente]:
                    if accion in acciones_vistas:
                        continue
                    acciones_vistas.add(accion)

                    nuevo_subgoal = regress(subgoal, accion)
                    if nuevo_subgoal is None:
                        continue

                    # Verificar predicados estaticos falsos
                    es_valido = True
                    for fl in nuevo_subgoal:
                        if fl[0] in predicados_estaticos:
                            if fl not in problem.initial_state:
                                es_valido = False
                                break

                    if not es_valido:
                        continue

                    if nuevo_subgoal not in visitados:
                        visitados.add(nuevo_subgoal)
                        cola.push((nuevo_subgoal, [accion] + plan))

    return []
    
    Se hizo uso de IA para mejorar el rendimiento, sin embargo,
    no se obtuvo una mejora significativa; de todas formas, se implementó un diccionario para almacenar las acciones relevantes para cada fluente específico
    y se utilizó deque para la cola en lugar de la implementación original con Queue.
    """
    todas_las_acciones = get_all_groundings(problem.domain, problem.objects)

    # Predicados estáticos
    predicados_dinamicos = set()
    for accion in todas_las_acciones:
        for fluente in accion.add_list | accion.del_list:
            predicados_dinamicos.add(fluente[0])

    predicados_estaticos = set()
    for fluente in problem.initial_state:
        if fluente[0] not in predicados_dinamicos:
            predicados_estaticos.add(fluente[0])

    acciones_validas = [
        a for a in todas_las_acciones
        if all(
            f[0] not in predicados_estaticos or f in problem.initial_state
            for f in a.precond_pos
        )
    ]

    fluente_a_acciones = defaultdict(list)
    for accion in acciones_validas:
        for fluente in accion.add_list:
            fluente_a_acciones[fluente].append(accion)

    def es_consistente(subgoal: State) -> bool:
        posiciones = {}
        for f in subgoal:
            if f[0] == "At":
                if f[1] in posiciones and posiciones[f[1]] != f[2]:
                    return False
                posiciones[f[1]] = f[2]
            elif f[0] == "Holding":
                if "HandsFree" in posiciones:
                    return False
                posiciones["Holding"] = f[1]
            elif f[0] == "HandsFree":
                if "Holding" in posiciones:
                    return False
                posiciones["HandsFree"] = f[1]
        return True

    cola = deque()
    goal_inicial = frozenset(problem.goal)
    cola.append(goal_inicial)
    visitados = {goal_inicial}
    padres = {}

    while cola:
        subgoal = cola.popleft()
        problem._expanded += 1

        if subgoal.issubset(problem.initial_state):
            plan = []
            actual = subgoal
            while actual in padres:
                padre, accion = padres[actual]
                plan.append(accion)
                actual = padre
            return plan

        acciones_relevantes = set()
        for fluente in subgoal:
            acciones_relevantes.update(fluente_a_acciones.get(fluente, []))

        for accion in acciones_relevantes:
            nuevo_subgoal = regress(subgoal, accion)
            if nuevo_subgoal is None:
                continue

            if nuevo_subgoal in visitados:
                continue

            if not es_consistente(nuevo_subgoal):
                continue

            visitados.add(nuevo_subgoal)
            padres[nuevo_subgoal] = (subgoal, accion)
            cola.append(nuevo_subgoal)

    return []



# ---------------------------------------------------------------------------
# Punto 4 – A* Planner
# ---------------------------------------------------------------------------

# Heuristic signature:  heuristic(state, goal, domain, objects) -> float
Heuristic = Callable[[State, State, list[ActionSchema], Objects], float]


def aStarPlanner(
    problem: Problem,
    heuristic: Heuristic = nullHeuristic,
) -> list[Action]:
    """
    Forward A* search guided by a heuristic.

    Combines the real accumulated cost g(n) with the heuristic estimate h(n)
    to prioritize which state to expand next: f(n) = g(n) + h(n).

    Returns a list of Action objects forming a valid plan, or [] if no plan exists.

    Tip: The heuristic signature is heuristic(state, goal, domain, objects) → float.
         Use PriorityQueue with priority = g + h(next_state).
         Track the best g-cost seen for each state to avoid stale expansions.
    """
    # Cola de prioridad ordenada por f(n) = g(n) + h(n)
    cola = PriorityQueue()
    estado_inicial = problem.getStartState()
    h_inicial = heuristic(estado_inicial, problem.goal, problem.domain, problem.objects)
    cola.push(estado_inicial, 0.0 + h_inicial)

    # Diccionario para el mejor costo g conocido para cada estado
    mejor_g = {}
    mejor_g[estado_inicial] = 0

    # Diccionario para reconstruir el plan
    padres = {}

    while not cola.isEmpty():
        estado_actual = cola.pop()

        if problem.isGoalState(estado_actual):
            # Reconstruir el plan
            plan = []
            estado = estado_actual
            while estado in padres:
                estado_anterior, accion = padres[estado]
                plan.append(accion)
                estado = estado_anterior
            plan.reverse()
            return plan

        # Obtener sucesores
        sucesores = problem.getSuccessors(estado_actual)

        for par in sucesores:
            siguiente_estado, accion, costo = par
            g_nuevo = mejor_g[estado_actual] + costo

            if siguiente_estado not in mejor_g or g_nuevo < mejor_g[siguiente_estado]:
                mejor_g[siguiente_estado] = g_nuevo
                padres[siguiente_estado] = (estado_actual, accion)
                h = heuristic(siguiente_estado, problem.goal, problem.domain, problem.objects)
                prioridad = float(g_nuevo) + h
                cola.push(siguiente_estado, prioridad)

    return []


# Aliases used by the command-line argument parser
tinyBaseSearch = tinyBaseSearch
forwardBFS = forwardBFS
backwardSearch = backwardSearch
aStarPlanner = aStarPlanner
