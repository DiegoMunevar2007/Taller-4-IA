from __future__ import annotations

from planning.pddl import Action, Problem, apply_action, is_applicable


# ---------------------------------------------------------------------------
# HTN Infrastructure
# ---------------------------------------------------------------------------


class HLA:
    """
    A High-Level Action (HLA) in HTN planning.

    An HLA is an abstract task that can be refined into sequences of
    more primitive actions (or other HLAs). Each refinement is a list
    of HLA or Action objects.

    name:        Human-readable name for display
    refinements: List of possible refinements, each a list of HLA/Action objects
    """

    def __init__(self, name: str, refinements: list[list] | None = None) -> None:
        self.name = name
        self.refinements = refinements or []

    def __repr__(self) -> str:
        return f"HLA({self.name})"


def is_primitive(action: Action | HLA) -> bool:
    """Return True if action is a primitive (grounded Action), False if it is an HLA."""
    return isinstance(action, Action)


def is_plan_primitive(plan: list[Action | HLA]) -> bool:
    """Return True if every step in the plan is a primitive action."""
    return all(is_primitive(step) for step in plan)


# ---------------------------------------------------------------------------
# Punto 5a – hierarchicalSearch
# ---------------------------------------------------------------------------


def hierarchicalSearch(problem: Problem, hlas: list[HLA]) -> list[Action]:
    """
    HTN planning via BFS over hierarchical plan refinements.

    Start with an initial plan containing a single top-level HLA.
    At each step, find the first non-primitive step in the plan and
    replace it with one of its refinements. Continue until the plan
    is fully primitive and achieves the goal when executed from the
    initial state.

    Returns a list of primitive Action objects, or [] if no plan found.

    Tip: The search space consists of (partial plan, current plan index) pairs.
         Use a Queue (BFS) to explore all refinement choices fairly.
         A plan is a solution when:
           1. It contains only primitive actions (is_plan_primitive), AND
           2. Executing it from the initial state reaches a goal state.
         To simulate execution, apply each action in order using apply_action().
    """
    # El plan inicial contiene el primer HLA de la lista
    from planning.utils import Queue
    plan_inicial = [hlas[0]]

    # Cola para BFS: cada elemento es un plan parcial
    cola = Queue()
    cola.push(plan_inicial)

    while not cola.isEmpty():
        plan = cola.pop()

        # Buscar el primer paso no primitivo (HLA) en el plan
        indice_hla = -1
        for i in range(len(plan)):
            if not is_primitive(plan[i]):
                indice_hla = i
                break

        if indice_hla == -1:
            # El plan es completamente primitivo
            # Verificar si ejecutandolo desde el estado inicial se cumple la meta
            estado = problem.initial_state
            for accion in plan:
                if not is_applicable(estado, accion):
                    estado = None
                    break
                estado = apply_action(estado, accion)

            if estado is not None and problem.isGoalState(estado):
                return plan

        else:
            # Reemplazar el HLA con cada refinamiento posible
            hla = plan[indice_hla]
            for refinamiento in hla.refinements:
                nuevo_plan = plan[:indice_hla] + refinamiento + plan[indice_hla + 1:]
                cola.push(nuevo_plan)

    return []


def build_htn_hierarchy(problem: Problem) -> list[HLA]:
    """
    Build HTN HLAs for the rescue domain.

    The hierarchy defines four HLA types:
      - Navigate(from, to):       Move the robot step by step from one cell to another
      - PrepareSupplies(s, m):    Collect supplies and set them up at the medical post
      - ExtractPatient(p, m):     Pick up the patient and bring them to the medical post
      - FullRescueMission(s,p,m): Complete one rescue: prepare supplies + extract + rescue

    Refinements are built from the ground state to generate concrete Action objects.

    Tip: Refinements for Navigate are all single-step Move sequences between
         adjacent cells. PrepareSupplies and ExtractPatient chain Navigate HLAs
         with primitive PickUp, SetupSupplies, PutDown, and Rescue actions.
    """
    # Obtener los esquemas de accion del dominio
    esquema_mover = problem.domain[0]
    esquema_recoger = problem.domain[1]
    esquema_soltar = problem.domain[2]
    esquema_rescatar = problem.domain[3]
    esquema_preparar = problem.domain[4]

    robot = "robot"

    # Construir el grafo de adyacencia desde el estado inicial
    adyacentes = {}
    for fluente in problem.initial_state:
        if fluente[0] == "Adjacent":
            desde = fluente[1]
            hasta = fluente[2]
            if desde not in adyacentes:
                adyacentes[desde] = []
            adyacentes[desde].append(hasta)

    # Construir Navigate HLAs para cada par de celdas adyacentes
    navigate_hlas = {}
    for desde in adyacentes:
        for hasta in adyacentes[desde]:
            nombre_nav = "Navigate(" + str(desde) + "," + str(hasta) + ")"
            accion_mover = esquema_mover.ground({
                "r": robot,
                "from_cell": desde,
                "to_cell": hasta
            })
            hla_nav = HLA(nombre_nav, [[accion_mover]])
            navigate_hlas[(desde, hasta)] = hla_nav

    # Obtener posiciones del estado inicial
    pos_robot = None
    pos_suministros = {}
    pos_pacientes = {}
    puestos_medicos = []

    for fluente in problem.initial_state:
        if fluente[0] == "At":
            if fluente[1] == robot:
                pos_robot = fluente[2]

    for fluente in problem.initial_state:
        if fluente[0] == "At":
            nombre_entidad = str(fluente[1])
            if nombre_entidad.startswith("supplies"):
                pos_suministros[fluente[1]] = fluente[2]

    for fluente in problem.initial_state:
        if fluente[0] == "At":
            nombre_entidad = str(fluente[1])
            if nombre_entidad.startswith("patient"):
                pos_pacientes[fluente[1]] = fluente[2]

    for fluente in problem.initial_state:
        if fluente[0] == "MedicalPost":
            puestos_medicos.append(fluente[1])

    # Funcion para encontrar el camino entre dos celdas usando BFS
    def encontrar_camino(inicio, fin):
        if inicio == fin:
            return [inicio]
        cola_camino = [[inicio]]
        visitados_camino = set()
        visitados_camino.add(inicio)
        while len(cola_camino) > 0:
            camino = cola_camino.pop(0)
            celda_actual = camino[-1]
            if celda_actual in adyacentes:
                for vecino in adyacentes[celda_actual]:
                    if vecino == fin:
                        return camino + [vecino]
                    if vecino not in visitados_camino:
                        visitados_camino.add(vecino)
                        cola_camino.append(camino + [vecino])
        return []

    def construir_refinamiento_navegacion(inicio, fin):
        """Construye una lista de HLAs Navigate desde inicio hasta fin."""
        camino = encontrar_camino(inicio, fin)
        refinamiento = []
        for i in range(len(camino) - 1):
            desde = camino[i]
            hasta = camino[i + 1]
            clave = (desde, hasta)
            if clave in navigate_hlas:
                refinamiento.append(navigate_hlas[clave])
        return refinamiento

    # Construir PrepareSupplies HLAs
    refinamientos_preparar = []
    for nom_suministro in pos_suministros:
        pos_s = pos_suministros[nom_suministro]
        for puesto in puestos_medicos:
            refinamiento = []
            navegaciones = construir_refinamiento_navegacion(pos_robot, pos_s)
            for nav in navegaciones:
                refinamiento.append(nav)
            accion_recoger = esquema_recoger.ground({
                "r": robot,
                "obj": nom_suministro,
                "loc": pos_s
            })
            refinamiento.append(accion_recoger)
            navegaciones2 = construir_refinamiento_navegacion(pos_s, puesto)
            for nav in navegaciones2:
                refinamiento.append(nav)
            accion_preparar = esquema_preparar.ground({
                "r": robot,
                "s": nom_suministro,
                "loc": puesto
            })
            refinamiento.append(accion_preparar)
            refinamientos_preparar.append(refinamiento)

    # Construir ExtractPatient HLAs
    refinamientos_extraer = []
    for nom_paciente in pos_pacientes:
        pos_p = pos_pacientes[nom_paciente]
        for puesto in puestos_medicos:
            refinamiento = []
            navegaciones = construir_refinamiento_navegacion(pos_robot, pos_p)
            for nav in navegaciones:
                refinamiento.append(nav)
            accion_recoger = esquema_recoger.ground({
                "r": robot,
                "obj": nom_paciente,
                "loc": pos_p
            })
            refinamiento.append(accion_recoger)
            navegaciones2 = construir_refinamiento_navegacion(pos_p, puesto)
            for nav in navegaciones2:
                refinamiento.append(nav)
            accion_soltar = esquema_soltar.ground({
                "r": robot,
                "obj": nom_paciente,
                "loc": puesto
            })
            refinamiento.append(accion_soltar)
            refinamientos_extraer.append(refinamiento)

    # Construir FullRescueMission HLAs
    refinamientos_mision = []
    for nom_suministro in pos_suministros:
        for nom_paciente in pos_pacientes:
            for puesto in puestos_medicos:
                pos_pac = pos_pacientes[nom_paciente]
                hla_preparar = HLA(
                    "PrepareSupplies(" + str(nom_suministro) + "," + str(puesto) + ")",
                    refinamientos_preparar
                )
                hla_extraer = HLA(
                    "ExtractPatient(" + str(nom_paciente) + "," + str(puesto) + ")",
                    refinamientos_extraer
                )
                accion_rescatar = esquema_rescatar.ground({
                    "r": robot,
                    "p": nom_paciente,
                    "loc": puesto
                })
                refinamiento = [hla_preparar, hla_extraer, accion_rescatar]
                refinamientos_mision.append(refinamiento)

    # HLA raiz
    hla_raiz = HLA("FullRescueMission", refinamientos_mision)

    return [hla_raiz]
