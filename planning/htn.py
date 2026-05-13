from __future__ import annotations

from planning.pddl import Action, Problem, apply_action, is_applicable
from planning.utils import Queue


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
    plan_inicial = [hlas[0]]

    # Buscar el indice del primer HLA en el plan inicial
    indice_inicial = 0

    # Cola para BFS: cada elemento es un par (plan, indice_del_primer_hla)
    cola = Queue()
    cola.push((plan_inicial, indice_inicial))

    while not cola.isEmpty():
        plan, indice_hla = cola.pop()
        problem._expanded += 1

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
                # Buscar el siguiente HLA en el nuevo plan empezando desde la misma posicion
                nuevo_indice = -1
                for i in range(indice_hla, len(nuevo_plan)):
                    if not is_primitive(nuevo_plan[i]):
                        nuevo_indice = i
                        break
                cola.push((nuevo_plan, nuevo_indice))

    return []


def encontrar_camino(adyacentes: dict, inicio, fin) -> list:
    """Encuentra el camino mas corto entre dos celdas usando BFS."""
    if inicio == fin:
        return [inicio]
    cola_camino = [[inicio]]
    visitados_camino = {inicio}
    while cola_camino:
        camino = cola_camino.pop(0)
        celda_actual = camino[-1]
        for vecino in adyacentes.get(celda_actual, []):
            if vecino == fin:
                return camino + [vecino]
            if vecino not in visitados_camino:
                visitados_camino.add(vecino)
                cola_camino.append(camino + [vecino])
    return []


def construir_navegacion(adyacentes: dict, navigate_hlas: dict, inicio, fin) -> list:
    """Construye una lista de HLAs Navigate desde inicio hasta fin."""
    camino = encontrar_camino(adyacentes, inicio, fin)
    refinamiento = []
    for i in range(len(camino) - 1):
        clave = (camino[i], camino[i + 1])
        if clave in navigate_hlas:
            refinamiento.append(navigate_hlas[clave])
    return refinamiento


def build_htn_hierarchy(problem: Problem) -> list[HLA]:
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

    # Construir un HLA Navigate por cada par de celdas adyacentes
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

    # Leer posiciones iniciales del estado
    pos_robot = None
    pos_suministros = {}
    pos_pacientes = {}
    puestos_medicos = []

    for fluente in problem.initial_state:
        if fluente[0] == "At" and fluente[1] == robot:
            pos_robot = fluente[2]
        elif fluente[0] == "At" and str(fluente[1]).startswith("supplies"):
            pos_suministros[fluente[1]] = fluente[2]
        elif fluente[0] == "At" and str(fluente[1]).startswith("patient"):
            pos_pacientes[fluente[1]] = fluente[2]
        elif fluente[0] == "MedicalPost":
            puestos_medicos.append(fluente[1])

    # Construir los refinamientos de FullRescueMission
    # Cada refinamiento es: [PrepareSupplies HLA, ExtractPatient HLA, accion Rescue]
    # Contexto de navegacion:
    #   PrepareSupplies: el robot empieza en pos_robot y termina en puesto
    #   ExtractPatient:  el robot empieza en puesto (despues de PrepareSupplies) y termina en puesto
    refinamientos_mision = []

    for nom_suministro, pos_s in pos_suministros.items():
        for nom_paciente, pos_p in pos_pacientes.items():
            for puesto in puestos_medicos:

                # El robot navega de pos_robot a pos_s, recoge los suministros,
                # navega de pos_s al puesto medico y los configura alli
                ref_preparar = []
                ref_preparar += construir_navegacion(adyacentes, navigate_hlas, pos_robot, pos_s)
                ref_preparar.append(esquema_recoger.ground({
                    "r": robot, "obj": nom_suministro, "loc": pos_s
                }))
                ref_preparar += construir_navegacion(adyacentes, navigate_hlas, pos_s, puesto)
                ref_preparar.append(esquema_preparar.ground({
                    "r": robot, "s": nom_suministro, "loc": puesto
                }))

                # El robot navega del puesto al paciente, lo recoge,
                # navega de regreso al puesto y lo deposita alli
                ref_extraer = []
                ref_extraer += construir_navegacion(adyacentes, navigate_hlas, puesto, pos_p)
                ref_extraer.append(esquema_recoger.ground({
                    "r": robot, "obj": nom_paciente, "loc": pos_p
                }))
                ref_extraer += construir_navegacion(adyacentes, navigate_hlas, pos_p, puesto)
                ref_extraer.append(esquema_soltar.ground({
                    "r": robot, "obj": nom_paciente, "loc": puesto
                }))

                # Cada combinacion genera sus propios HLAs con un unico refinamiento
                hla_preparar = HLA(
                    "PrepareSupplies(" + str(nom_suministro) + "," + str(puesto) + ")",
                    [ref_preparar]
                )
                hla_extraer = HLA(
                    "ExtractPatient(" + str(nom_paciente) + "," + str(puesto) + ")",
                    [ref_extraer]
                )
                accion_rescatar = esquema_rescatar.ground({
                    "r": robot, "p": nom_paciente, "loc": puesto
                })

                # Agregar el refinamiento completo a la mision
                refinamientos_mision.append([hla_preparar, hla_extraer, accion_rescatar])

    # HLA raiz que representa la mision completa de rescate
    hla_raiz = HLA("FullRescueMission", refinamientos_mision)
    return [hla_raiz]