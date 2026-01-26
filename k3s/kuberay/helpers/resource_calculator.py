"""Calculadora de recursos para anticipar cuántos recursos necesita Tune+Train.

Esta herramienta nació para estimar CPU (Ray Tune + Ray Train). Para uso local
(laptop/desktop) es común reservar CPUs/RAM para el SO y overhead de Ray.
"""

from __future__ import annotations

import argparse
import os
import sys
from math import ceil
from typing import Optional

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from resources import ResourceConfig


def _clamp_non_negative(value: float, name: str) -> float:
    if value < 0:
        raise ValueError(f"{name} must be >= 0")
    return value


def usable_cluster_cpus(total_cluster_cpus: int, reserve_cpus: int) -> int:
    if total_cluster_cpus <= 0:
        raise ValueError("total cluster cpus must be positive")
    if reserve_cpus < 0:
        raise ValueError("reserve_cpus must be >= 0")
    return max(0, total_cluster_cpus - reserve_cpus)


def per_trial_memory_gb(cfg: ResourceConfig, mem_gb_head: float, mem_gb_per_worker: float) -> float:
    _clamp_non_negative(mem_gb_head, "mem_gb_head")
    _clamp_non_negative(mem_gb_per_worker, "mem_gb_per_worker")
    return mem_gb_head + cfg.num_workers * mem_gb_per_worker


def max_concurrent_trials(total_cluster_cpus: int, cfg: ResourceConfig) -> int:
    per_trial = cfg.total_cpus_per_trial()
    if per_trial <= 0:
        raise ValueError("per trial CPU must be positive")
    return total_cluster_cpus // per_trial


def max_concurrent_trials_with_memory(
    usable_cpus: int,
    cfg: ResourceConfig,
    usable_mem_gb: Optional[float],
    mem_gb_head: float,
    mem_gb_per_worker: float,
) -> int:
    cpu_based = max_concurrent_trials(usable_cpus, cfg) if usable_cpus > 0 else 0
    if usable_mem_gb is None:
        return cpu_based

    per_trial_mem = per_trial_memory_gb(cfg, mem_gb_head, mem_gb_per_worker)
    if per_trial_mem <= 0:
        return cpu_based

    mem_based = int(usable_mem_gb // per_trial_mem)
    return min(cpu_based, mem_based)


def required_cluster_cpus(target_trials: int, cfg: ResourceConfig) -> int:
    return cfg.total_cpus_per_trial() * target_trials


def print_summary(
    total_cluster_cpus: int,
    reserve_cpus: int,
    cluster_mem_gb: Optional[float],
    reserve_mem_gb: float,
    mem_gb_head: float,
    mem_gb_per_worker: float,
    cfg: ResourceConfig,
) -> None:
    usable_cpus = usable_cluster_cpus(total_cluster_cpus, reserve_cpus)
    usable_mem_gb = None
    if cluster_mem_gb is not None:
        _clamp_non_negative(cluster_mem_gb, "cluster_mem_gb")
        usable_mem_gb = max(0.0, cluster_mem_gb - reserve_mem_gb)

    print("\nResumen de recursos calculados:")
    print(f"  Workers por trial: {cfg.num_workers}")
    print(f"  CPUs por worker: {cfg.cpus_per_worker}")
    print(f"  CPUs destinados al head: {cfg.cpus_head}")
    print(f"  Total por trial (head + workers): {cfg.total_cpus_per_trial()}")
    print(f"  CPUs totales: {total_cluster_cpus} (reservadas: {reserve_cpus} → utilizables: {usable_cpus})")

    if cluster_mem_gb is not None:
        per_trial_mem = per_trial_memory_gb(cfg, mem_gb_head, mem_gb_per_worker)
        print(f"  RAM total: {cluster_mem_gb:.2f} GB (reservada: {reserve_mem_gb:.2f} → utilizable: {usable_mem_gb:.2f} GB)")
        if per_trial_mem > 0:
            print(f"  RAM por trial (head + workers): {per_trial_mem:.2f} GB (estimada)")

    max_trials = max_concurrent_trials_with_memory(
        usable_cpus=usable_cpus,
        cfg=cfg,
        usable_mem_gb=usable_mem_gb,
        mem_gb_head=mem_gb_head,
        mem_gb_per_worker=mem_gb_per_worker,
    )
    if cluster_mem_gb is None:
        print(f"  Máximo de trials concurrentes con {usable_cpus} CPUs utilizables: {max_trials} (teórico)")
    else:
        print(f"  Máximo de trials concurrentes (CPU+RAM): {max_trials} (teórico)")


def print_suggestions(
    total_cluster_cpus: int,
    reserve_cpus: int,
    desired_trials: Optional[int],
    cfg: ResourceConfig,
) -> None:
    if not desired_trials:
        return

    usable_cpus = usable_cluster_cpus(total_cluster_cpus, reserve_cpus)
    if usable_cpus <= 0:
        print("\n[CRÍTICO] No hay CPUs utilizables (todo quedó reservado).")
        return

    current_max = max_concurrent_trials(usable_cpus, cfg)
    if current_max >= desired_trials:
        return

    print("\nSugerencias para lograr el paralelismo deseado (CPU):")
    per_trial_budget = usable_cpus // desired_trials
    if per_trial_budget <= 0:
        print(f"  [CRÍTICO] Con {usable_cpus} CPUs utilizables no puedes correr {desired_trials} trials.")
        return

    if per_trial_budget <= cfg.cpus_head:
        print(
            f"  [CRÍTICO] Con {desired_trials} trials, tu presupuesto por trial es {per_trial_budget} CPU(s),"
            f" pero el head ya consume {cfg.cpus_head}. Reduce trials o reserva menos CPUs."
        )
        return

    remaining_for_workers = per_trial_budget - cfg.cpus_head
    suggested_cpus_per_worker = remaining_for_workers // max(1, cfg.num_workers)
    if suggested_cpus_per_worker <= 0:
        print(
            f"  Baja workers por trial o baja el paralelismo: con {cfg.num_workers} worker(s) por trial,"
            f" no alcanza CPU para {desired_trials} trials."
        )
        return

    print(
        f"  Para {desired_trials} trials en paralelo con {usable_cpus} CPUs utilizables,"
        f" apunta a <= {per_trial_budget} CPUs por trial."
    )
    print(
        f"  Manteniendo num_workers={cfg.num_workers}, prueba cpus_per_worker≈{suggested_cpus_per_worker}"
        f" (hoy tienes {cfg.cpus_per_worker})."
    )


def print_node_assessment(
    cluster_nodes: Optional[int], cpus_per_node: Optional[int], cfg: ResourceConfig, desired_trials: Optional[int]
) -> None:
    if not cluster_nodes or not cpus_per_node:
        return

    print("\nVerificación de arquitectura de nodos (Bin Packing):")
    total_node_cpus = cluster_nodes * cpus_per_node
    print(f"  Configuración: {cluster_nodes} nodos × {cpus_per_node} CPUs = {total_node_cpus} CPUs totales")

    # ¿Cabe un worker en un nodo?
    if cpus_per_node < cfg.cpus_per_worker:
        print(f"  [ERROR] INFACTIBLE: El worker pide {cfg.cpus_per_worker} CPUs y el nodo solo tiene {cpus_per_node}.")
        return

    # ¿Cuántos workers caben por nodo?
    workers_per_node = cpus_per_node // cfg.cpus_per_worker
    waste_per_node = cpus_per_node % cfg.cpus_per_worker
    print(f"  Capacidad real: {workers_per_node} worker(s) por nodo.")
    if waste_per_node > 0:
        print(f"  Aviso: Cada nodo desperdicia {waste_per_node} CPU(s) debido al tamaño del worker.")

    # ¿Cuántos nodos consume un trial?
    nodes_for_workers = ceil(cfg.num_workers / workers_per_node)
    print(f"  Un solo trial consumirá aproximadamente {nodes_for_workers} nodo(s) para sus workers.")

    if desired_trials:
        # Recomendar replicas para el YAML
        print("\nRecomendación para autoscaling (RayJob YAML):")
        # minReplicas: al menos lo necesario para 1 trial
        min_reps = nodes_for_workers
        # maxReplicas: lo necesario para los trials deseados
        max_reps = nodes_for_workers * desired_trials
        
        print(f"  minReplicas recomendadas: {min_reps} (para asegurar que 1 trial pueda arrancar siempre)")
        print(f"  maxReplicas recomendadas: {max_reps} (para permitir {desired_trials} trials paralelos)")

        # CPUs teóricos vs reales
        needed_cpus = required_cluster_cpus(desired_trials, cfg)
        if total_node_cpus >= needed_cpus:
            print(f"\n  [OK] CPUs suficientes: Tienes {total_node_cpus}, necesitas {needed_cpus}.")
        else:
            print(f"\n  [CRÍTICO] CPUs insuficientes: Faltan {needed_cpus - total_node_cpus} CPUs.")

        # Nodos necesarios considerando fragmentación
        total_nodes_needed = ceil(needed_cpus / cpus_per_node)
        if cluster_nodes < total_nodes_needed:
            print(f"  [!] Alerta: El clúster físico es pequeño. Necesitas que el Cluster Autoscaler de K8s pueda subir hasta {total_nodes_needed} nodos.")


if __name__ == "__main__":
    DEFAULT_CLUSTER_CPUS = 32
    parser = argparse.ArgumentParser(
        description="Calcula cuánto CPU necesitas y recomienda parámetros de autoscaling."
    )
    parser.add_argument("--cluster-cpus", type=int, default=DEFAULT_CLUSTER_CPUS, help="CPUs totales del clúster")
    parser.add_argument(
        "--reserve-cpus",
        type=int,
        default=2,
        help="CPUs a reservar para SO/overhead (útil en laptop o nodos pequeños)",
    )
    parser.add_argument("--cluster-nodes", type=int, help="Número de nodos físicos")
    parser.add_argument("--cpus-per-node", type=int, help="CPUs por cada nodo")

    parser.add_argument(
        "--cluster-mem-gb",
        type=float,
        help="RAM total del clúster (GB). Si se especifica, también limita trials por RAM.",
    )
    parser.add_argument(
        "--reserve-mem-gb",
        type=float,
        default=2.0,
        help="RAM a reservar para SO/overhead (GB) cuando usas --cluster-mem-gb",
    )
    parser.add_argument(
        "--mem-gb-head",
        type=float,
        default=0.0,
        help="RAM estimada del head/driver por trial (GB). 0 = ignorar en el cálculo.",
    )
    parser.add_argument(
        "--mem-gb-per-worker",
        type=float,
        default=0.0,
        help="RAM estimada por worker por trial (GB). 0 = ignorar en el cálculo.",
    )

    parser.add_argument("--desired-trials", type=int, help="Trials paralelos objetivo")
    parser.add_argument("--num-workers", type=int, help="Workers de Ray Train por trial")
    parser.add_argument("--cpus-per-worker", type=int, help="CPUs por cada worker")
    parser.add_argument("--cpus-head", type=int, help="CPU para el driver del trial")
    args = parser.parse_args()

    env_cfg = ResourceConfig.from_env()
    cfg = ResourceConfig(
        num_workers=args.num_workers if args.num_workers is not None else env_cfg.num_workers,
        cpus_per_worker=args.cpus_per_worker if args.cpus_per_worker is not None else env_cfg.cpus_per_worker,
        cpus_head=args.cpus_head if args.cpus_head is not None else env_cfg.cpus_head,
    )

    cluster_cpus = args.cluster_cpus
    if args.cluster_nodes and args.cpus_per_node:
        if args.cluster_cpus == DEFAULT_CLUSTER_CPUS:
            cluster_cpus = args.cluster_nodes * args.cpus_per_node

    print_summary(
        total_cluster_cpus=cluster_cpus,
        reserve_cpus=args.reserve_cpus,
        cluster_mem_gb=args.cluster_mem_gb,
        reserve_mem_gb=args.reserve_mem_gb,
        mem_gb_head=args.mem_gb_head,
        mem_gb_per_worker=args.mem_gb_per_worker,
        cfg=cfg,
    )
    print_node_assessment(args.cluster_nodes, args.cpus_per_node, cfg, args.desired_trials)
    print_suggestions(
        total_cluster_cpus=cluster_cpus,
        reserve_cpus=args.reserve_cpus,
        desired_trials=args.desired_trials,
        cfg=cfg,
    )

    if args.desired_trials:
        needed = required_cluster_cpus(args.desired_trials, cfg)
        print(f"\nResultado final: Necesitas {needed} CPUs para {args.desired_trials} trials.")
