# curvature_manual.py - VERSIÓN MEJORADA HET-Q2
import numpy as np
import networkx as nx
from itertools import combinations

def calcular_curvatura_ricci(estado_cuantico, alpha=0.5):
    """
    Calcula curvatura de Ricci-Ollivier para estado cuántico
    ¡AHORA CON MANEJO CORRECTO DE STATEVECTOR!
    """
    # Convertir Statevector a array numpy si es necesario
    if hasattr(estado_cuantico, 'data'):
        probs = np.abs(estado_cuantico.data)**2
    else:
        probs = np.abs(estado_cuantico)**2
    
    # Normalizar probabilidades (IMPORTANTE)
    probs = probs / np.sum(probs)
    
    n_qubits = int(np.log2(len(probs)))
    n_states = len(probs)
    
    # Grafo de estados con distancia Hamming
    G = nx.Graph()
    estados = [format(i, f'0{n_qubits}b') for i in range(n_states)]
    
    # Conectar estados con distancia Hamming = 1 (vecinos)
    for i, estado_i in enumerate(estados):
        for j, estado_j in enumerate(estados):
            if i < j:
                hamming_dist = sum(bit_i != bit_j for bit_i, bit_j in zip(estado_i, estado_j))
                if hamming_dist == 1:
                    G.add_edge(i, j, weight=hamming_dist)
    
    # Calcular curvatura para cada arista
    curvaturas = []
    for edge in G.edges():
        u, v = edge
        d_uv = G[u][v]['weight']
        
        # Distribuciones de masa (simplificado)
        mu = probs[u] + alpha * (1 - probs[u])
        mv = probs[v] + alpha * (1 - probs[v])
        
        # Distancia Wasserstein aproximada
        wasserstein_approx = abs(mu - mv) * d_uv
        
        # Curvatura de Ricci-Ollivier
        if d_uv > 0:
            curvatura = 1 - (wasserstein_approx / d_uv)
            curvaturas.append(curvatura)
    
    return np.mean(curvaturas) if curvaturas else 0.0