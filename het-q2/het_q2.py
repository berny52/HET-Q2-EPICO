# het_q2.py - VERSION CORREGIDA
import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector, entropy
from curvature_manual import calcular_curvatura_ricci
import warnings
warnings.filterwarnings('ignore')

# =============================================
# CONFIGURACION
# =============================================
plt.style.use('default')
SIMULATOR = AerSimulator()
SHOTS = 10000

class QuantumHET:
    def __init__(self, objetivo='110'):
        self.objetivo = objetivo  # Orden estandar (q2,q1,q0)
        # En Qiskit, el string '110' representa q0=0, q1=1, q2=1
        self.objetivo_qiskit = objetivo  # Mismo string, diferente interpretacion
        self.n_qubits = 3
        self.qr = QuantumRegister(self.n_qubits, 'q')
        self.cr = ClassicalRegister(self.n_qubits, 'c')
        self.qc = QuantumCircuit(self.qr, self.cr)
        self.snapshots = []
        
    def preparacion_inicial(self):
        """Superposicion inicial con entrelazamiento"""
        self.qc.h(self.qr)
        self.qc.cx(self.qr[0], self.qr[1])
        self.qc.cx(self.qr[1], self.qr[2])
        self._capturar_snapshot('Preparacion inicial')
        return self
    
    def oraculo_corregido(self):
        """Oraculo CORREGIDO para |110⟩ estandar"""
        # Para |110⟩ estandar (q2=1, q1=1, q0=0):
        # En Qiskit: q0=0, q1=1, q2=1 -> string '110'
        
        # 1. Convertir a |111⟩ aplicando X a q0
        self.qc.x(self.qr[0])  # q0: 0 -> 1
        
        # 2. Aplicar fase controlada (marca |111⟩)
        self.qc.h(self.qr[2])
        self.qc.ccx(self.qr[0], self.qr[1], self.qr[2])
        self.qc.h(self.qr[2])
        
        # 3. Restaurar estado original
        self.qc.x(self.qr[0])  # q0: 1 -> 0
        
        self._capturar_snapshot('Post-oraculo')
        return self
    
    def difusion_grover(self):
        """Operador de difusion estandar"""
        self.qc.h(self.qr)
        self.qc.x(self.qr)
        self.qc.h(self.qr[2])
        self.qc.ccx(self.qr[0], self.qr[1], self.qr[2])
        self.qc.h(self.qr[2])
        self.qc.x(self.qr)
        self.qc.h(self.qr)
        
        self._capturar_snapshot('Post-difusion')
        return self
    
    def _capturar_snapshot(self, etapa):
        """Capturar metricas del estado actual"""
        estado = Statevector(self.qc)
        snapshot = {
            'etapa': etapa,
            'curvatura': calcular_curvatura_ricci(estado),
            'fidelidad': self._calcular_fidelidad(estado),
            'entropia': self._calcular_entropia(estado)
        }
        self.snapshots.append(snapshot)
        return snapshot
    
    def _calcular_entropia(self, estado):
        """Calcular entropia de von Neumann"""
        if hasattr(estado, 'data'):
            estado = estado.data
        rho = np.outer(estado, np.conj(estado))
        return entropy(rho)
    
    def _calcular_fidelidad(self, estado):
        """Calcular fidelidad con estado objetivo"""
        if hasattr(estado, 'data'):
            estado = estado.data
        idx_objetivo = int(self.objetivo, 2)  # '110' -> 6
        return np.abs(estado[idx_objetivo])**2
    
    def ejecutar(self, iteraciones=2):
        """Ejecutar algoritmo cuantico completo"""
        print("INICIANDO HET-Q2...")
        print(f"Objetivo estandar: |{self.objetivo}>")
        print(f"Objetivo Qiskit: |{self.objetivo_qiskit}>")
        print(f"Iteraciones: {iteraciones}")
        print("=" * 60)
        
        self.preparacion_inicial()
        
        for i in range(iteraciones):
            print(f"Ejecutando iteracion {i+1}...")
            self.oraculo_corregido()
            self.difusion_grover()
        
        self.qc.measure(self.qr, self.cr)
        
        print("Ejecutando simulacion cuantica...")
        result = SIMULATOR.run(self.qc, shots=SHOTS).result()
        counts = result.get_counts()
        
        return counts, self.snapshots

# =============================================
# VISUALIZACION
# =============================================
def visualizar_resultados(snapshots, counts, objetivo, objetivo_qiskit):
    """Visualizacion cientifica de resultados"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'HET-Q2 - Analisis Cuantico |{objetivo}> (Qiskit: |{objetivo_qiskit}>)', fontsize=16)
    
    # 1. Evolucion de curvatura
    etapas = range(len(snapshots))
    curvaturas = [s['curvatura'] for s in snapshots]
    ax1.plot(etapas, curvaturas, 'o-', linewidth=3, markersize=8, color='blue')
    ax1.set_xlabel('Etapa')
    ax1.set_ylabel('Curvatura de Ricci')
    ax1.set_title('Evolucion de Geometria Cuantica')
    ax1.grid(True, alpha=0.3)
    
    # 2. Fidelidad
    fidelidades = [s['fidelidad'] for s in snapshots]
    ax2.plot(etapas, fidelidades, 's-', linewidth=3, markersize=8, color='green')
    ax2.set_xlabel('Etapa')
    ax2.set_ylabel('Fidelidad')
    ax2.set_title('Convergencia al Estado Objetivo')
    ax2.grid(True, alpha=0.3)
    
    # 3. Entropia
    entropias = [s['entropia'] for s in snapshots]
    ax3.plot(etapas, entropias, 'd-', linewidth=3, markersize=8, color='red')
    ax3.set_xlabel('Etapa')
    ax3.set_ylabel('Entropia')
    ax3.set_title('Evolucion de Entropia Cuantica')
    ax3.grid(True, alpha=0.3)
    
    # 4. Resultados de medicion
    estados = list(counts.keys())
    conteos = list(counts.values())
    # Colorear el estado objetivo en Qiskit
    colores = ['red' if estado == objetivo_qiskit else 'blue' for estado in estados]
    ax4.bar(estados, conteos, color=colores, alpha=0.8)
    ax4.set_xlabel('Estado Cuantico (Qiskit)')
    ax4.set_ylabel('Conteos')
    ax4.set_title('Distribucion de Resultados Finales')
    
    plt.tight_layout()
    plt.savefig('het_q2_results.png', dpi=300, bbox_inches='tight')
    plt.close()

# =============================================
# ANALISIS CIENTIFICO
# =============================================
def analisis_cientifico(snapshots, counts, objetivo, objetivo_qiskit):
    """Analisis cientifico completo"""
    print("\n" + "=" * 60)
    print("ANALISIS CIENTIFICO HET-Q2")
    print("=" * 60)
    
    ultimo = snapshots[-1]
    # Buscar el estado objetivo en la representacion de Qiskit
    prob_objetivo = (counts.get(objetivo_qiskit, 0) / SHOTS) * 100
    
    print(f"Objetivo estandar: |{objetivo}>")
    print(f"Objetivo Qiskit: |{objetivo_qiskit}>")
    print(f"Probabilidad objetivo: {prob_objetivo:.2f}%")
    print(f"Curvatura final: {ultimo['curvatura']:.6f}")
    print(f"Fidelidad final: {ultimo['fidelidad']:.6f}")
    print(f"Entropia final: {ultimo['entropia']:.6f}")
    
    # Analisis de eficiencia
    if prob_objetivo > 90:
        print("\nEVALUACION: Algoritmo altamente efectivo")
    elif prob_objetivo > 70:
        print("\nEVALUACION: Efectividad satisfactoria")
    else:
        print("\nEVALUACION: Efectividad suboptima - revisar oraculo")
    
    # Verificacion de coherencia
    if abs(prob_objetivo - ultimo['fidelidad']*100) < 5:
        print("COHERENCIA: Resultados consistentes entre simulacion y metricas")
    else:
        print("ADVERTENCIA: Inconsistencia entre simulacion y metricas")

# =============================================
# FUNCION DE VERIFICACION CORREGIDA
# =============================================
def verificar_oraculo(objetivo='110'):
    """Verificar que estados marca el oraculo - METODO CORREGIDO"""
    print(f"\nVERIFICANDO ORACULO PARA |{objetivo}>")
    print("=" * 40)
    
    # Metodo 1: Verificacion con estado base
    qc_test = QuantumCircuit(3)
    # Preparar el estado objetivo
    if objetivo[0] == '1':
        qc_test.x(2)
    if objetivo[1] == '1':
        qc_test.x(1)
    if objetivo[2] == '1':
        qc_test.x(0)
    
    # Aplicar oraculo
    # Para |110⟩ estandar (q2=1, q1=1, q0=0)
    qc_test.x(0)  # Convertir q0 a 1
    qc_test.h(2)
    qc_test.ccx(0, 1, 2)
    qc_test.h(2)
    qc_test.x(0)  # Restaurar q0
    
    # Medir
    qc_test.measure_all()
    
    result = SIMULATOR.run(qc_test, shots=1000).result()
    counts = result.get_counts()
    
    print("Verificacion con estado base:")
    for estado, count in sorted(counts.items(), key=lambda x: x[1], reverse=True):
        porcentaje = count / 1000 * 100
        print(f"|{estado}>: {count} veces ({porcentaje:.1f}%)")
    
    # Metodo 2: Verificacion con interferencia
    qc_test2 = QuantumCircuit(3)
    qc_test2.h([0, 1, 2])
    
    # Aplicar oraculo
    qc_test2.x(0)
    qc_test2.h(2)
    qc_test2.ccx(0, 1, 2)
    qc_test2.h(2)
    qc_test2.x(0)
    
    # Aplicar inversa de Hadamard
    qc_test2.h([0, 1, 2])
    qc_test2.measure_all()
    
    result2 = SIMULATOR.run(qc_test2, shots=1000).result()
    counts2 = result2.get_counts()
    
    print("\nVerificacion con interferencia:")
    for estado, count in sorted(counts2.items(), key=lambda x: x[1], reverse=True):
        porcentaje = count / 1000 * 100
        print(f"|{estado}>: {count} veces ({porcentaje:.1f}%)")
    
    # Verificar si el objetivo esta marcado
    if objetivo in counts2 and counts2[objetivo] > 500:
        print(f"\nORACULO CORRECTO: |{objetivo}> marcado correctamente")
    else:
        print(f"\nERROR: Oraculo no marca |{objetivo}> correctamente")

# =============================================
# EJECUCION PRINCIPAL
# =============================================
if __name__ == "__main__":
    print("HET-Q2 - ALGORITMO CUANTICO CON ORACULO CORREGIDO")
    print("=" * 70)
    
    # Primero verificar el oraculo
    verificar_oraculo('110')
    
    print("\n" + "=" * 70)
    print("EJECUTANDO ALGORITMO COMPLETO")
    print("=" * 70)
    
    quantum_het = QuantumHET(objetivo='110')
    counts, snapshots = quantum_het.ejecutar(iteraciones=2)
    
    print("\nRESULTADOS OBTENIDOS (Qiskit):")
    print(counts)
    
    analisis_cientifico(snapshots, counts, '110', quantum_het.objetivo_qiskit)
    
    print("\nGENERANDO VISUALIZACION...")
    visualizar_resultados(snapshots, counts, '110', quantum_het.objetivo_qiskit)
    
    print("\nHET-Q2 COMPLETADO!")
    print("Resultados guardados en: het_q2_results.png")