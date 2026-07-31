#!/usr/bin/env bash
#
# ============================================================================
#  PISTE 2 : M = 5  (jeu de manoeuvres complet)
# ============================================================================
#
#  A M = 5, le nombre de qubits (n * 5) explose : n = 10 -> 50 qubits,
#  n = 50 -> 250 qubits. La simulation QAOA exacte devient IMPOSSIBLE sur
#  toute machine bien avant les tailles cibles (voir le budget qubits).
#  Cette piste sert donc a :
#    1) faire la campagne CLASSIQUE (Gurobi + recuit) sur n = 10,12,25,50,
#    2) faire la petite echelle QAOA encore simulable (n = 3, 4),
#    3) PROUVER l'infaisabilite de la simulation par le compte de qubits,
#       ce qui justifie le passage au materiel reel IBM Quantum (section 3.7).
#
#  IMPORTANT (budget calcul sur 8 Go) :
#    - L'echelle QAOA est PLAFONNEE a n = 4 (20 qubits). n = 5 (25 qubits)
#      est un run de nuit possible mais lourd ; lance-le a la main si tu veux.
#    - PAS de simulation BRUITEE ici : a M = 5, n >= 3 fait >= 15 qubits, et
#      la simulation bruitee (matrice densite) demanderait 2^(2*15) = 16 Go
#      des n = 3. L'etude du bruit est donc faite dans la piste M = 3.
#    - benchmark.py n'ecrit son CSV qu'a la FIN : laisse-le finir.
#
#  Prerequis : environnement virtuel actif, depuis la racine du projet.
#    source .venv/bin/activate
#    bash run_m5.sh
# ============================================================================

set -euo pipefail
cd "$(dirname "$0")"

echo "============================================================"
echo " PISTE M = 5"
echo "============================================================"

echo
echo "[0/3] Budget qubits (M = 5) : la PREUVE d'infaisabilite --------"
python3 analysis/qubit_budget.py --sizes 3 4 5 6 7 8 9 10 12 25 50

echo
echo "[1/3] Campagne CLASSIQUE grosses instances (Gurobi + recuit) ---"
echo "      n = 10, 12, 25, 50   ->  results/scaling_large_m5.csv"
python3 analysis/scaling_large.py --sizes 10 12 25 50 --nh 5 --threads 4
mv -f results/scaling_large.csv results/scaling_large_m5.csv 2>/dev/null || true
mv -f results/scaling_large.png results/scaling_large_m5.png 2>/dev/null || true

echo
echo "[1b/3] Campagne CLASSIQUE - JEU FAISABLE (courbes propres) -----"
echo "      densite basse + filtre de faisabilite Gurobi"
echo "      -> results/scaling_feasible_m5.csv"
python3 analysis/scaling_large.py --sizes 10 12 25 50 --nh 5 --threads 4 \
    --feasible-only \
    --out results/scaling_feasible_m5.csv

echo
echo "[2/3] Echelle QAOA SIMULEE (n = 3, 4 ; numpy ideal seulement) ---"
python3 analysis/benchmark.py \
    --sizes 3 4 \
    --nh 5 --d 8 \
    --seeds 5 --depths 1 2 3 --restarts 16 \
    --random-per-size 1 \
    --qaoa-numpy-max-vars 20 \
    --no-qiskit \
    --out results/benchmark_m5.csv

echo
echo "[3/3] Conclusion M = 5 -----------------------------------------"
cat <<'EOF'
  A M = 5, les instances cibles demandent 50 a 250 qubits. Aucune
  simulation classique (statevecteur) ne peut les traiter : la memoire
  requise depasse toute machine existante (voir budget qubits ci-dessus).
  => La seule voie "quantique" pour ces tailles est le MATERIEL REEL
     IBM Quantum, jusqu'a la limite de fidelite/nombre de qubits de la
     machine disponible (section 3.7 du memoire).
EOF

echo
echo "Termine (M = 5). Fichiers : results/scaling_large_m5.csv,"
echo "results/benchmark_m5.csv, results/qubit_budget_m5.csv."
