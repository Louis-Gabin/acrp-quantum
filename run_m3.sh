#!/usr/bin/env bash
#
# ============================================================================
#  PISTE 1 : M = 3  (jeu de manoeuvres reduit)
# ============================================================================
#
#  A M = 3, il existe des tailles reellement SIMULABLES en QAOA sur ton Mac,
#  donc on fait tout : classique (Gurobi + recuit) sur les grosses instances,
#  ET QAOA simule sur la petite echelle. C'est la piste "on fait le maximum
#  simulable".
#
#  IMPORTANT (budget calcul sur 8 Go) :
#    - L'echelle QAOA est PLAFONNEE a n = 6 (18 qubits). n = 7 (21 qubits)
#      demande des heures et n'est pas pratique en local ; il releve du
#      classique + IBM (voir budget qubits). Tu peux le tenter a la main la
#      nuit si tu veux, mais ce n'est pas dans ce script.
#    - seeds / profondeurs / restarts sont volontairement REDUITS pour que le
#      run se termine en un temps raisonnable. Augmente-les si tu veux plus de
#      robustesse statistique (au prix du temps).
#    - benchmark.py n'ecrit son CSV qu'a la FIN : laisse-le finir chaque etape.
#
#  Prerequis : environnement virtuel actif, depuis la racine du projet.
#    source .venv/bin/activate
#    bash run_m3.sh
#
#  Les resultats sont ecrits dans results/ (suffixe _m3).
# ============================================================================

set -euo pipefail
cd "$(dirname "$0")"

echo "============================================================"
echo " PISTE M = 3"
echo "============================================================"

echo
echo "[0/3] Budget qubits (M = 3) ------------------------------------"
python3 analysis/qubit_budget.py --sizes 3 4 5 6 7 8 9 10 12 25 50

echo
echo "[1/3] Campagne CLASSIQUE grosses instances (Gurobi + recuit) ---"
echo "      n = 10, 12, 25, 50   ->  results/scaling_large_m3.csv"
python3 analysis/scaling_large.py --sizes 10 12 25 50 --nh 3 --threads 4
mv -f results/scaling_large.csv results/scaling_large_m3.csv 2>/dev/null || true
mv -f results/scaling_large.png results/scaling_large_m3.png 2>/dev/null || true

echo
echo "[1b/3] Campagne CLASSIQUE - JEU FAISABLE (courbes propres) -----"
echo "      densite basse + filtre de faisabilite Gurobi"
echo "      -> results/scaling_feasible_m3.csv"
python3 analysis/scaling_large.py --sizes 10 12 25 50 --nh 3 --threads 4 \
    --feasible-only \
    --out results/scaling_feasible_m3.csv

echo
echo "[2/3] Echelle QAOA SIMULEE (n = 3..6, reglages alleges) --------"
echo "      numpy ideal jusqu'a n=6 (18q) ; bruite seulement n<=4 (12q)."
python3 analysis/benchmark.py \
    --sizes 3 4 5 6 \
    --nh 3 --d 8 \
    --seeds 5 --depths 1 2 3 --restarts 16 \
    --random-per-size 1 \
    --qaoa-numpy-max-vars 18 \
    --qaoa-qiskit-max-vars 12 \
    --qiskit --noisy --shots 2048 --qiskit-maxiter 60 \
    --out results/benchmark_m3.csv

echo
echo "[3/3] Tests statistiques ---------------------------------------"
python3 analysis/stats_tests.py || echo "[info] stats_tests optionnel; a lancer une fois les CSV prets."

echo
echo "Termine (M = 3). Fichiers : results/scaling_large_m3.csv,"
echo "results/benchmark_m3.csv, results/qubit_budget_m3.csv."
