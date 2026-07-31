#!/usr/bin/env bash
#
# ============================================================================
#  RUN DE NUIT : robustesse M = 3  (chiffres "qualite memoire")
# ============================================================================
#
#  But : reprendre la piste M = 3 simulable avec des reglages LOURDS (plus de
#  graines, profondeurs 1-4, plus de restarts, budget recuit plus grand) afin
#  d'obtenir des chiffres stables pour le memoire (sections 3.3 et 3.4).
#
#  SECURISE CONTRE LE BLOCAGE :
#    - Chaque bloc ecrit SON PROPRE CSV. Si le run est interrompu, tout ce qui
#      est deja termine est conserve (contrairement a un seul gros CSV ecrit
#      a la toute fin).
#    - QAOA plafonnee a n <= 6 (18 qubits). n = 7 (21q) est ce qui avait bloque
#      et n'est PAS inclus.
#    - Qiskit (ideal + bruite) seulement pour n <= 4 (<= 12 qubits).
#    - Blocs ordonnes du plus utile au plus lourd : si tout ne finit pas, les
#      resultats importants sont deja sauvegardes.
#
#  Prerequis : env virtuel actif, depuis la racine du projet.
#  RECOMMANDE (empeche le Mac de se mettre en veille, capot ferme) :
#      caffeinate -i bash run_overnight_m3.sh
#  Sinon simplement :
#      source .venv/bin/activate
#      bash run_overnight_m3.sh
# ============================================================================

# NB: pas de 'set -e' : on veut que les blocs suivants tournent meme si un bloc
# echoue. Chaque bloc est protege par '|| echo [warn] ...'.
set -uo pipefail
cd "$(dirname "$0")"
mkdir -p results
ts() { date "+%Y-%m-%d %H:%M:%S"; }

echo "================================================================"
echo " RUN DE NUIT M = 3 (robustesse)  -  demarre $(ts)"
echo "================================================================"

echo
echo "[1/3] n = 3,4  ETUDE DU BRUIT (qiskit ideal + bruite)  $(ts)"
echo "      5 graines, profondeurs 1-4  ->  results/overnight_m3_noise_n34.csv"
python3 analysis/benchmark.py \
    --sizes 3 4 --nh 3 --d 8 \
    --seeds 5 --depths 1 2 3 4 --restarts 16 \
    --random-per-size 2 \
    --qaoa-numpy-max-vars 12 --qaoa-qiskit-max-vars 12 \
    --qiskit --noisy --shots 8192 --qiskit-maxiter 150 \
    --sa-reads 80 --sa-sweeps 1500 \
    --out results/overnight_m3_noise_n34.csv \
    || echo "[warn] bloc bruit n=3,4 a echoue $(ts)"

echo
echo "[2/3] n = 3,4,5  ROBUSTESSE numpy ideal (jusqu'a 15q)  $(ts)"
echo "      10 graines, profondeurs 1-4, 48 restarts"
echo "      ->  results/overnight_m3_numpy_n345.csv"
python3 analysis/benchmark.py \
    --sizes 3 4 5 --nh 3 --d 8 \
    --seeds 10 --depths 1 2 3 4 --restarts 48 \
    --random-per-size 2 \
    --qaoa-numpy-max-vars 15 --no-qiskit \
    --sa-reads 80 --sa-sweeps 1500 \
    --out results/overnight_m3_numpy_n345.csv \
    || echo "[warn] bloc numpy n=3,4,5 a echoue $(ts)"

echo
echo "[3/3] n = 6  numpy ideal (18q, bloc le plus lourd, en dernier)  $(ts)"
echo "      6 graines, profondeurs 1-3, 24 restarts"
echo "      ->  results/overnight_m3_numpy_n6.csv"
python3 analysis/benchmark.py \
    --sizes 6 --nh 3 --d 8 \
    --seeds 6 --depths 1 2 3 --restarts 24 \
    --random-per-size 2 \
    --qaoa-numpy-max-vars 18 --no-qiskit \
    --sa-reads 80 --sa-sweeps 1500 \
    --out results/overnight_m3_numpy_n6.csv \
    || echo "[warn] bloc numpy n=6 a echoue $(ts)"

echo
echo "================================================================"
echo " RUN DE NUIT M = 3 termine  -  $(ts)"
echo " Fichiers : results/overnight_m3_noise_n34.csv,"
echo "            results/overnight_m3_numpy_n345.csv,"
echo "            results/overnight_m3_numpy_n6.csv"
echo "================================================================"
