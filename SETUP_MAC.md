# Running the project on macOS

Step-by-step guide to run everything on your Mac, including the parts that could
not run in the assistant sandbox (Qiskit QAOA, noisy simulation, and the Gurobi
baseline).

## 1. Prerequisites

- macOS (Apple Silicon or Intel).
- Python 3.10 to 3.12 recommended (3.13 also works for the core; some quantum
  packages lag on the newest Python, so 3.11/3.12 is the safe choice).

Check your Python:

```bash
python3 --version
```

If you do not have a suitable Python, install it with Homebrew:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
brew install python@3.12
```

## 2. Get the project and create a virtual environment

```bash
cd ~/Downloads
unzip acrp-quantum.zip
cd acrp-quantum

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

## 3. Run the offline core first (no quantum deps needed)

This confirms the pipeline works before installing anything heavy:

```bash
pip install numpy matplotlib pandas
python3 src/run_demo.py
```

Expected: on CP_3 and CP_4 the simulated annealing energy matches the exact QUBO
optimum (gap 0); a figure is written to `results/CP_5_solution.png`.

## 4. Install the full stack (Qiskit + SciPy)

```bash
pip install -r requirements.txt
```

Then run QAOA:

```bash
# Dependency-free exact QAOA (fast on small instances):
python3 src/run_qaoa.py --n 3 --p 1
python3 src/run_qaoa.py --n 3 --p 2

# Thesis Qiskit implementation (noiseless):
python3 src/run_qaoa.py --n 3 --p 2 --backend qiskit

# Qiskit noisy campaign:
python3 src/run_qaoa.py --n 3 --p 2 --backend qiskit --noisy
```

And the full comparison table (exact vs simulated annealing vs QAOA):

```bash
python3 src/run_benchmark.py
# -> prints a table and writes results/benchmark.csv
```

The benchmark auto-detects Qiskit: if it is installed, a `qaoa_qiskit` row is
added; otherwise it is skipped with a message.

### Cross-check (reproducibility control)

The numpy QAOA and the Qiskit QAOA should agree on the expected energy for the
same (gamma, beta). This is a good sanity test to mention in the methodology
chapter.

## 5. Classical exact baseline (Gurobi)

David Rey's baseline lives in `baseline_gurobi/`.

```bash
pip install gurobipy   # needs Gurobi >= 12
```

Get a free academic licence at https://www.gurobi.com/academia/ and run
`grbgetkey` as instructed. Then open `baseline_gurobi/ACRP_gurobi.ipynb`
(you will need the `.dat` instance files; ask David or export from `acrp-lib`).

Important: the model uses `gurobipy.nlfunc.cos/sin`, which exists only in
Gurobi >= 12. On an older version the import fails.

What to check once it runs: on a shared small instance, the Gurobi optimum
(number of manoeuvres) should equal the exact brute-force QUBO optimum from
`run_benchmark.py`. That equality validates that the QUBO and the MINLP describe
the same problem.

## 6. Optional: emulated quantum annealing (D-Wave)

```bash
pip install dwave-neal
```

Then `neal.SimulatedAnnealingSampler` can sample the same QUBO (build it with
`src/qubo.py` and pass `qubo.Q`). This is the "Could" item in the roadmap.

## Troubleshooting

- `ModuleNotFoundError`: make sure the virtual environment is active
  (`source .venv/bin/activate`) and run scripts from the project root.
- Qiskit install issues on Python 3.13: recreate the venv with Python 3.12.
- The numpy QAOA gets slow beyond ~16 variables (2^V statevector). Keep it to
  small instances; use the Qiskit sampler for larger ones.

