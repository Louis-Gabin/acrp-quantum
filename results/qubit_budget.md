# Budget qubits et faisabilite de simulation

### Budget qubits, M = 3 (RAM de reference : 8 Go)

| n | qubits (n*M) | RAM statevecteur | Simulation QAOA (ideale) | Sim. bruitee locale | Temps local estime |
|---|---|---|---|---|---|
| 3 | 9 | 8 KB | Oui, local | oui | < 1 min |
| 4 | 12 | 64 KB | Oui, local | oui | < 1 min |
| 5 | 15 | 512 KB | Oui, local | non | 1-3 min |
| 6 | 18 | 4 MB | Oui, local | non | 5-15 min |
| 7 | 21 | 32 MB | Oui, local | non | 20-45 min |
| 8 | 24 | 256 MB | Oui, local | non | 1-3 h |
| 9 | 27 | 2.0 GB | Oui, local | non | 3-8 h (nuit) |
| 10 | 30 | 16.0 GB | HPC seulement | non | hors capacite locale |
| 12 | 36 | 1.0 TB | HPC seulement | non | hors capacite locale |
| 25 | 75 | ~6.0e23 octets | Impossible (toute machine) | non | hors capacite locale |
| 50 | 150 | ~2.3e46 octets | Impossible (toute machine) | non | hors capacite locale |

### Budget qubits, M = 5 (RAM de reference : 8 Go)

| n | qubits (n*M) | RAM statevecteur | Simulation QAOA (ideale) | Sim. bruitee locale | Temps local estime |
|---|---|---|---|---|---|
| 3 | 15 | 512 KB | Oui, local | non | 1-3 min |
| 4 | 20 | 16 MB | Oui, local | non | 20-45 min |
| 5 | 25 | 512 MB | Oui, local | non | 3-8 h (nuit) |
| 6 | 30 | 16.0 GB | HPC seulement | non | hors capacite locale |
| 7 | 35 | 512.0 GB | HPC seulement | non | hors capacite locale |
| 8 | 40 | 16.0 TB | HPC seulement | non | hors capacite locale |
| 9 | 45 | 512.0 TB | HPC seulement | non | hors capacite locale |
| 10 | 50 | ~1.8e16 octets | Impossible (toute machine) | non | hors capacite locale |
| 12 | 60 | ~1.8e19 octets | Impossible (toute machine) | non | hors capacite locale |
| 25 | 125 | ~6.8e38 octets | Impossible (toute machine) | non | hors capacite locale |
| 50 | 250 | ~2.9e76 octets | Impossible (toute machine) | non | hors capacite locale |
