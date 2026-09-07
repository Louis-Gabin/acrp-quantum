# Série de profondeurs IBM pour CP₃

Cette version prépare et exécute une série contrôlée aux profondeurs
`p = 1, 2, 3, 4, 6` sur `ibm_kingston`.

Le script historique `analysis/run_ibm_hardware.py` ne doit pas être modifié.
Il reste la référence du premier run `p=1`.

## Principales garanties

- Les angles sont optimisés localement avec SciPy COBYLA et 48 redémarrages
  aléatoires par défaut.
- Les angles du premier run `p=1` sont ajoutés comme candidat déterministe.
- À chaque profondeur, la meilleure solution précédente est complétée par des
  couches nulles. L'énergie QUBO obtenue ne peut donc pas se dégrader à cause
  d'un mauvais redémarrage.
- `p=5` est optimisé localement comme pont entre `p=4` et `p=6`, mais il n'est
  pas inclus dans les circuits envoyés au QPU.
- Le plan local validé est écrit dans un fichier immuable, puis rechargé tel
  quel pour l'exécution IBM. Les angles ne sont pas réoptimisés entre les deux
  étapes.
- Les cinq circuits utilisent le même layout physique initial.
- Les cinq circuits sont envoyés dans un seul job SamplerV2.
- Le script sauvegarde l'identifiant du job immédiatement après la soumission.
- Le temps QPU est lu avec `job.usage()`. Il n'est pas confondu avec le temps
  mur du terminal.
- Aucun fichier de résultat existant n'est écrasé.
- Rien n'est soumis sans la saisie exacte de `SUBMIT ALL`.

## Étape 0 : installer les fichiers

Place ces deux fichiers dans le dossier `analysis` de ton véritable projet :

```text
analysis/run_ibm_depth_series.py
analysis/DEPTH_SERIES_INSTRUCTIONS.md
```

Ne remplace pas `analysis/run_ibm_hardware.py`.

## Étape 1 : préparer le plan local

Dans le terminal :

```bash
cd "/Users/louis-gabin/Library/CloudStorage/OneDrive-SKEMABusinessSchool/SKEMA/MASTER 2/S6/THESIS LOUIS/acrp-quantum 2"
source .venv/bin/activate

python3 -u analysis/run_ibm_depth_series.py \
  --local-only \
  --depths 1 2 3 4 6 \
  --restarts 48 \
  --maxiter 4000 \
  --seed 1
```

Cette commande ne crée pas de connexion IBM et ne peut consommer aucun quota.
L'optimisation peut prendre plusieurs minutes.

Le résultat doit se terminer par :

```text
VALIDATION PASSED
...
Saved immutable plan: .../ibm_depth_series_plan_cp3_<horodatage>.json
```

Le script bloque automatiquement si :

- SciPy ou COBYLA est indisponible ;
- aucune optimisation COBYLA ne converge ;
- le meilleur candidat retenu n'est pas issu d'une convergence COBYLA ;
- l'énergie QUBO remonte avec la profondeur ;
- le point `p=1` est moins bon en énergie que les angles du premier run.

La table locale contient six lignes. La ligne `p=5` porte le rôle `bridge` ;
les cinq autres portent le rôle `submit`. Seules les lignes `submit` seront
transpilées et envoyées à IBM.

Arrête-toi après cette étape et conserve le chemin exact du plan affiché.
Envoie la sortie complète pour vérification avant toute connexion IBM.

## Étape 2 : répétition IBM sans soumission

Remplace `<PLAN_JSON>` par le chemin exact imprimé à l'étape 1 :

```bash
python3 -u analysis/run_ibm_depth_series.py \
  --plan "<PLAN_JSON>" \
  --backend ibm_kingston \
  --shots 4096
```

Le script va :

1. vérifier l'intégrité du QUBO et recalculer les témoins statevector ;
2. vérifier que Kingston est opérationnel ;
3. choisir un layout à partir du circuit le plus profond ;
4. imposer ce même layout aux cinq circuits ;
5. transpiler les circuits ;
6. afficher le plan final de soumission.

À l'invite :

```text
Type exactly "SUBMIT ALL" ...
```

tape volontairement :

```text
CANCEL
```

Tu dois obtenir :

```text
Submission cancelled. No QPU job was sent.
```

Cette répétition contacte IBM pour lire le backend, mais ne soumet aucun job et
ne consomme pas de temps QPU.

## Étape 3 : vraie soumission

Après validation de la sortie de l'étape 2, relance exactement la même commande.
Vérifie une dernière fois :

- backend `ibm_kingston` ;
- 9 qubits logiques ;
- 5 circuits dans un seul job ;
- profondeurs `1, 2, 3, 4, 6` ;
- 4096 shots par circuit ;
- layout physique identique pour toute la série ;
- dynamical decoupling et twirling désactivés.

Tape ensuite exactement :

```text
SUBMIT ALL
```

Le script écrit immédiatement un manifeste contenant le `job_id`. Même si le
terminal est ensuite interrompu, le job reste récupérable sur la plateforme IBM.

## Fichiers produits

Tous les noms contiennent un identifiant de campagne horodaté :

```text
results/ibm_depth_series_submission_cp3_<campagne>.json
results/ibm_hardware_cp3_p1_<campagne>.json
results/ibm_hardware_cp3_p2_<campagne>.json
results/ibm_hardware_cp3_p3_<campagne>.json
results/ibm_hardware_cp3_p4_<campagne>.json
results/ibm_hardware_cp3_p6_<campagne>.json
results/ibm_depth_series_cp3_<campagne>.json
```

Le résumé combiné contient notamment :

- les angles exacts ;
- les profondeurs ISA et nombres de portes à deux qubits ;
- les layouts initial et final ;
- le taux de faisabilité et `P(opt)` ;
- les nombres absolus de solutions faisables et optimales ;
- les intervalles de Wilson, explicitement limités à l'incertitude de tirage ;
- le temps QPU total rapporté par IBM ;
- les versions de Python, NumPy, SciPy, Qiskit et Qiskit Runtime.

La série doit être interprétée à partir des résultats réels et de leurs témoins
statevector. Le script ne produit volontairement aucune « fidélité prédite » à
partir d'un simple produit d'erreurs moyennes.

## CP₁₀

Ne lance pas CP₁₀ avec ce script. CP₁₀ nécessite un protocole séparé : optimum
Gurobi, absence d'énumération exhaustive, méthode documentée de transfert des
angles et nouvelle estimation du coût matériel.
