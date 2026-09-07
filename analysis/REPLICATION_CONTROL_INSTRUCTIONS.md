# Campagne IBM de confirmation et de contrôle CP3

Le script `run_ibm_replication_control.py` crée une nouvelle campagne. Il ne
modifie ni les scripts historiques, ni les anciens fichiers JSON.

## 1. Préparation locale, sans connexion IBM

Depuis la racine du dépôt ACRP complet, avec son environnement activé :

```bash
python3 -u analysis/run_ibm_replication_control.py prepare \
  --restarts 96 \
  --maxiter 5000
```

Cette étape récupère en priorité les angles en pleine précision du job
`da8aqcse74ec73aie27g` dans `results/`. Si ce JSON n'est pas disponible, elle
utilise les valeurs à six décimales du handover et le signale dans le plan.
Elle réoptimise ensuite un vrai `p=3`, applique les validations A à F et écrit
un nouveau plan `ibm_replication_control_plan_cp3_*.json`.

## 2. Répétition sans soumettre

```bash
python3 -u analysis/run_ibm_replication_control.py submit \
  --plan "results/ibm_replication_control_plan_cp3_<horodatage>.json"
```

Le script contacte IBM, vérifie `ibm_kingston`, choisit un layout physique
commun et affiche les huit circuits ISA. À l'invite, saisir `CANCEL` ne soumet
rien.

## 3. Soumission

Relancer exactement la commande précédente. Vérifier l'ordre suivant :

```text
p2_A, p6, p3_true, p2_B, p4, p3_null, p1, p2_C
```

Vérifier également 4096 shots par circuit et un layout commun, puis saisir
exactement `SUBMIT ALL`. Le script sauvegarde immédiatement un manifeste avec
le job ID et se termine sans attendre la file IBM.

## 4. Récupération

```bash
python3 -u analysis/run_ibm_replication_control.py retrieve \
  --manifest "results/ibm_replication_control_submission_cp3_<horodatage>.json"
```

La commande attend la fin du job si nécessaire, calcule les taux de
faisabilité, P(opt), les intervalles de Wilson et sauvegarde les distributions
brutes dans un nouveau JSON combiné. Aucun fichier existant n'est écrasé.
