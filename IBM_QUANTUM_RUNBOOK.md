# Runbook : exécuter le QAOA de l'ACRP sur IBM Quantum (Section 3.7)

> Objectif : lancer une seule exécution confirmatoire sur un vrai QPU IBM pour l'instance `CP_3` (9 qubits, p=1), avec les angles pré-optimisés sur simulateur, puis comparer taux de faisabilite et P(opt) aux simulations idéale et bruitée de la Section 3.4.
>
> Etat verifie (2026) : la plateforme IBM Quantum a migré sur IBM Cloud ; l'ancien canal `channel="ibm_quantum"` est retiré. On utilise desormais une cle API IBM Cloud + un CRN d'instance, et le canal `ibm_quantum_platform`.

---

## 0. Ce qu'il faut savoir avant de commencer

- **Plan Open (gratuit)** : 10 minutes de temps QPU par fenetre glissante de 28 jours, region **us-east** uniquement. C'est du temps d'**exécution** facturé, pas du temps d'attente en file. Une exécution de 9 qubits / 4096 shots consomme de l'ordre de quelques secondes a une minute de QPU, donc le budget suffit pour plusieurs essais.
- **Promo en cours (mars 2026)** : les comptes Open ayant consomme 20 min sur 12 mois peuvent debloquer 180 min supplementaires sur 12 mois. A activer depuis le dashboard si eligible.
- **Carte bancaire** : la creation d'un compte IBM Cloud demande une carte pour verification d'identite. En restant sur le plan Open, il n'y a **aucun debit**.
- **Versions** : `qiskit-ibm-runtime >= 0.45` exige `qiskit >= 2.0`. Ton `requirements.txt` epingle `qiskit>=1.2` : il faut passer a Qiskit 2.x pour la partie hardware. Le code de simulation existant (base sur `Statevector`) fonctionne sans changement sous Qiskit 2.x.

---

## 1. Creer le compte et l'instance

1. Aller sur <https://quantum.cloud.ibm.com> et se connecter / creer un compte (Google ou IBMid acceptes).
2. Choisir la region **us-east** (menu deroulant en haut) : le plan Open n'existe que la.
3. Creer une **instance** de service quantique sur le **plan Open** (elle apparait dans le dashboard). Elle est identifiee par un **CRN** (Cloud Resource Name) de la forme :
   ```
   crn:v1:bluemix:public:quantum-computing:us-east:a/....:....::
   ```

## 2. Recuperer la cle API et le CRN

1. **Cle API** : <https://cloud.ibm.com/iam/apikeys> -> "Create" -> copier la cle (elle ne s'affiche qu'une fois).
2. **CRN** : sur le dashboard IBM Quantum, ouvrir l'instance Open et copier son "Cloud Resource Name (CRN)".

> Garde ces deux chaines de cote. Ne les commit jamais dans le repo Git.

## 3. Installer / mettre a jour l'environnement (sur ton Mac)

```bash
cd ~/Library/CloudStorage/OneDrive-SKEMABusinessSchool/SKEMA/MASTER\ 2/S6/THESIS\ LOUIS/acrp-quantum\ 2
source .venv/bin/activate
pip install --upgrade "qiskit>=2.0" "qiskit-ibm-runtime>=0.40" qiskit-aer scipy numpy
```

Verifier :
```bash
python3 -c "import qiskit, qiskit_ibm_runtime; print(qiskit.__version__, qiskit_ibm_runtime.__version__)"
```

## 4. Enregistrer le compte (une seule fois)

Dans un shell Python (`python3`) :
```python
from qiskit_ibm_runtime import QiskitRuntimeService
QiskitRuntimeService.save_account(
    channel="ibm_quantum_platform",   # nouveau canal unifie (ibm_cloud fonctionne aussi)
    token="COLLE_TA_CLE_API",
    instance="COLLE_TON_CRN",
    set_as_default=True,
    overwrite=True,
)
```
Test de connexion :
```python
from qiskit_ibm_runtime import QiskitRuntimeService
service = QiskitRuntimeService()
print([b.name for b in service.backends(operational=True, simulator=False)])
```

## 5. Lancer l'exécution hardware

Le script `analysis/run_ibm_hardware.py` (fourni) fait tout le pipeline. Il :
1. reconstruit le meme QUBO `CP_3` M=3 que la campagne,
2. optimise (gamma, beta) sur le **simulateur local** (gratuit),
3. calcule l'optimum de reference (brute force 2^9, pour P(opt)),
4. construit le circuit final fixe + mesures,
5. choisit le QPU le moins occupe (ou celui force par `--backend`),
6. transpile en circuit ISA pour ce backend (`optimization_level=3`),
7. echantillonne via **SamplerV2** en un seul job,
8. calcule taux de faisabilite et P(opt), recupere une photo de calibration, et ecrit `results/ibm_hardware_cp3.json`.

```bash
python3 analysis/run_ibm_hardware.py --n 3 --heading 3 --p 1 --shots 4096
```

Pour cibler un QPU precis (ex. Heron r2) :
```bash
python3 analysis/run_ibm_hardware.py --backend ibm_kingston --shots 4096
```

## 6. Ce que tu recuperes pour la Section 3.7

Le JSON `results/ibm_hardware_cp3.json` contient exactement les champs du placeholder de la Section 3.7 :

| Champ du placeholder 3.7 | Cle dans le JSON |
|---|---|
| Nom du backend et nb de qubits | `backend`, `num_qubits_device` |
| Photo de calibration (T1/T2, erreurs) | `calibration` (+ a completer depuis l'UI) |
| Profondeur du circuit transpile | `depth_transpiled`, `two_qubit_gates` |
| Nombre de shots | `shots` |
| Probabilite de succes P(opt) | `p_opt` |
| Taux de faisabilite | `feasibility_rate` |
| A mettre en regard du simulateur | valeurs ideale/bruitee de la Section 3.4 |

Renvoie-moi ce JSON et je remplis le tableau + la prose de la Section 3.7 (et je retire le dernier placeholder).

## 7. Bonnes pratiques budget et pieges

- **Un seul job a la fois.** Garde `--shots` a 4096 (suffisant a 9 qubits). N'optimise jamais les angles sur le QPU.
- **File d'attente != temps facture.** L'attente peut durer des minutes ; seul le temps d'exécution compte dans les 10 min.
- **Suivi de conso** : dashboard -> page "Workloads" / "Usage".
- **Circuit ISA obligatoire** : SamplerV2 refuse un circuit non transpile pour le backend. Le script s'en charge via `generate_preset_pass_manager`.
- **Nom du registre** : `measure_all()` cree un registre `meas`, d'ou `result[0].data.meas.get_counts()`.
- **Reproductibilite** : les angles, le job_id et la calibration sont enregistres dans le JSON pour l'annexe "Use of AI" et la tracabilite.

---

## 8. Ce qu'il reste a faire sur la these (hors IBM)

**Bloquant / important**
1. **Bibliographie** : verifier les references (DOI deja collectes), fournir les PDF, viser >= 20 articles a comite de lecture + 1 livre.
2. **URL GitHub dans `main.tex`** : encore `acrp-lib`, a corriger en `acrp-quantum` (le vrai nom du repo).
3. **Annexe obligatoire "Use of AI"** : a rediger (exigence SKEMA), + consolidation des annexes 1.A / 2.A avant les References.
4. **Section 3.7** : l'exécution IBM ci-dessus (seul placeholder restant du Chapitre 3).

**Finalisation (quand le fond est stabilise)**
5. Fondre les encadres bleus "In plain terms" dans la prose (ils restent tant qu'on est en brouillon).
6. Verifier la longueur (>= 50 pages) une fois la mise en page figee.
7. Chapitre 2 : confirmer que la version "neophyte" te convient.
8. Redaction d'un resume / memoire sommaire.

**Optionnel / a discuter avec David**
9. Baseline recuit quantique emule D-Wave (`dwave-neal`) si tu veux un 5e solveur.
10. Omnibus Friedman + Nemenyi multi-solveurs (le script actuel ne compare que 2 methodes a la fois) si un correcteur le demande.
11. Clause SKEMA "rigueur atypique / equivalente" a clarifier.
12. Elargir l'enveloppe de controle (caps de virage / vitesse) pour rendre faisables les grandes instances denses : choix scientifique a arbitrer.
