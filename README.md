# Optimisation-des-Small-Language-Models
# Optimisation des Small Language Models pour le question-réponse médical

**Étude comparative de la précision numérique, de la quantification 4 bits et du fine-tuning QLoRA sur PubMedQA**

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-GPU-red)
![Hugging%20Face](https://img.shields.io/badge/Hugging%20Face-Transformers-yellow)
![License](https://img.shields.io/badge/Status-Research%20Project-orange)

## Présentation du projet

Les Small Language Models (SLM) représentent une alternative intéressante aux grands modèles de langage pour les applications spécialisées nécessitant un compromis entre qualité des prédictions, temps d'inférence et ressources matérielles.

Dans le domaine biomédical, leur utilisation soulève plusieurs questions : un modèle préentraîné peut-il répondre correctement à des questions scientifiques sans adaptation spécifique ? La quantification permet-elle de réduire les ressources nécessaires sans dégrader les performances ? Le fine-tuning avec un nombre limité de paramètres entraînables permet-il d'améliorer la qualité des prédictions ?

Ce projet étudie ces questions à travers une comparaison expérimentale de trois configurations de **Qwen2.5-1.5B-Instruct** sur le jeu de données PubMedQA :

- **FP16** : modèle de base utilisant des poids en demi-précision.
- **4-bit NF4** : modèle de base quantifié sur 4 bits.
- **QLoRA** : fine-tuning à faible rang sur un modèle quantifié en 4 bits.

L'objectif est de mesurer les compromis entre performance prédictive, latence d'inférence et mémoire GPU, à protocole d'évaluation constant.

> **Périmètre actuel :** les expériences terminées portent sur FP16, la quantification NF4 et QLoRA. La distillation de connaissances est identifiée comme une extension future, mais n'a pas encore été expérimentée.

## Résultats principaux

Les résultats ci-dessous proviennent du jeu de test indépendant de 200 exemples.

| Configuration | Accuracy | Macro-F1 | Latence médiane | Pic mémoire GPU |
|---|---:|---:|---:|---:|
| FP16 | 55,0 % | 0,2743 | 0,227 s | 4,59 Go |
| 4-bit NF4 | 53,5 % | 0,2714 | 0,306 s | 5,68 Go |
| **QLoRA** | **90,5 %** | **0,8321** | 0,275 s | 6,76 Go |

*Les latences et consommations mémoire sont celles observées sur l'environnement expérimental utilisé. Elles peuvent varier selon le matériel, les versions logicielles et les conditions d'exécution.*

### Observations

- QLoRA atteint une accuracy de **90,5 %**, soit 35,5 points de pourcentage de plus que le modèle FP16.
- Le Macro-F1 passe de 0,2743 en FP16 à 0,8321 avec QLoRA, ce qui indique une amélioration importante des performances entre les différentes classes.
- La quantification 4 bits seule ne produit pas d'amélioration notable sur cette tâche.
- QLoRA offre les meilleures performances prédictives parmi les trois configurations évaluées, mais son pic mémoire mesuré est le plus élevé.
- FP16 présente la latence médiane la plus faible dans cette expérience.

Ces résultats suggèrent que l'adaptation du modèle à la tâche est plus déterminante que la quantification seule pour la qualité des prédictions.

## Objectifs scientifiques

Le projet poursuit cinq objectifs :

1. Établir une référence de performance avec un modèle préentraîné sans fine-tuning.
2. Mesurer l'effet de la quantification 4 bits sur la qualité des prédictions.
3. Évaluer l'apport du fine-tuning QLoRA avec un nombre réduit de paramètres entraînables.
4. Comparer les configurations à l'aide de métriques prédictives et de mesures d'efficacité.
5. Identifier les limites expérimentales et préparer une éventuelle extension par distillation de connaissances.

## Modèle et données

### Modèle de langage

Le modèle retenu est [`Qwen/Qwen2.5-1.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct), un modèle instruction-tuned d'environ 1,5 milliard de paramètres.

Il est utilisé dans trois configurations : FP16, NF4 et QLoRA.

### Jeu de données

Le projet utilise [`qiaojin/PubMedQA`](https://huggingface.co/datasets/qiaojin/PubMedQA), dans la configuration `pqa_labeled`.

PubMedQA est un jeu de données de question-réponse biomédicales fondé sur des résumés de littérature scientifique. La tâche consiste ici à prédire l'une des trois décisions associées à une question :

- `yes`
- `no`
- `maybe`

Les entrées utilisées comprennent le contexte scientifique et la question. La sortie attendue est un label unique.

### Préparation des données

L'étude utilise les 1 000 exemples disponibles dans la configuration sélectionnée, répartis selon une stratégie stratifiée avec une graine aléatoire fixée à 42.

| Sous-ensemble | Nombre d'exemples | Proportion |
|---|---:|---:|
| Entraînement | 700 | 70 % |
| Validation | 100 | 10 % |
| Test | 200 | 20 % |
| **Total** | **1 000** | **100 %** |

La validation sert à suivre l'entraînement et à sélectionner le checkpoint QLoRA. Le jeu de test est réservé à l'évaluation finale.

**Important :** cette répartition est une séparation expérimentale personnalisée ; elle ne doit pas être présentée comme la séparation officielle du jeu PubMedQA.

## Méthodologie

### 1. Baseline FP16

Le modèle est chargé avec ses poids en demi-précision, sans adaptation sur les exemples d'entraînement.

Cette configuration sert de référence pour évaluer les performances du modèle préentraîné sur la tâche de classification biomédicale.

### 2. Quantification 4 bits NF4

La configuration utilise la quantification 4 bits de BitsAndBytes :

- Quantification : `NF4`
- Type de calcul : `float16`
- Double quantification : activée

La quantification vise à réduire la représentation mémoire des poids du modèle. Son effet sur les performances est mesuré sans fine-tuning supplémentaire.

### 3. Fine-tuning QLoRA

QLoRA combine un modèle de base quantifié en 4 bits et des adaptateurs LoRA entraînables.

Paramètres principaux utilisés :

| Paramètre | Valeur |
|---|---|
| Modèle de base | Qwen2.5-1.5B-Instruct |
| Quantification | NF4 |
| Rang LoRA (`r`) | 8 |
| Alpha LoRA | 16 |
| Dropout LoRA | 0,05 |
| Modules ciblés | `q_proj`, `k_proj`, `v_proj`, `o_proj` |
| Nombre d'époques | 2 |
| Learning rate | 2e-4 |
| Batch size par appareil | 1 |
| Gradient accumulation steps | 8 |
| Weight decay | 0,01 |
| Scheduler | Cosine |
| Longueur maximale | 1 024 tokens |
| Graine aléatoire | 42 |

Les paramètres entraînables représentent environ **0,141 %** des paramètres du modèle lors de la configuration de l'entraînement.

Le modèle a été entraîné sur 700 exemples, avec suivi de la loss de validation. Le checkpoint retenu est celui présentant la meilleure loss de validation.

### 4. Protocole d'évaluation

Les trois configurations sont évaluées avec le même prompt, les mêmes exemples de test et les mêmes paramètres de génération déterministes.

La génération est limitée à quatre nouveaux tokens, avec `do_sample=False`. Le premier label reconnu parmi `yes`, `no` et `maybe` est utilisé comme prédiction. Les réponses ne commençant pas par un label reconnu sont classées comme invalides.

## Métriques

Les indicateurs utilisés sont :

- **Accuracy** : proportion totale de prédictions correctes.
- **Macro-F1** : moyenne non pondérée des scores F1 des trois classes.
- **Latence médiane** : temps médian nécessaire à la génération d'une prédiction.
- **Taux de sorties invalides** : proportion des sorties pour lesquelles aucun label reconnu n'est trouvé au début de la réponse.
- **Pic de mémoire GPU allouée** : maximum de mémoire GPU suivie par PyTorch pendant l'évaluation.

Le Macro-F1 est particulièrement utile ici, car les classes ne sont pas représentées en proportions égales.

## Analyse détaillée des résultats

### Performances par classe sur le jeu de test

| Classe | FP16 — F1 | 4-bit NF4 — F1 | QLoRA — F1 |
|---|---:|---:|---:|
| `yes` | 0,72 | 0,71 | 0,93 |
| `no` | 0,03 | 0,03 | 0,95 |
| `maybe` | 0,08 | 0,07 | 0,62 |

Le modèle FP16 et le modèle NF4 ont tendance à privilégier la classe `yes`. Ils reconnaissent très peu les exemples `no` et `maybe`, ce qui explique leurs Macro-F1 faibles.

QLoRA améliore fortement la reconnaissance des trois classes. La classe `maybe` demeure toutefois plus difficile à prédire : son rappel est de 55 %, contre 96 % pour `yes` et 93 % pour `no`.

Cette observation montre pourquoi il est important d'examiner les métriques par classe plutôt que de se limiter à l'accuracy globale.

## Environnement technique

Les expériences ont été réalisées dans Google Colab avec un GPU NVIDIA Tesla T4.

Environnement observé :

- Python : version 3.10 ou ultérieure recommandée pour reproduire l'environnement.
- PyTorch : `2.11.0+cu130`
- Transformers : `5.20.1`
- PEFT : `0.21.2`
- BitsAndBytes : `0.50.2`
- GPU : NVIDIA Tesla T4
- Mémoire GPU disponible annoncée : environ 14,56 Go

Les versions indiquées décrivent l'environnement de l'expérience. Elles ne constituent pas nécessairement une combinaison universelle pour toutes les versions de Colab.

## Installation

Dans un environnement compatible avec un GPU CUDA, installe les dépendances :

```bash
pip install datasets transformers accelerate pandas scikit-learn matplotlib peft bitsandbytes
```

Vérifie ensuite la disponibilité du GPU :

```python
import torch

print("PyTorch :", torch.__version__)
print("CUDA disponible :", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU :", torch.cuda.get_device_name(0))
    print(
        "Mémoire totale :",
        round(
            torch.cuda.get_device_properties(0).total_memory
            / 1024**3,
            2
        ),
        "Go"
    )
```

Les API de Transformers et de PEFT évoluent. Pour une reproduction stricte, il est recommandé de figer les versions validées dans un fichier `requirements.txt`.

## Reproduire l'expérience

Le notebook expérimental suit le workflow suivant :

1. Installer les dépendances et vérifier le GPU.
2. Charger le jeu `qiaojin/PubMedQA`, configuration `pqa_labeled`.
3. Construire les splits stratifiés avec la graine 42.
4. Évaluer le modèle de base FP16 sur la validation.
5. Évaluer le modèle quantifié 4 bits sur la validation.
6. Entraîner les adaptateurs QLoRA sur le train et suivre la loss sur la validation.
7. Harmoniser le prompt et les paramètres de génération.
8. Comparer les trois configurations sur la validation.
9. Charger les checkpoints retenus et réaliser l'évaluation finale sur les 200 exemples de test.
10. Sauvegarder les prédictions, les métriques et les résultats de comparaison.

Les fichiers de résultats sont enregistrés sous `/content/slm_medical/`, notamment dans `final_test/`.

Le dossier `/content` étant temporaire dans Google Colab, il faut télécharger les résultats ou les copier vers Google Drive avant de fermer ou de réinitialiser la session.

## Structure du projet

La structure expérimentale attendue est la suivante :

```text
slm_medical/
├── experiment_config.json
├── pubmedqa_splits/
│   ├── train/
│   ├── validation/
│   └── test/
├── qlora_adapter/
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   └── tokenizer files...
├── qlora_checkpoints/
├── fp16_validation.csv
├── 4bit_validation.csv
├── qlora_validation.csv
└── final_test/
    ├── FP16_predictions.csv
    ├── 4-bit-NF4_predictions.csv
    ├── QLoRA_predictions.csv
    └── final_comparison.csv
```

Les fichiers exacts peuvent varier selon les cellules exécutées et les sauvegardes effectuées. Les fichiers CSV contiennent les prédictions et les latences par exemple ; le tableau récapitulatif contient les métriques agrégées.

## Limites de l'étude

Plusieurs limites doivent être prises en compte dans l'interprétation des résultats :

1. **Taille du jeu de données :** l'étude utilise 1 000 exemples de la configuration `pqa_labeled`, ce qui limite la portée des conclusions.
2. **Un seul split :** les résultats proviennent d'une séparation aléatoire stratifiée avec une graine fixée. Une validation croisée ou plusieurs graines permettraient d'étudier la variabilité.
3. **Classe `maybe` :** malgré l'amélioration de QLoRA, cette classe demeure plus difficile à reconnaître.
4. **Mesure de mémoire :** le pic suivi par PyTorch n'inclut pas nécessairement toutes les consommations système ou les allocations qui ne sont pas suivies par PyTorch.
5. **Mesure de latence :** les résultats dépendent du matériel, de l'environnement logiciel, du chargement et des paramètres de génération. Une comparaison de production demanderait également des mesures de débit et des tests répétés.
6. **Format des sorties :** le parseur valide la présence d'un label au début de la réponse ; il ne garantit pas que toute la réponse ne contienne que ce label.
7. **Sécurité médicale :** les résultats sont expérimentaux. Le modèle n'est pas validé pour fournir des diagnostics, des recommandations thérapeutiques ou des décisions cliniques.

Enfin, la distillation n'a pas été implémentée dans la version actuelle. Aucune conclusion comparative sur son efficacité ne peut donc être tirée de ces résultats.

## Pistes d'amélioration

Plusieurs extensions sont envisageables :

- Implémenter une distillation teacher–student et comparer ses performances à celles de QLoRA.
- Étudier plusieurs tailles de modèles étudiants.
- Répéter les expériences avec plusieurs graines aléatoires.
- Analyser les erreurs de classification, notamment pour la classe `maybe`.
- Tester la robustesse sur un autre jeu de données biomédical.
- Mesurer le débit, la mémoire en inférence et le temps d'entraînement.
- Améliorer le contrôle du format des sorties et évaluer strictement le respect du format attendu.

## Conclusion

Cette étude montre que, dans le protocole expérimental considéré, le fine-tuning QLoRA améliore nettement les performances de Qwen2.5-1.5B-Instruct sur la tâche de classification de questions biomédicales de PubMedQA.

La quantification NF4 seule ne suffit pas à obtenir un gain prédictif dans cette expérience. QLoRA atteint un Macro-F1 de 0,8321 sur le jeu de test, contre 0,2743 en FP16, tout en conservant une latence médiane inférieure à 0,3 seconde par exemple sur la Tesla T4 utilisée.

Les résultats mettent en évidence l'intérêt de l'adaptation efficace des modèles de langage, tout en soulignant la nécessité de tester leur robustesse, leur généralisation et leur comportement sur les classes les plus difficiles.

## Références

- **PubMedQA** — [Jeu de données sur Hugging Face](https://huggingface.co/datasets/qiaojin/PubMedQA)
- **Qwen2.5-1.5B-Instruct** — [Modèle sur Hugging Face](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)
- **QLoRA** — Dettmers et al., *QLoRA: Efficient Finetuning of Quantized LLMs* (2023), [arXiv:2305.14314](https://arxiv.org/abs/2305.14314)
- **LoRA** — Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models* (2021), [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)

---

*Projet expérimental réalisé avec Python, PyTorch, Hugging Face Transformers, PEFT et BitsAndBytes.*
