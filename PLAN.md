# LFM3-1.7B + Engram : plan des diagnostics et du POC

Rédigé le 17 novembre 2026, avant les diagnostics.

> **Mise à jour du même jour.** Les diagnostics D0 à D4 ont été lancés. Ils écartent l'Engram classique et font émerger une autre architecture : voir [RESULTS.md](RESULTS.md), qui remplace les sections 5 à 7 de ce plan.

## 1. Résumé

**Objectif.** Rendre LFM3-1.7B (le modèle instruct) meilleur en connaissances, sans toucher à ses poids et sans refaire de SFT, en lui branchant une mémoire de type Engram.

**Pourquoi les connaissances.** C'est son point faible mesuré, et c'est exactement ce qu'un Engram stocke (section 3).

**Le problème.** Avec 60 $ de crédits on dispose d'environ 26 h d'A100, soit de l'ordre de 100 à 130 millions de tokens par run. Les travaux qui ajoutent une mémoire à un modèle existant en utilisent 200 milliards pour gagner 1 point. À 1 milliard de tokens, un Engram classique sur Gemma n'a rien gagné.

**Notre pari.** Ne pas apprendre la table par descente de gradient. On la remplit directement :

1. on compte les n-grammes sur des milliards de tokens, sur CPU, ce qui ne coûte presque rien ;
2. on écrit ces statistiques dans la table par une formule, sans entraînement ;
3. le GPU ne sert qu'à apprendre les portes, c'est-à-dire quand faire confiance à la mémoire. Ce sont peu de paramètres, donc peu de tokens suffisent.

**Avant de construire.** Six diagnostics, pour environ 13 $, disent si le pari a une chance et où brancher la mémoire. Deux d'entre eux peuvent arrêter le projet tôt.

**Critère de réussite, fixé d'avance.** L'Engram bat un LoRA entraîné sur les mêmes données et le même nombre de tokens, sur les connaissances, sans dégrader le suivi d'instructions, les appels d'outils ni la capacité du modèle à dire « je ne sais pas ».

## 2. Ce qu'on sait, et ce que ça impose

| Fait vérifié | Source | Conséquence pour nous |
|---|---|---|
| Engram entraîné dès le départ : +3,0 MMLU, +5,0 BBH, pour un écart de perte de seulement 0,012 nat | papier Engram, table 1 | Un petit écart de perte peut suffire. Seuil de décision à 0,02 nat. |
| Couper l'Engram fait tomber les questions factuelles à 29 à 44 % du score, la compréhension de texte reste à 81 à 93 % | papier Engram, analyse | L'Engram stocke des faits. Il n'améliorera ni le raisonnement ni le code agentique. |
| Dans les ablations d'Engram, la porte contextuelle et la compression du tokenizer comptent beaucoup, la conv interne très peu, les 4-grammes n'aident pas à capacité fixe | papier Engram, ablations | On garde la porte et la compression, on retire la conv. |
| Meilleure couche unique : la 2, juste après une première attention | papier Engram | Vrai pour un modèle entraîné avec. À revérifier sur un modèle gelé (diagnostic D3). |
| Écrire un fait dans un modèle gelé ne marche que près de la fin du réseau ; en couche basse le rappel tombe à un quart | User as Engram | Contredit le point précédent dans notre cas. D3 tranche. |
| Gemma 4 E2B + Engram, 1,05 milliard de tokens d'anglais général : gain de 0,001 nat, donc nul. Gains concentrés sur les langues peu couvertes. | RFC NeMo | Du texte général ne donnera rien. Il faut viser ce que le modèle ne sait pas. |
| Même source : table seule, modèle gelé, récupère environ 55 % du gain d'un fine-tuning complet. Couper la table rend le modèle d'origine exactement. | RFC NeMo | Le mode « modèle gelé » est viable et réversible. |
| Même source : si la table et la projection de sortie démarrent toutes deux à zéro, le gradient est nul et rien n'apprend | RFC NeMo | Seule l'échelle de sortie démarre à zéro. |
| STAPLE sur OLMo-2 1B : +1,2 point de moyenne avec 200 milliards de tokens et 9,9 milliards de paramètres de table. Sa variante façon Engram : +0,4. | STAPLE, table 1 | L'approche « tout apprendre par gradient » est hors budget d'un facteur 1000. |
| À 0,9 milliard de paramètres, l'Engram classique fait moins bien qu'un MoE, à cause des collisions de hachage | Memory Grafting | Table exacte, sans collision, pour les clés importantes. |
| Des tables initialisées depuis les états d'un autre modèle, puis gelées, font +1,4 point par rapport à l'Engram classique | Memory Grafting | Remplir la table autrement que par gradient fonctionne. |
| Mélanger un modèle de comptage de n-grammes avec un LLM réduit sa perplexité, mais dégrade la génération libre si le mélange est aveugle | Infini-gram | Les comptages contiennent de l'information utile. Il faut une porte qui décide quand s'en servir. |

Faits propres à LFM3, lus dans son code et sa fiche :

- **Structure.** 24 couches, taille cachée 2048. Les couches 2, 6, 10, 14, 18, 22 sont des attentions globales, les couches 4, 8, 12, 16, 20 des attentions à fenêtre de 256, les 13 autres des convolutions de largeur 7.
- **Embeddings liés.** La matrice d'entrée sert aussi de matrice de sortie. On peut donc traduire « augmenter la probabilité du token X » en un vecteur à ajouter en fin de réseau.
- **Normalisation double.** Chaque bloc normalise son entrée et sa sortie. L'amplitude de ce qu'on injecte doit donc être calée sur celle du flux à cet endroit.
- **Base et instruct jumeaux.** Même tokenizer (hash identique), mêmes dimensions. Base a déjà vu le format chat pendant son pré-entraînement.
- **Entraînement.** Les noyaux Triton de la conv ne servent qu'en inférence. À l'entraînement c'est du PyTorch standard, donc pas de surprise de vitesse selon le GPU.
- **Recette officielle.** Liquid fournit un notebook d'entraînement sur texte brut directement sur le modèle instruct : LoRA de rang 64, taux d'apprentissage 2e-4, lignes de 2048 tokens. C'est notre baseline.

## 3. La cible : là où LFM3 est faible

Scores de la fiche du modèle :

| Benchmark | LFM3-1.7B | Qwen3.5-4B | Lecture |
|---|---|---|---|
| GPQA Diamond | 63,4 | 75,8 | Connaissances scientifiques en retrait |
| Humanity's Last Exam, sans outils | 7,2 | 12,8 | Idem |
| AA-Omniscience, exactitude | 7,1 | 15,7 | Le plus bas des modèles comparés |
| AA-Omniscience, non-hallucination | 92,9 | 13,6 | Le plus haut : il s'abstient au lieu d'inventer |
| IFEval | 96,7 | 92,5 | Point fort, à ne pas casser |
| BFCL V4 | 71,4 | 55,1 | Point fort, à ne pas casser |

Liquid écrit lui-même d'utiliser de la recherche documentaire pour les tâches de connaissances. Le modèle sait donc peu de choses, mais il sait qu'il ne sait pas. Deux conséquences :

- la marge de progression est là, et c'est le terrain d'un Engram ;
- le risque principal est de lui donner de fausses certitudes et de faire chuter sa non-hallucination. On le mesure à chaque étape.

Ce que l'Engram ne corrigera pas : le long contexte (RULER 128K à 65,6), le raisonnement en plusieurs étapes, le code agentique.

## 4. Budget

| Poste | Heures A100 40 Go | Coût |
|---|---|---|
| Diagnostics D0 à D5 | 6,3 | 13 $ |
| POC, 3 runs principaux | 7,5 | 16 $ |
| Ablations, 4 runs courts | 5 | 10,5 $ |
| Évaluations finales | 3 | 6 $ |
| Comptage sur CPU | 0 | environ 3 $ |
| Réserve | 4 | 8,4 $ |
| **Total** | **25,8** | **57 $** |

Hypothèses à mesurer en D0, car tout le reste en dépend :

- **Vitesse.** J'estime 12 000 à 17 000 tokens par seconde à l'entraînement sur A100, soit 100 à 130 millions de tokens par run de 2 h 30. C'est un calcul, pas une mesure.
- **CPU et mémoire.** Modal les facture en plus du GPU. Le montant est à relever sur le premier job.

Les deux comptes ne partagent pas leurs volumes. Les données tokenisées et les statistiques passent donc par un dépôt privé Hugging Face.

## 5. Diagnostics

Ordre d'exécution : D0, puis D1 et D2 en parallèle, puis D3, puis D4 et D5.

### D0. Mise en route et vitesse (30 min, 1 $)

- **Question.** Combien de tokens par dollar ?
- **Méthode.** Charger LFM3-1.7B sur A100 40 Go. Mesurer les tokens par seconde en inférence, puis en entraînement avec le modèle gelé, avec et sans `torch.compile`, en lignes de 2048 tokens. Refaire 5 minutes sur H100.
- **Sortie.** Le GPU retenu, la taille de batch, et le nombre de tokens par run pour toute la suite.

### D1. Carte des erreurs (1 h de GPU, 2 $, plus le comptage CPU)

- **Question.** Où le modèle perd-il de la probabilité, et quelle part de cette perte une mémoire de n-grammes peut-elle corriger ?
- **Méthode.**
  1. Passer le modèle gelé sur environ 5 millions de tokens de chaque corpus candidat : Wikipédia anglais, Wikipédia français, résumés scientifiques, web général, code, conversations.
  2. Enregistrer pour chaque token sa perte, les 3 tokens précédents, et si la prédiction est juste.
  3. Sur CPU, compter les n-grammes de 2 à 4 tokens sur un grand échantillon de chaque corpus.
  4. Sur un jeu tenu à l'écart, mélanger la prédiction du modèle avec celle des comptages et mesurer la baisse de perte. C'est une borne basse de ce qu'une mémoire de n-grammes peut apporter.
  5. Mesurer la concentration : combien de clés couvrent 50 % et 80 % de ce gain.
- **Sortie.** Le gain potentiel en nats par token pour chaque domaine, et la taille de table nécessaire.
- **Décision.** On retient le domaine au plus fort gain. Si aucun ne dépasse 0,02 nat, on arrête : un Engram ne peut pas aider avec ce budget. Ce seuil vient du papier Engram, où 0,012 nat accompagnait +3 points de MMLU ; c'est un repère, pas une garantie.

### D2. Ressemblance Base / instruct (20 min, 1 $)

- **Question.** Peut-on entraîner l'Engram sur l'un et le brancher sur l'autre ? Et le texte brut ressemble-t-il assez au format chat pour le modèle ?
- **Méthode.**
  1. Sans GPU : écart relatif des poids entre Base et instruct, couche par couche. Montre quelles couches le post-training a modifiées.
  2. Sur les mêmes textes, comparer les états internes des deux modèles à l'entrée de chaque couche (similarité cosinus par token).
  3. Même comparaison, pour le modèle instruct seul, entre un passage en texte brut et le même passage placé dans un tour d'assistant.
- **Décision.** Similarité au-dessus de 0,9 aux couches d'injection : le transfert est jouable et l'entraînement sur texte brut est sûr. En dessous : entraînement sur instruct uniquement, avec une part de données au format chat.

### D3. Où le modèle gelé accepte une écriture (1 h 30, 3 $)

- **Question.** À quelle couche un vecteur ajouté corrige-t-il une prédiction, à quelle amplitude, et avec quels dégâts ailleurs ? Deux papiers se contredisent sur ce point (section 2).
- **Méthode.**
  1. Tirer de D1 300 cas où le modèle se trompe alors que les comptages donnent une suite nette.
  2. Pour 8 couches réparties de 1 à 23, optimiser un vecteur ajouté à la position de la clé, en 20 pas, pour corriger la prédiction.
  3. Mesurer le taux de succès à amplitude limitée (10 % de la norme du flux), puis rejouer ce vecteur sur la même clé dans d'autres contextes pour mesurer les dégâts.
  4. Relever la norme du flux à chaque couche.
- **Sortie.** Une courbe d'efficacité d'écriture selon la profondeur, propre à LFM3.
- **Décision.** Les 2 sites d'injection et l'amplitude de départ. Si aucune couche ne dépasse 50 % de succès, le modèle gelé ne se laisse pas corriger par un vecteur par clé, et il faut passer à Engram + LoRA.

### D4. Mini-run de contrôle (1 h 30, 3 $)

- **Question.** Sur un essai court, l'Engram fait-il jeu égal avec un LoRA ?
- **Méthode.** Environ 30 millions de tokens du domaine retenu. Un Engram classique aux sites donnés par D3, modèle gelé, contre le LoRA rang 64 de Liquid, mêmes données.
- **Mesures.** Perte sur données tenues à l'écart, perte sur données de chat (dérive), statistiques d'ouverture des portes, surcoût en vitesse.
- **Décision.** Si l'Engram obtient moins de la moitié du gain du LoRA, on ne lance pas le POC complet tel quel.

### D5. Scores de référence (1 h 30, 3 $)

- **Question.** Quels sont nos scores de départ, avec notre protocole ?
- **Méthode.** Faire tourner toute la suite d'évaluation de la section 8 sur le modèle intact. Les chiffres de la fiche ne sont pas réutilisables : réglages et sous-ensembles différents.
- **Sortie.** La ligne de base et le bruit de mesure de chaque test.

## 6. Engram+ : le design

Légende : **repris** d'un papier, **adapté** à LFM3, **nouveau** (je ne l'ai trouvé dans aucun des papiers lus, ce qui ne prouve pas que personne ne l'a fait).

**T1. Modèle gelé, mémoire débranchable.** *Repris.* Aucun poids de LFM3 ne change. Porte fermée, la sortie est identique à celle du modèle d'origine.

**T2. Deux sites choisis par mesure.** *Adapté.* D3 choisit les couches. Hypothèse de départ :
- un site à l'entrée de la couche 3, après la première attention globale, pour enrichir la représentation ;
- un site tardif, à l'entrée de la couche 19 ou 21, pour écrire directement la réponse.

Les deux sont des entrées de couches conv : la conv de LFM3 diffuse le signal sur les tokens voisins, donc on retire celle d'Engram.

**T3. Porte fermée par défaut.** *Adapté.* Formule de porte du papier Engram, avec un biais initial négatif pour qu'elle démarre vers 0,05. L'échelle de sortie démarre à zéro, la table non.

**T4. Adressage hybride.** *Adapté de Memory Grafting.*
- Table exacte, sans collision, pour 1 à 2 millions de clés de 2 à 4 tokens, avec correspondance la plus longue.
- Table hachée (2-grammes et 3-grammes, 4 têtes) pour le reste.
- Compression du tokenizer du papier Engram : minuscules et normalisation, soit 23 % de vocabulaire en moins.
- Les clés ne traversent jamais une frontière de document ou de tour de parole.

**T5. Clés choisies par l'erreur.** *Nouveau.* Memory Grafting prend les n-grammes les plus fréquents. Nous prenons ceux où le modèle perd le plus par rapport aux comptages, d'après D1. On ne stocke pas ce que le modèle sait déjà.

**T6. Table écrite, pas apprise.** *Nouveau, c'est le cœur du pari.*
- **Site tardif.** Pour chaque clé, on prend les suites les plus probables d'après les comptages, et on écrit dans la table la combinaison de leurs vecteurs de sortie. C'est possible parce que les embeddings de LFM3 sont liés. Aucun entraînement, et les comptages viennent de milliards de tokens.
- **Site précoce.** Pendant la passe de D1, on calcule pour chaque position la direction qui réduirait l'erreur, et on en fait la moyenne par clé. C'est l'équivalent d'un pas de gradient exact sur tout le corpus.
- Le GPU affine ensuite, avec un taux d'apprentissage faible sur la table.

**T7. Protection du comportement.** *Adapté.*
- Pénalité sur l'ouverture moyenne des portes.
- 10 à 20 % des batchs au format chat, avec une contrainte : la sortie doit rester proche de celle du modèle d'origine.
- Une partie des passages bruts est placée dans un tour d'assistant, pour que les portes apprennent aussi à s'ouvrir en conversation. Aucun jeu de SFT n'est nécessaire.

**T8. Optimisation.** *Repris.* Adam clairsemé sur les lignes de table touchées, sans weight decay, taux d'apprentissage 4 à 5 fois celui des projections (valeurs d'Engram et de STAPLE).

**Taille visée pour le POC.** Environ 0,5 milliard de paramètres de table, soit 2 Go. Plus grand ne servirait à rien : les lignes ne seraient pas vues assez souvent.

**Piste gardée pour plus tard.** Remplir la table depuis les états internes d'un modèle plus gros, comme Memory Grafting. Écartée du POC : il faut apprendre une projection entre les deux modèles, ce qui coûte des tokens.

## 7. Le POC

Tous les runs utilisent les mêmes données, dans le même ordre, avec le même nombre de tokens (fixé par D0).

| Run | Contenu | Rôle |
|---|---|---|
| R0 | Modèle intact | Référence (déjà faite en D5) |
| R1 | LoRA rang 64, recette Liquid | La baseline à battre |
| R2 | Engram classique, modèle gelé, table aléatoire | Mesure ce qu'apportent nos techniques |
| R3 | Engram+ complet (T1 à T8) | Le candidat |
| R4a | R3 sans écriture directe (table aléatoire) | Teste T6 |
| R4b | R3 avec clés choisies par fréquence | Teste T5 |
| R4c | R3 sans table exacte (hachage seul) | Teste T4 |
| R4d | R3 sans protection du comportement | Teste T7 |
| R5 | Engram+ et LoRA ensemble | Seulement si R3 bat R1 |

Répartition : compte 1 pour R1 et les évaluations, compte 2 pour R2 et R3, en parallèle. Les ablations R4 sont des runs deux fois plus courts.

**Lecture des résultats.**
- R3 bat R1 et les garde-fous tiennent : le pari est gagné, on passe à plus de données.
- R3 bat R2 mais pas R1 : nos techniques améliorent l'Engram, mais un LoRA reste plus rentable. On teste R5.
- R3 ne bat pas R2 : l'écriture directe n'apporte rien, on revoit T6.

## 8. Évaluation

**Signal principal, peu bruité.**
- Perte sur données tenues à l'écart : domaine cible, même domaine depuis une autre source, web général, conversations.
- QCM notés par probabilité, sans génération : MMLU complet (14 000 questions, bruit de mesure d'environ 0,4 point), ARC-Challenge.

**Connaissances en génération, livre fermé.**
- TriviaQA (2 000 questions) et PopQA, qui classe les questions par popularité de l'entité : on verra si le gain porte sur les entités moyennement connues.
- Chaque réponse est classée juste, fausse ou abstention. On suit les trois taux, pas seulement l'exactitude.
- Deux modes : réflexion désactivée pour le coût, et réflexion activée sur un échantillon, parce que c'est l'usage réel.

**Garde-fous.**
- IFEval (541 consignes vérifiées par règles).
- Validité du format des appels d'outils sur un échantillon de BFCL.
- GSM8K sur un échantillon, pour le raisonnement.
- Écart de sortie avec le modèle d'origine sur des conversations.

**Contrôles de mécanisme.**
- Portes coupées : on doit retrouver exactement les scores de R0.
- Carte d'ouverture des portes sur quelques textes : s'ouvrent-elles sur les noms propres et les termes techniques ?

**Statistiques.** Décodage glouton, comparaison appariée question par question, intervalle de confiance pour chaque écart. Un écart inférieur au bruit mesuré en D5 ne compte pas.

**Limite.** Le modèle modifié ne tourne ni sur vLLM ni sur llama.cpp. Les évaluations agentiques de la fiche (SWE-bench, LiveClaw) sont hors de portée du POC.

## 9. Organisation du code

```
lfm3-engram/
  PLAN.md
  engram/      module Engram+, hachage sur GPU, branchement par hook
  diag/        d0_speed.py … d5_baseline.py
  data/        tokenisation, comptage de n-grammes sur CPU
  train/       boucle d'entraînement, configs R1 à R5
  eval/        suite d'évaluation
  modal_app.py images, volumes, lancement sur compte1 ou compte2
  results/     un fichier de résultats par diagnostic et par run
```

Points d'implémentation déjà identifiés :

- **Branchement.** Un hook à l'entrée de la couche cible, sans modifier le code de Liquid.
- **Génération.** Avec le cache, le modèle ne reçoit qu'un token à la fois. Il faut garder les 3 derniers tokens en mémoire pour reconstruire la clé.
- **Hachage.** Celui de la démo DeepSeek est en numpy sur CPU. À réécrire en torch sur GPU.
- **Versions.** Le notebook de Liquid épingle `transformers==5.20.2`, `torch==2.14.1`, `peft==0.21.2`. On reprend les mêmes.

## 10. Risques et critères d'arrêt

| Risque | Comment on le voit | Réaction |
|---|---|---|
| Rien à gagner par n-grammes | D1 sous 0,02 nat partout | Arrêt après environ 4 $ de dépense (D0, D1, D2) |
| Le modèle gelé n'accepte pas d'écriture | D3 sous 50 % de succès | Passage direct à Engram + LoRA |
| Un LoRA fait aussi bien | R1 au niveau de R3 | L'Engram n'est pas justifié à cette échelle |
| Le gain en perte ne se voit pas dans les réponses | Perte en baisse, TriviaQA et PopQA plats | Davantage de données au format chat (T7) |
| Fausses certitudes | Taux de réponses fausses en hausse | Pénalité de porte plus forte, sinon échec |
| Budget dépassé | Vitesse de D0 sous l'estimation | Runs plus courts, ablations réduites à R4a et R4b |

**Attentes réalistes.** Aucun papier ne montre un gain avec si peu de tokens. Le résultat le plus probable est un gain net sur le domaine ciblé et faible ou nul ailleurs. Un gain général de plusieurs points sur tous les benchmarks n'est pas atteignable avec 60 $.

## 11. Décisions à prendre

1. **Domaine cible.** Par défaut D1 choisit entre connaissances générales en anglais, français, sciences et code. Si un usage précis est visé, autant le dire maintenant : on l'ajoute aux candidats.
2. **Plafond de budget.** 60 $ ferme, ou possibilité d'ajouter des crédits si D1 à D4 sont bons ?
3. **Déploiement.** Si le but final est de faire tourner le modèle en local (GGUF, MLX), il faudra écrire le support de l'Engram dans le moteur. Ce n'est pas dans ce plan.

## Sources

- [Liquid AI, annonce LFM3](https://www.liquid.ai/blog/lfm3)
- [Fiche LFM3-1.7B](https://huggingface.co/LiquidAI/LFM3-1.7B) et [fiche LFM3-1.7B-Base](https://huggingface.co/LiquidAI/LFM3-1.7B-Base)
- [Notebook Liquid d'entraînement sur texte brut](https://github.com/Liquid4All/cookbook/blob/main/finetuning/notebooks/lfm3_cpt_with_trl.ipynb)
- [Engram, papier](https://arxiv.org/html/2601.07372v1) et [code de démo](https://github.com/deepseek-ai/Engram)
- [STAPLE](https://arxiv.org/abs/2610.12599) et [STEM](https://arxiv.org/html/2601.10639v1)
- [Memory Grafting](https://arxiv.org/html/2605.20948v1)
- [User as Engram](https://arxiv.org/html/2606.19172)
- [RFC NeMo Automodel n° 4292](https://github.com/NVIDIA-NeMo/Automodel/issues/4292)
- [Infini-gram](https://arxiv.org/html/2401.17377v4)
- [Tarifs Modal](https://modal.com/pricing)
