# LFM3-1.7B + mémoire : résultats des diagnostics et nouvelle architecture

17 novembre 2026. Ce document remplace les sections 5 à 7 de [PLAN.md](PLAN.md) : les diagnostics ont écarté l'Engram classique et fait émerger une autre architecture. Les chiffres bruts sont dans `results/*.json`.

## 1. Conclusion

- **L'Engram classique est écarté pour ce modèle gelé.** Un vecteur ajouté au flux interne, même optimisé exprès, corrige 5 % des réponses fausses (contrôle : 3 %).
- **Une mémoire lue par l'attention fonctionne sans aucun entraînement.** Le début de l'article de l'entité, rendu visible aux 6 couches d'attention globale, corrige 73,5 % des réponses fausses sur PopQA, contre 74,5 % quand on colle le texte dans le prompt.
- **Meilleur système à ce jour (D14, sans LoRA, tête de choix entraînée) : PopQA 14,4 % → 59,5 %, TriviaQA 31,5 % → 43,1 %, MKQA anglais 23,3 % → 28,9 %, HotpotQA 14,0 % → 24,0 %.** Le LoRA de lecture testé en D14 est rejeté.
- **Premier résultat de bout en bout (D5, ancienne source), PopQA 15,2 % → 54,7 %** quand le système trouve seul l'entité (plafond avec la bonne entité fournie : 72,5 %).
- **Ce qui reste ouvert : les homonymes et la robustesse.** Le système fait 83 % quand le nom est unique et 37 % quand il a 5 candidats ou plus. Un passage hors sujet casse encore environ 20 % des réponses justes dans le test dur.

## 2. Résultats

### D0. Vitesse (LFM3-1.7B, lignes de 2048 tokens)

| Carte | Batch | Entraînement, modèle gelé | Inférence | Coût par million de tokens d'entraînement |
|---|---|---|---|---|
| A100 40 Go (2,10 $/h) | 4 | 15 300 tok/s | | 0,038 $ |
| A100 80 Go (2,50 $/h) | 8 | 15 600 tok/s | 44 000 tok/s | 0,044 $ |
| H100 (3,95 $/h) | 8 | 33 900 tok/s | 96 000 tok/s | 0,032 $ |

Le H100 est 2,2 fois plus rapide et le moins cher par token. Mémoire : 30 Go à batch 4, 47 Go à batch 8.

### D1. Marge d'une mémoire de n-grammes remplie par comptage

Gain en nats par token, clés de 2 à 4 tokens, comptages sur 200 millions de tokens (100 pour le code), mesuré sur 1,5 million de tokens tenus à l'écart.

| Corpus | Perte Base | Perte instruct | Gain sur Base | Gain sur instruct |
|---|---|---|---|---|
| Web général | 2,51 | 2,77 | 0,012 | 0,029 |
| Wikipédia anglais | 2,29 | 2,60 | 0,036 | 0,074 |
| Wikipédia français | 2,07 | 2,59 | 0,061 | 0,212 |
| arXiv | 2,54 | 2,69 | 0,089 | 0,100 |
| Code | 1,88 | 1,95 | 0,195 | 0,212 |

- Le post-training a coûté 0,06 à 0,52 nat en prédiction de texte brut. L'écart de gain entre les deux modèles vient de cette dérive, pas de connaissances manquantes.
- Sur Wikipédia anglais (Base), les mots à majuscule pèsent 22 % de la perte et 14 % du gain : le comptage apporte des tournures, peu de faits.
- Artefacts : 55 % du gain arXiv porte sur le token de fin de document, 37 % du gain code sur la ponctuation.
- Taille de table (Wikipédia anglais, Base) : 0,7 million de clés captent 40 % du gain, 14 millions en captent 76 %.

### D2. Base contre instruct

| Mesure | Résultat |
|---|---|
| Écart relatif des poids | 35 % (29 à 39 % selon la couche) |
| Similarité des états, entrée de la couche 3 | 0,92 à 0,93 |
| Similarité, couches 11 à 21 | 0,87 à 0,91 |
| Même passage en texte brut ou dans un tour d'assistant | 0,976 au minimum |
| Norme du flux : couches 3, 11, 19, sortie | 0,9 / 2,1 / 5,9 / 12,9 |

Base et instruct ne sont pas interchangeables. Le format (brut ou chat) change très peu la représentation d'un passage.

### D3. Un vecteur par entité (l'Engram en miniature)

PopQA, format chat, mode sans réflexion forcé. 240 questions ratées, 80 réussies. Un vecteur par entité, optimisé 80 pas sur les 256 premiers tokens de son article, injecté au dernier token de chaque mention du nom.

| Couche | Corrigées, bon vecteur | Corrigées, vecteur d'une autre entité | Réponses justes cassées |
|---|---|---|---|
| 3 | 5,0 % | 2,9 % | 22,5 % |
| 9 | 4,6 % | 2,1 % | 17,5 % |
| 15 | 2,9 % | 2,1 % | 7,5 % |
| 21 | 1,3 % | 1,3 % | 7,5 % |

Référence : 87 % de corrigées avec l'extrait de l'article dans le prompt. Le vecteur mémorise le texte (perte 2,60 → 1,16 à la couche 3) et dégrade la suite non vue (2,66 → 3,11).

Scores de départ sur 5 000 questions PopQA : 13,5 % de bonnes réponses, 82,0 % de fausses, 4,5 % d'abstentions. Avec réflexion autorisée (400 questions) : 12,5 % de bonnes réponses, et 47 % de réponses inachevées après 384 tokens.

### D4. Mémoire lue par l'attention, sans entraînement

400 questions ratées, 150 réussies. La condition « mémoire » reproduit exactement le mécanisme proposé : passage, puis 256 tokens masqués, puis la question. Le passage est alors invisible pour les convolutions et les attentions à fenêtre, visible pour les 6 attentions globales.

Questions ratées corrigées :

| Passage | Réponse présente dans le passage | Texte dans le prompt | Passage juste avant le prompt | **Mémoire** | Mauvais passage |
|---|---|---|---|---|---|
| 32 tokens | 60 % | 59,5 % | 58,3 % | **56,0 %** | 3,8 % |
| 64 tokens | 83 % | 74,5 % | 76,8 % | **73,5 %** | 4,3 % |
| 128 tokens | 90 % | 78,0 % | 80,3 % | **77,0 %** | 4,0 % |

Autres mesures (passage de 64 tokens) :

| Mesure | Résultat |
|---|---|
| Corrigées quand la réponse est dans le passage | 86,2 % |
| Corrigées avec 4 passages dont 3 hors sujet | 55,3 % |
| Réponses justes conservées, bon passage | 95,3 % |
| Réponses justes conservées, mauvais passage | 71,3 % |
| Réflexion autorisée, sans mémoire (120 questions) | 0,8 % de bonnes réponses, 50 % inachevées après 768 tokens |
| Réflexion autorisée, avec mémoire | 59,2 % de bonnes réponses, 17,5 % inachevées |

Projection sur PopQA avec le bon passage de 64 tokens : environ 76 % de bonnes réponses contre 13,5 % (0,135 × 0,953 + 0,865 × 0,735). Elle suppose qu'on connaît la bonne entité, ce que le test faisait d'office.

### D5. Passage hors sujet et choix automatique de l'entité, sans entraînement

**Partie A.** Mêmes 400 questions ratées et 150 réussies que D4, passage de 64 tokens.

| Réglage | Ratées corrigées, bon passage | Réussies conservées, bon passage | Réussies conservées, mauvais passage |
|---|---|---|---|
| Passage brut (D4) | 73,5 % | 95,3 % | 71,3 % |
| En-tête « note de référence » | 74,8 % | 96,0 % | 74,7 % |
| Consigne système | 78,3 % | 96,0 % | 78,0 % |
| En-tête et consigne système | 77,0 % | 96,7 % | 79,3 % |
| Malus d'attention 0,25 | 70,5 % | 96,0 % | 76,0 % |
| Malus 0,5 | 64,8 % | 94,0 % | 74,7 % |
| Malus 1,0 | 46,5 % | 94,7 % | 80,7 % |
| Malus 2,0 | 7,0 % | 85,3 % | 86,0 % |

- La consigne système seule, sans aucune mémoire, ne conserve que 84 % des réponses réussies : elle change le comportement du modèle.
- Avec un malus énorme (mémoire invisible pour l'attention), 88 % des réussies sont conservées, pas 100 %. Une partie des dégâts vient donc de la mise en page du test (tokens masqués, décalage des positions), pas du contenu du passage.
- Le malus d'attention est un mauvais compromis : il retire des corrections plus vite qu'il ne protège.

**Partie B.** 1 000 questions PopQA tirées au hasard, le système trouve seul l'entité. Passage de 128 tokens.

| Configuration | Bonnes réponses | Réponses auparavant justes conservées |
|---|---|---|
| Sans mémoire | 15,2 % | |
| Bonne entité fournie (plafond) | 72,5 % | |
| 1 candidat, article le plus long | 41,8 % | 86,2 % |
| 1 candidat, choisi par le modèle | 46,3 % | 86,8 % |
| 1 candidat, modèle et longueur d'article | 47,3 % | 90,1 % |
| 3 candidats | 49,7 % | 90,1 % |
| Tous les candidats (6 au plus) | 47,8 % | 86,8 % |
| **3 candidats, en-tête et consigne système** | **54,7 %** | **91,4 %** |

Selon l'ambiguïté du nom (meilleure configuration) :

| Nombre de candidats pour le nom | Questions | Sans mémoire | Automatique | Bonne entité fournie |
|---|---|---|---|---|
| 1 | 310 | 25,5 % | 83,2 % | 77,7 % |
| 2 à 4 | 169 | 12,4 % | 56,8 % | 63,9 % |
| 5 ou plus | 507 | 9,1 % | 36,9 % | 72,4 % |

- Le nom est repéré dans 98,6 % des questions. La bonne entité figure parmi les candidats dans 74,3 % des cas.
- Sur les noms ambigus, le choix par le modèle trouve la bonne entité dans 41 % des cas, 50 % en ajoutant la longueur de l'article, contre 39 % pour l'article le plus long seul. Les questions PopQA sont souvent trop courtes pour trancher (« Dans quel pays est Norfolk ? », réponse attendue : Norfolk dans le Connecticut).

### Défaut de la source Wikipédia (découvert pendant D6)

Le jeu `wikimedia/wikipedia` de Hugging Face (20231101.en), utilisé pour D3, D4 et D5, ne contient pas certains articles principaux : « Shirley Temple », « Microwave », « Skin », « Manga », « Norfolk », « Cosmic microwave background », « Maxwell's equations ». Faute de l'article principal, un nom renvoyait vers un homonyme. L'API de Wikipédia n'est pas utilisable depuis Modal (erreur 429). À partir de D6, la source est l'export structuré de Wikimedia (`wikimedia/structured-wikipedia`) : 6,5 millions d'articles en anglais (3,9 Go), 2,3 millions en français (1,1 Go), avec résumé, description courte et identifiant commun aux langues.

Conséquence : les conclusions de D3 à D5 sur le mécanisme tiennent, mais les chiffres de D5 sur les homonymes sont pessimistes.

### D6. Raisonnement, réflexion activée

Dictionnaire complet, déclenchement par règles strictes (titre exact uniquement), pas de consigne système, tirage à température 0,6 comme dans les évaluations de Liquid. Quand le budget de réflexion est épuisé, le bloc est fermé et le modèle doit répondre.

**HotpotQA** (250 questions, budget de réflexion 768 tokens)

| Condition | Bonnes réponses | Deux sauts | Comparaison | Corrigées | Cassées | Notes attendues montrées |
|---|---|---|---|---|---|---|
| Sans mémoire | 14,8 % | 10,0 % | 34,7 % | | | |
| Notes déclenchées par la question | 26,4 % | 15,9 % | 69,4 % | 38 | 9 | 35 % |
| Idem, plus déclenchement pendant la réflexion | 26,4 % | 17,9 % | 61,2 % | 39 | 10 | 44 % |
| Bonnes notes fournies (plafond) | 42,0 % | 35,8 % | 67,3 % | 78 | 10 | 89 % |

- La mémoire automatique fait passer de 14,8 % à 26,4 %.
- Le déclenchement pendant la réflexion ajoute 4 notes par question et montre plus souvent une note attendue (44 % contre 35 %), mais n'améliore pas le score global : les notes en plus sont souvent hors sujet.
- La réflexion atteint le budget dans 76 à 88 % des cas.

**GPQA Diamond** (100 questions, budget 3 072 tokens)

| Condition | Bonnes réponses | Corrigées | Cassées |
|---|---|---|---|
| Sans mémoire | 48 % | | |
| Mémoire, déclenchement pendant la réflexion | 51 % | 17 | 14 |
| Idem, notes peu nouvelles écartées | 54 % | 17 | 11 |

Les écarts ne sont pas significatifs sur 100 questions. La mémoire ne dégrade pas GPQA ; elle ne l'améliore pas de façon démontrée. Le score sans mémoire (48 %) est inférieur aux 63,4 % de la fiche de Liquid, mesurés sans limite de longueur.

**GSM8K** (150 questions) : 87,3 % sans mémoire, 87,3 % avec. Aucun effet.

### D7. Français (MKQA, 1 000 questions, sans réflexion, sans entraînement)

| | Sans mémoire | Avec mémoire | Corrigées | Cassées | Note déclenchée | Réponse présente dans une note |
|---|---|---|---|---|---|---|
| Français, notes françaises | 20,1 % | 22,5 % | 44 | 20 | 31 % | 7,9 % |
| Anglais, mêmes questions | 24,6 % | 28,9 % | 59 | 16 | 38 % | 11,0 % |

Quand une note se déclenche : 19,5 % → 27,2 % en français, 26,1 % → 38,0 % en anglais. Le gain global est faible pour deux raisons : les questions sont écrites sans nom propre net, donc rien ne se déclenche dans les deux tiers des cas ; et le résumé d'un article contient rarement la réponse à ce type de question.

### D8. Briques du module d'activation (TriviaQA, modèle gelé)

**Quand.** Une sonde linéaire sur l'état interne à la fin de la question prédit que la réponse sans mémoire sera fausse. Fiabilité (AUC) de 0,82 à la couche 16, sur 2 000 questions non vues. Dans la moitié des questions qu'elle signale, 84,6 % des réponses sont fausses, contre 41,6 % dans l'autre moitié.

**Quoi.** Retrouver l'article de la réponse parmi 200 000 notes, à partir d'une question qui ne la nomme pas (40 000 paires d'entraînement, 3 000 de test, deux projections linéaires).

| | En tête | Dans les 5 | Dans les 20 |
|---|---|---|---|
| Toutes les questions de test | 49,4 % | 66,1 % | 74,4 % |
| Réponses jamais vues à l'entraînement (15 % du test) | 16,9 % | 32,0 % | 46,2 % |
| Sans entraînement, réponses jamais vues | 16,6 % | 30,9 % | 41,9 % |

La tête entraînée mémorise les réponses fréquentes et ne généralise pas aux entités nouvelles. Non mesuré : la part de ces succès qui porte sur des questions où le modèle se trompe.

### D9. La sonde « quand » branchée sur la mémoire (sans réflexion)

Sonde linéaire à la couche 16, entraînée sur 5 000 questions TriviaQA en anglais et 2 000 questions MKQA en français hors échantillon de test. Mémoire par titre exact, dictionnaire complet. 1 000 questions par jeu.

Fiabilité de la sonde (AUC) : 0,80 sur TriviaQA, 0,85 sur PopQA, 0,82 sur MKQA anglais, 0,77 sur MKQA français.

| Jeu | Sans mémoire | Mémoire toujours | Corrigées / cassées | Mémoire si la sonde s'inquiète (seuil 0,3) | Corrigées / cassées |
|---|---|---|---|---|---|
| TriviaQA | 31,3 % | 42,2 % | 140 / 31 | 41,7 % | 120 / 16 |
| PopQA | 14,7 % | 30,3 % | 165 / 9 | 29,1 % | 151 / 7 |
| MKQA français | 19,4 % | 21,7 % | 39 / 16 | 22,0 % | 38 / 12 |
| MKQA anglais | 24,2 % | 28,5 % | 58 / 15 | 27,7 % | 49 / 14 |

Utilisée comme porte, la sonde divise à peu près par deux les réponses cassées mais fait perdre autant de corrections : le score ne monte pas. Avec les règles strictes, la mémoire casse déjà peu de réponses.

Utilisée pour s'abstenir (remplacer la réponse par « je ne sais pas » quand la sonde s'inquiète et que la mémoire n'a rien à montrer, seuil 0,5), calculé sans régénérer :

| Jeu | Abstentions | Dont réponses qui auraient été fausses | Justes | Fausses | Fausses sans la sonde (mémoire toujours) |
|---|---|---|---|---|---|
| TriviaQA | 20,4 % | 83,3 % | 35,5 % | 44,1 % | 57,8 % |
| PopQA | 49,8 % | 96,2 % | 25,8 % | 24,4 % | 69,7 % |
| MKQA français | 57,7 % | 85,8 % | 14,1 % | 28,2 % | 78,3 % |
| MKQA anglais | 48,1 % | 86,9 % | 21,1 % | 30,8 % | 71,5 % |

PopQA avec titre exact uniquement atteint 30,3 %, contre 54,7 % en D5 où plusieurs homonymes étaient montrés : une note ne se déclenche que pour 41 % des questions. Le choix entre homonymes reste le principal manque sur ce jeu.

### D10. Que faudrait-il mettre dans la note ? (CPU, sans génération)

Part des questions dont une réponse acceptée figure dans le texte, selon ce que contient la note. « Système actuel » : les articles que le déclenchement par nom trouve. « Bons articles » : ceux que le jeu désigne (PopQA, HotpotQA).

| Jeu | Articles | Résumé, 96 tokens (note actuelle) | Résumé, 256 tokens | Résumé 96 + fiche de faits | Corps, 1 000 tokens | Article entier |
|---|---|---|---|---|---|---|
| PopQA | Système actuel | 24,9 % | 26,2 % | 29,3 % | 29,0 % | 32,3 % |
| PopQA | Bons articles | 80,8 % | 84,9 % | 90,1 % | 89,4 % | 93,7 % |
| HotpotQA | Système actuel | 26,7 % | 28,0 % | 27,5 % | 33,9 % | 39,8 % |
| HotpotQA | Bons articles | 63,1 % | 68,2 % | 69,1 % | 75,4 % | 75,9 % |
| TriviaQA | Système actuel | 20,9 % | 26,3 % | 25,3 % | 34,2 % | 44,0 % |
| MKQA anglais | Système actuel | 9,8 % | 13,2 % | 14,2 % | 18,6 % | 25,9 % |
| MKQA français | Système actuel | 6,5 % | 9,7 % | 11,4 % | 14,6 % | 19,1 % |

- Le système actuel montre un article pour 41 % des questions PopQA, 84 % de HotpotQA, 59 % de TriviaQA, 37 % de MKQA anglais, 30 % de MKQA français.
- 86 % des articles ont une fiche de faits, de 132 tokens en médiane.
- Trouver le bon article est le premier levier : sur PopQA, 29 % avec le système actuel contre 90 % avec les bons articles.
- Ajouter la fiche de faits au résumé vaut autant que 1 000 tokens de corps d'article, pour environ 130 tokens.
- L'article entier double la couverture sur TriviaQA et MKQA, à condition de savoir y choisir le bon passage.

### D12. Audit de notre propre chaîne (PopQA, 600 questions, bon article fourni, sans réflexion)

**Pénalité de répétition.** Le modèle est livré avec une pénalité de répétition de 1,1 dans ses réglages par défaut. Tous les runs précédents l'ont appliquée sans le vouloir. Elle abaisse la probabilité des tokens déjà présents dans le contexte, donc de ceux de la note.

| Réglage | Sans mémoire | Avec la bonne note |
|---|---|---|
| Pénalité partout (tous les runs jusqu'ici) | 13,0 % | 76,7 % |
| Aucune pénalité | 12,3 % | 78,8 % |
| Pénalité partout sauf sur la note | 13,0 % | 81,0 % |

Exclure la note de la pénalité rapporte 4,3 points avec la bonne note, sans rien changer au reste.

**Traitement par lots et mise en page** (200 questions). Le score est le même pour une question traitée seule (13,0 %), dans un lot (12,5 %) et avec la mise en page à vide (13,0 %). En revanche le texte généré n'est identique que dans 64,5 % des cas entre « seule » et « en lot ». Les réponses changent donc de formulation pour des raisons numériques sans que la justesse bouge. Conséquence : les mesures de D5 du type « réponses justes conservées » sont biaisées vers le bas, puisqu'une part des changements est du bruit.

**Longueur des réponses.** Sans mémoire, 58,7 % des réponses atteignent la limite de 40 tokens, contre 11,5 % avec la note. Mais le score ne change presque pas à 96 tokens : 13,0 % → 13,5 % sans mémoire, 76,7 % → 77,7 % avec. La coupure ne fausse pas la comparaison.

**Défaut trouvé à la relecture du code.** Le dictionnaire reconstruit pour D6 ne proposait plus comme homonymes les titres à virgule (« Norfolk, Connecticut » pour « Norfolk »). Corrigé dans le code ; D6, D7, D9 et D11 ont tourné avec ce défaut.

### D11. Note enrichie et tête de choix d'article (première composante entraînée)

À partir d'ici, tout n'est plus « sans entraînement ». LFM3 reste gelé, mais une petite tête est entraînée : deux projections linéaires sur l'état du modèle (question d'un côté, note candidate de l'autre), la taille de l'article comme indice, et une option « aucun ». Entraînement sur 12 284 questions PopQA ; les 983 questions partageant une entité ou un nom avec l'échantillon de test ont été retirées. Dictionnaire complet avec fiches de faits, mais sans les homonymes à virgule (défaut corrigé depuis) et avec la pénalité de répétition par défaut.

**Choix du bon article** (1 037 noms repérés dans les 1 000 questions de test ; le bon article figure parmi les candidats dans 77 % des cas)

| Méthode | Tous les noms | Noms ambigus (7,5 candidats en moyenne) |
|---|---|---|
| Article le plus développé | 58,6 % | 34,3 % |
| État interne brut, sans entraînement | 61,9 % | 39,6 % |
| Tête entraînée | 76,4 % | 62,6 % |
| Tête entraînée sans l'indice de taille | 75,7 % | 61,4 % |
| Tête entraînée, bon article dans les 3 premiers | | 88,7 % |

La tête répond « aucun » dans 44 % des cas où le bon article est absent des candidats, et dans 6 % des cas où il est présent.

**PopQA de bout en bout** (1 000 questions, sans réflexion)

| Configuration | Bonnes réponses | Corrigées / cassées | Bon article montré |
|---|---|---|---|
| Sans mémoire | 14,7 % | | |
| Titre exact, résumé seul (D9) | 30,2 % | 164 / 9 | 29,5 % |
| Titre exact, résumé et fiche de faits | 33,2 % | 194 / 9 | 29,5 % |
| Article le plus développé, 1 note | 44,1 % | 315 / 21 | 47,1 % |
| Articles les plus développés, 3 notes | 52,6 % | 398 / 19 | 63,9 % |
| Tête entraînée, 1 note | 56,8 % | 438 / 17 | 59,7 % |
| Tête entraînée, 2 notes | 57,9 % | 446 / 14 | 67,4 % |
| Tête entraînée, 3 notes | 57,9 % | 444 / 12 | 70,3 % |
| Bon article fourni, résumé seul | 72,1 % | 581 / 7 | 92,8 % |
| Bon article fourni, résumé et fiche de faits | 79,1 % | 651 / 7 | 92,8 % |

**Transfert vers d'autres jeux** (1 000 questions chacun)

| Configuration | TriviaQA | Corrigées / cassées | MKQA anglais | Corrigées / cassées |
|---|---|---|---|---|
| Sans mémoire | 31,3 % | | 24,2 % | |
| Titre exact, résumé seul | 42,3 % | 143 / 33 | 28,4 % | 58 / 16 |
| Titre exact, résumé et fiche de faits | 43,8 % | 159 / 34 | 29,2 % | 67 / 17 |
| Tête entraînée, 1 note | 42,4 % | 168 / 57 | 29,6 % | 73 / 19 |
| Tête entraînée, 3 notes | 42,7 % | 166 / 52 | 30,0 % | 75 / 17 |

- La fiche de faits apporte 3 points avec le titre exact et 7 points quand le bon article est fourni.
- Sur PopQA, la tête fait passer de 33,2 % à 56,8 % avec une seule note. La version sans entraînement la plus proche (3 articles les plus développés) est à 52,6 %.
- La tête ne se transpose pas : sur TriviaQA elle corrige un peu plus de réponses mais en casse presque deux fois plus, et le score ne monte pas. Elle n'a vu que les 16 formes de question de PopQA.

### D13. Fiches de méthode en maths et en logique (plafond : le type de problème est donné)

Réflexion activée, tirage à température 0,6, note montrée par la mémoire. « Fiche » : une méthode rédigée d'environ 90 tokens par type de problème. « Exemples » : deux problèmes résolus du même type, environ 500 tokens.

| Jeu | Questions | Sans note | Fiche de méthode | Corrigées / cassées | Exemples résolus | Corrigées / cassées |
|---|---|---|---|---|---|---|
| BIG-Bench Hard, 14 types de logique | 420 | 73,1 % | 75,5 % | 52 / 42 | 70,7 % | 47 / 57 |
| MATH, niveaux 3 à 5, 7 domaines | 210 | 64,8 % | 66,7 % | 18 / 14 | 63,3 % | 12 / 15 |
| GSM8K | 120 | 80,0 % | 80,8 % | 10 / 9 | 77,5 % | 7 / 10 |

Aucun effet démontré. La fiche est à +1 à +2 points sur les trois jeux, dans le bruit : sur BIG-Bench Hard, deux passages sans note diffèrent déjà de 43 et 47 réponses dans chaque sens. Les exemples résolus font un peu moins bien que rien. La mémoire aide quand il manque un fait, pas quand il manque une méthode.

**Pénalité de répétition et maths.** Sans note, avec la pénalité par défaut du modèle (1,1) : 58,1 % sur MATH contre 64,8 % sans pénalité (9 corrigées, 23 cassées). Sur BIG-Bench Hard et GSM8K, pas d'écart. La pénalité coûte donc environ 7 points sur les maths de niveau concours.

### D14. LoRA de lecture : rejeté selon le critère fixé d'avance

Détail du protocole et du critère dans [D14.md](D14.md). Corrections intégrées : pénalité de répétition hors des notes, homonymes à virgule, bonne entité par identifiant Wikidata, réponses validées par mots entiers, positions identiques à l'entraînement et à la génération (écart des sorties 0,0 avec la correction, 4,5 sans). Tête de choix entraînée sur PopQA et HotpotQA (17 500 exemples), LoRA sur 6 570 exemples, 3 passes, 18 millions de tokens, perte 0,87 → 0,28.

**Le système sans LoRA, avec toutes les corrections** (meilleur résultat à ce jour, sans réflexion)

| Jeu | Sans mémoire | Mémoire | Corrigées / cassées | Bonnes notes fournies |
|---|---|---|---|---|
| PopQA (1 000) | 14,4 % | 59,5 % | 470 / 19 | 83,0 % |
| TriviaQA (1 000) | 31,5 % | 43,1 % | 170 / 54 | |
| MKQA anglais (1 000) | 23,3 % | 28,9 % | 73 / 17 | |
| HotpotQA (250) | 14,0 % | 24,0 % | 32 / 7 | 45,2 % |

**Avec le LoRA**

| Jeu | Justes : mémoire → LoRA | Gagnées / perdues | p (McNemar) | Abstentions | Fausses : mémoire → LoRA |
|---|---|---|---|---|---|
| PopQA | 59,5 % → 54,1 % | 35 / 89 | < 0,001 | 0,1 % → 19,0 % | 40,4 % → 26,9 % |
| TriviaQA | 43,1 % → 32,0 % | 16 / 127 | < 0,001 | 0 % → 50,6 % | 56,9 % → 17,4 % |
| MKQA anglais | 28,9 % → 21,0 % | 2 / 81 | < 0,001 | 1,6 % → 29,2 % | 69,5 % → 49,8 % |
| HotpotQA | 24,0 % → 19,2 % | 7 / 19 | 0,029 | 1,2 % → 54,8 % | 74,8 % → 26,0 % |

Contrôles sur PopQA : avec les bonnes notes, 83,0 % → 77,5 % (21 gagnées, 76 perdues, 10,3 % d'abstentions). Avec des homonymes seuls, 16,1 % → 15,0 % de justes, abstentions 0,5 % → 42,9 %, fausses 83,4 % → 42,1 %.

Style : sur 5 demandes de texte long avec une note, 2 réponses s'effondrent avec le LoRA (166 → 36 mots, 151 → 27 mots).

**Verdict.** Le critère 1 échoue sur les quatre jeux : les bonnes réponses baissent de façon significative. Le critère 3 échoue aussi : la hausse des abstentions ne vient pas seulement des réponses fausses. Le LoRA a surtout appris à s'abstenir. Il divise les réponses fausses par 1,5 à 3, mais en sacrifiant des réponses justes, y compris quand la bonne note est présente. Il n'est pas retenu.

**Cause probable.** La cible « I don't know. » dès que la réponse ne figure dans aucune note. Le modèle ne sait pas distinguer « je connais la réponse sans la note » de « je ne la connais pas », donc il généralise en « la note ne répond pas, je m'abstiens ».

### I1. Intégration : le magasin de notes et le modèle en un seul objet (17 novembre 2026)

Aucune mesure de qualité ici : c'est la mise en forme du système de D14 (« mémoire », sans LoRA) pour qu'il se charge avec `from_pretrained` et s'utilise avec `generate`, sans code autour.

**Magasin de notes** ([lfm3e/store.py](lfm3e/store.py), construction dans [lfm3e/store_build.py](lfm3e/store_build.py)) : un seul fichier safetensors, lu par projection en mémoire.

| | |
|---|---|
| Articles | 8 269 417 (6 056 292 en anglais, 2 213 125 en français) |
| Noms indexés | 8 622 392 |
| Taille du fichier | 2,31 Go (notes 2,08 Go, index 0,23 Go) |
| Par article | 279 octets, pour 203 tokens en moyenne (titre 13, résumé 85, faits 106) |
| Par token | 1,23 octet |
| Lecture d'une note | 0,18 ms |
| Vérification | 600 articles tirés au hasard relus token pour token par l'index des noms : 0 écart |

Ce qui rend le fichier compact : les notes sont stockées en tokens ; chaque token est écrit par son rang de fréquence en longueur variable, puis la note est compressée avec zlib amorcé par un dictionnaire de notes typiques, un par langue ; les noms sont des empreintes de 64 bits triées. L'ancien format (sqlite, texte) pesait 6,56 Go pour l'anglais seul.

L'export Wikimedia contient plusieurs versions de certains articles : 519 845 doublons ont été écartés (on garde la version la plus développée). Les 6,5 millions d'articles anglais annoncés jusqu'ici en comptaient donc environ 0,45 million en double.

Le fichier stocke jusqu'à 192 tokens de résumé et 256 de faits ; le modèle n'en lit aujourd'hui que 96 et 128. Seuls 3 % des résumés et 8 % des blocs de faits atteignent le plafond stocké : allonger encore ne rapporterait presque rien. Il reste environ 1 Go sous la limite fixée de 3 à 3,5 Go.

**Modèle intégré** ([lfm3e/modeling_lfm3_mem.py](lfm3e/modeling_lfm3_mem.py), [lfm3e/configuration_lfm3_mem.py](lfm3e/configuration_lfm3_mem.py), assemblage du dossier par [lfm3e/export_hf.py](lfm3e/export_hf.py)). `AutoModelForCausalLM.from_pretrained(dossier, trust_remote_code=True)` rend le modèle LFM3 d'origine, chargé depuis le dépôt de Liquid, avec la mémoire attachée. `generate` lit le dernier message de l'utilisateur, choisit les notes, les place devant le prompt, génère, puis rend les séquences comme si les notes n'avaient jamais été là.

Test local sur CPU, avec les vrais poids ([tests/test_integrated.py](tests/test_integrated.py)) :

| Vérification | Résultat |
|---|---|
| Prompt sans note | Sortie identique au modèle d'origine, token pour token |
| Prompt avec note | Sortie identique à la mise en page construite à la main et passée au `generate` d'origine |
| Lot mêlant prompts avec et sans note, réglages par défaut | Notes aux bonnes lignes, prompts rendus intacts |
| `return_dict_in_generate`, plusieurs séquences par prompt, flux (`streamer`) | Fonctionnent |
| Réflexion activée, prompt en français | Fonctionnent |
| Recherche des notes | 3 à 13 ms par prompt sur le Mac |

Ce que ce test ne dit pas :
- **La tête de choix n'est pas encore branchée.** Elle doit être réentraînée dans le format final des notes. En attendant, l'article le plus développé est pris : « What is the capital of Norfolk? » reçoit la note de Norfolk en Virginie et la réponse devient fausse, alors que le modèle seul répondait Norwich.
- **Les termes techniques ne déclenchent pas encore** (« pyridinium chlorochromate ») : la table de fréquence des mots voyage avec le fichier de la tête.
- **Aucun chiffre de qualité** n'a été refait à travers le modèle intégré. Les exemples essayés à la main (Kiss and Tell, Victor Hugo, Tosca : faux sans note, justes avec) ne sont pas une mesure.

Un défaut trouvé et corrigé pendant ce test : la règle des noms de D5, écrite pour des questions d'une phrase, prenait les nombres pour des noms. « what is 17 times 3 » recevait deux notes sans rapport et la réponse se dégradait. Pour un prompt quelconque, la règle ignore maintenant les nombres seuls, la majuscule de début de phrase et les mots-outils, et un nom ne peut plus chevaucher deux phrases. Le mode d'origine reste disponible et inchangé pour refaire les mesures passées.

### I2. Tête de choix réentraînée dans le format du modèle intégré (17 novembre 2026)

Code : [lfm3e/i2_head.py](lfm3e/i2_head.py). Résultats bruts : [results/i2_head.json](results/i2_head.json). Même tête que D11 et D14, mais les questions, les notes et les clés sont construites par le code du modèle intégré. Entraînement sur 11 292 questions PopQA et 7 500 HotpotQA, chacune vue seule et derrière la consigne des mesures. Jeux de test figés de D14, aucun article ni nom partagé avec l'entraînement.

Deux têtes comparées : l'une entraînée sur les questions seules, l'autre avec en plus 3 115 problèmes de calcul (GSM8K, partie entraînement) où la bonne réponse pour chaque nom est « aucune note ».

| | Sans tête (article le plus développé) | Questions seules | Questions et problèmes de calcul |
|---|---|---|---|
| PopQA : bonne note montrée (955 questions) | 50,4 % | **75,1 %** | 73,9 % |
| HotpotQA : au moins une bonne note (242) | 62,0 % | 61,2 % | 61,6 % |
| HotpotQA : les deux bonnes notes | 23,3 % | 23,3 % | 23,3 % |
| TriviaQA : questions recevant une note | 85,0 % | 78,2 % | 77,8 % |
| MKQA anglais : questions recevant une note | 37,2 % | 37,2 % | 37,2 % |
| GSM8K (300 problèmes de test) : problèmes recevant une note | 62,3 % | 55,7 % | **34,3 %** |
| Noms à plusieurs articles, bon article parmi les candidats (570) : premier choix juste | 38,8 % | 60,4 % | 59,7 % |
| Bon article absent des candidats (240) : la tête répond « aucun » | | 25,8 % | 29,2 % |

Lecture :
- **La tête fait son travail sur PopQA** : la bonne note passe de 50,4 % à 75,1 %. D14 affichait 73,0 % avec l'ancien format (dénominateur un peu différent, donc à peu près égal).
- **Elle n'apporte rien sur HotpotQA** : 61 à 62 % dans les trois cas.
- **Règle fixée avant le run** : la tête entraînée avec les problèmes de calcul n'est retenue que si la bonne note baisse de moins de 1 point. Elle baisse de 1,15 point sur PopQA : c'est donc la tête « questions seules » qui est exportée (`head.safetensors`). L'autre est conservée à côté.
- **Ce choix a un prix** : avec la tête exportée, 55,7 % des problèmes de calcul reçoivent une note qui n'a rien à voir (« Easter », « Tom »), contre 34,3 % avec l'autre. Ce que ces notes coûtent en bonnes réponses n'est pas mesuré.
- **La tête ne sait pas dire « aucun »** : quand le bon article n'est pas parmi les candidats, elle en montre quand même un trois fois sur quatre.

Aperçu de bout en bout, 200 premières questions PopQA à travers `generate` : 54 % avec mémoire, 12 % sans. Ce n'est pas une mesure (200 questions, une seule exécution).

**Deux défauts de déclenchement mis au jour, hors de portée de la tête :**
- Les termes techniques (règle de D6) déclenchent sur des prompts banals : « say hello in one word » reçoit la note de la chanson « Say Hello », un problème de calcul celle de « Piggy bank » ou de « Water slide ». La tête ne voit pas ces notes : elles passent par les titres exacts. C'est la première source des notes hors sujet restantes sur GSM8K.
- La règle des noms pour prompts quelconques (I1) perd 2 points de couverture sur PopQA par rapport à la règle d'origine : bon article parmi les candidats pour 85,2 % des questions contre 87,2 %. Ce qui reste perdu, ce sont les titres faits d'un nombre (« 2012 », « 18 ») ou d'un mot-outil (« One »), écartés exprès. Une première version coupait aussi les noms à initiale et les titres à deux-points (« James M. Geraghty », « Covert One: The Hades Factor ») : corrigé après le run, donc les chiffres du tableau sont un peu en dessous de ce que donne le code actuel.

### I3. Tête de choix v2 et essai de lecture par l'attention (17 novembre 2026)

Code : [lfm3e/i3_head.py](lfm3e/i3_head.py). Résultats bruts : [results/i3_head.json](results/i3_head.json).

Ce qui change par rapport à I2 :
- « aucune note » n'est plus une constante mais une fonction de la question ;
- 48 candidats par nom au lieu de 12, et un indicateur « le titre est exactement le nom » ;
- les notes trouvées par les titres exacts peuvent passer par la tête ;
- l'entraînement ne contient plus seulement des questions de quiz : 19 156 PopQA, 15 000 HotpotQA, 5 287 questions françaises fabriquées depuis les gabarits de PopQA, 3 000 consignes sur un nom (« Tell me about X »), et 6 413 prompts où aucune note n'est voulue (problèmes de calcul, questions fournies avec leur texte, réécritures).

Les jeux figés de D14 servent désormais de jeux de réglage. Des jeux finals ont été préparés à part (PopQA, TriviaQA, MKQA, HotpotQA, GSM8K), disjoints de l'entraînement et jamais regardés.

**Bonne note montrée (jeux de réglage)**

| Tête | PopQA (955) | PopQA en français (493) | HotpotQA, au moins une (242) |
|---|---|---|---|
| Sans tête | 51,6 % | 65,7 % | 64,0 % |
| v1 : celle de I2, avec 48 candidats | 81,4 % | 55,0 % | 63,2 % |
| v2, toutes les notes filtrées | 80,0 % | 80,9 % | 52,9 % |
| **v2, seuls les termes techniques filtrés (exportée)** | **80,3 %** | **81,5 %** | **61,6 %** |
| v2 avec 12 candidats | 77,3 % | 80,7 % | 52,9 % |
| v2 avec clés sans passe du modèle | 77,9 % | 78,3 % | 48,4 % |

**Prompts qui reçoivent une note**

| Tête | GSM8K | Code | Maths | Écriture libre | TriviaQA | MKQA anglais | MKQA français |
|---|---|---|---|---|---|---|---|
| Sans tête | 63,0 % | 54,0 % | 18,0 % | 42,0 % | 85,2 % | 37,2 % | 25,7 % |
| v1 | 56,3 % | 46,7 % | 15,0 % | 36,0 % | 79,2 % | 37,2 % | 17,7 % |
| v2, toutes filtrées | 0,0 % | 0,3 % | 1,0 % | 18,0 % | 56,8 % | 20,1 % | 21,9 % |
| **v2, termes seuls (exportée)** | **7,3 %** | **4,0 %** | **3,3 %** | 26,0 % | 72,0 % | 20,1 % | 24,0 % |

Le code et les maths sont des genres absents de l'entraînement : la tête a bien appris « ce prompt n'a besoin de rien », pas « ce prompt vient de GSM8K ».

Lecture :
- **Notes hors sujet : problème en grande partie réglé.** De 56 % à 7 % sur les problèmes de calcul, de 47 % à 4 % sur le code.
- **48 candidats : +2,7 points** de bonnes notes sur PopQA (77,3 % → 80,0 %, test apparié p < 0,001). La tête de I2 passe de 75,1 % à 81,4 % avec ce seul changement.
- **Français : 55,0 % → 81,5 %.** Mais ce jeu est fabriqué avec les mêmes gabarits que l'entraînement : chiffre optimiste, à confirmer sur MKQA français de bout en bout.
- **Filtrer toutes les notes coûte sur HotpotQA** : 63,2 % → 52,9 % (27 questions perdues, 2 gagnées, p < 0,001). Selon la règle fixée avant le run, c'est donc la variante « termes techniques seuls » qui est exportée : elle ne perd significativement ni sur PopQA (p = 0,21) ni sur HotpotQA (p = 0,29).
- **Clés sans passe du modèle : -2,1 points** sur PopQA (p = 0,054) et -4,5 sur HotpotQA. Non retenues par défaut ; elles restent une option pour un usage local où le temps compte.

**Risque ouvert de la tête exportée.** Sur MKQA anglais, elle ne montre plus une note que pour 20,1 % des questions, contre 37,2 % avant. Ces questions sont en minuscules, et leurs notes venaient surtout de la règle des termes, que la tête refuse désormais souvent. D14 gagnait 5,6 points sur ce jeu : ce gain peut avoir fondu. Même question sur TriviaQA (79,2 % → 72,0 %). Seule une mesure de bout en bout tranchera ; la tête v1 à 48 candidats est conservée pour la comparaison.

**Essai : laisser le modèle choisir par son attention.** On montre au modèle les premières lignes des 12 candidats d'un nom et on lit, dans les six couches d'attention globale, lequel le dernier token du prompt regarde le plus. 572 noms à plusieurs articles :

| Méthode | Premier choix juste |
|---|---|
| Article le plus développé | 39,0 % |
| Attention totale | 30,6 % |
| Attention, poids appris par tête d'attention | 49,5 % |
| Tête de choix v2 | **64,0 %** |

Abandonné : l'attention brute fait moins bien que le hasard informé, et même avec des poids appris elle reste 14 points sous la tête. Elle sépare parfaitement les questions de quiz des problèmes de calcul (aire sous la courbe 1,00 avec poids appris), mais la tête le fait déjà.

Essais à la main en local avec la tête exportée : plus aucune note sur « Tom has 3 apples… at Easter », « say hello in one word », « Write a Python function… » ; note correcte et en anglais pour « Tell me about Le Monde ». Deux échecs restent visibles : « capital of Norfolk » reçoit trois notes (le comté anglais en premier) et la réponse reste fausse ; « who sang hallelujah first » reçoit la note de l'interjection religieuse.

## 3. L'architecture retenue

**Principe.** Dans LFM3, les convolutions voient 7 tokens en arrière et les attentions à fenêtre 256. Au-delà, seules les 6 couches d'attention globale (2, 6, 10, 14, 18, 22) voient le contexte. « Le modèle a lu ce passage plus tôt » se résume donc aux clés et valeurs d'attention du passage sur ces 6 couches.

**Fonctionnement.**

1. **Mémoire.** Un dictionnaire nom → début d'article, stocké en texte.
2. **Déclenchement.** Recherche exacte des noms dans le flux de tokens, en temps constant, comme l'adressage d'Engram. Elle s'applique au prompt et au texte que le modèle génère, réflexion comprise.
3. **Lecture.** Le modèle gelé lit le passage une fois ; ses clés et valeurs sur les 6 couches globales deviennent consultables par la suite du contexte.
4. **Cache.** Les clés et valeurs des entités fréquentes sont gardées, les autres recalculées à la demande.

**Ordres de grandeur.**

| Élément | Taille |
|---|---|
| Texte d'un passage de 64 tokens | environ 140 octets |
| Clés et valeurs du même passage (6 couches, bf16) | 786 Ko |
| Wikipédia anglais, 6,4 millions de passages en texte | environ 1 Go |
| Lecture d'un passage de 64 tokens sur H100 | 0,7 ms au débit mesuré en lot (latence d'une requête seule non mesurée) |

**Ce qui est repris d'ailleurs.** Les caches d'attention précalculés existent (TurboRAG), les mémoires d'entités lues par l'attention aussi (TOME, KBLaM), mais elles demandent un entraînement ou un moteur de recherche. Ce qui est propre ici : l'adressage exact par nom, aucun entraînement, et le stockage limité à 6 couches que permet l'architecture de LFM3.

## 4. Problèmes ouverts et solutions envisagées (analyse du 17 novembre 2026)

Depuis cette analyse, I3 a traité trois lignes du tableau : notes hors sujet (réglé en grande partie), 48 candidats (+2,7 points), choix par l'attention (essayé, abandonné). Le français a une tête entraînée, à confirmer.

Mesures locales, sans le modèle, sur les jeux figés (955 questions PopQA et 242 HotpotQA dont le bon article est connu, 300 problèmes GSM8K).

### Où partent les points sur PopQA

| Étape | Questions qui passent | Perdu à cette étape |
|---|---|---|
| Le bon article est parmi les 12 candidats | 85,2 % | 14,8 % |
| dont : bon article au-delà du 12e candidat | | 7,6 % |
| dont : nom écarté exprès par la règle (nombre, mot-outil, minuscules) | | 3,5 % |
| dont : nom de la question absent de l'index (autre nom de l'article) | | 2,9 % |
| dont : un nom plus long a été préféré | | 0,7 % |
| La tête montre la bonne note | 75,1 % | 10,1 % |
| Le modèle répond juste avec la bonne note seule (D14) | 83,0 % de ces cas | 17 % |

Avec plus de candidats par nom, classés par taille d'article : 89,8 % à 24, 91,7 % à 48, 92,9 % à 300.

Sur HotpotQA le problème est ailleurs : le bon article est parmi les candidats pour 65,3 % des questions, et quand il y est, il est premier 94 fois sur 100. Pour 26,4 % des questions, aucun nom de la question ne mène à l'article : c'est le deuxième saut, que la question ne nomme pas. Et même avec les deux bonnes notes, le modèle ne répond juste qu'à 45,2 % (D14).

### D'où viennent les notes hors sujet (GSM8K)

| Source | Problèmes touchés | Ce qu'elle rapporte ailleurs |
|---|---|---|
| Règle des noms (jours, mois, prénoms : « Monday », « John », « April ») | 56,3 % ont un nom à candidats | C'est le chemin principal sur PopQA |
| Règle des termes techniques (« piggy bank », « ice cream », « watermelon ») | 19,7 % | Seul accès au bon article pour 0,2 % de PopQA et 2,1 % de HotpotQA |
| Titres exacts de noms (« Los Angeles », « Cookie Monster ») | environ 6 % | Compte dans les 75,1 % |

Durcir le seuil de rareté de la règle des termes ne suffit pas : à un seuil vingt fois plus strict, elle touche encore 9 % des problèmes.

### Problèmes et solutions

| Problème | Cause identifiée | Solution retenue | Ce qui reste à vérifier |
|---|---|---|---|
| Notes hors sujet sur les prompts qui ne sont pas des questions de culture générale | L'option « aucune note » de la tête est une constante : elle ne peut pas dire « ce prompt n'a besoin de rien ». Et deux chemins sur trois ne passent pas par la tête | « Aucune note » devient une fonction de la question. Toutes les notes passent par la tête. Exemples négatifs variés : calcul, consignes, texte fourni avec sa question | Que la tête n'apprenne pas « ressemble à un quiz » : exemples positifs sous forme de consigne, et taux de notes mesuré sur des prompts d'un genre absent de l'entraînement |
| Bon article hors des 12 candidats (7,6 %) | Trop d'homonymes pour 12 places | 48 candidats | La tête choisit-elle aussi bien parmi 48 ? Et le temps de calcul en local, d'où l'essai de clés sans passe du modèle |
| La tête se trompe d'article (10,1 %) | Deux projections linéaires sur des états gelés | Essai : laisser le modèle choisir, en lui montrant les titres des candidats et en lisant où va son attention | Jamais mesuré. Peut ne rien donner : le modèle se laisse peut-être attirer par le mauvais homonyme |
| Notes en trop à côté de la bonne | Jusqu'à 3 notes par nom | Mesurer 1, 2 ou 3 notes de bout en bout | |
| Noms écartés par la règle (3,5 %), questions tout en minuscules (MKQA : 37 % reçoivent une note) | La règle s'appuie sur les majuscules | Les réadmettre une fois que la tête sait refuser | Dépend du premier point |
| Autres noms d'un même article (2,9 % sur PopQA, une part des 26,4 % de HotpotQA) | L'export utilisé n'a pas les redirections | Les ajouter à l'index depuis les fichiers de Wikipédia (environ 130 Mo) | Gain réel inconnu |
| Deuxième saut (HotpotQA) | L'article n'est pas nommé dans la question | Chercher les noms cités dans la première note | Plafond bas : 45,2 % avec les deux bonnes notes. Priorité faible |
| Lecture : 17 % d'échecs avec la bonne note | Réponse hors des 96 et 128 tokens lus, ou mal lue | Lire la note en entier (192 et 256 tokens stockés) quand une seule note est montrée | À mesurer |
| Raisonnement non amélioré | La mémoire porte des faits, pas des méthodes | Pas de solution prouvée. Piste hors mémoire : la pénalité de répétition par défaut coûte environ 7 points sur MATH (D12) ; en exempter les chiffres | Modifie le modèle sans note : option à activer, à mesurer |
| Français | Tête entraînée en anglais seulement | Questions françaises fabriquées depuis les gabarits de PopQA et les titres français | Mesure réelle sur MKQA français |
| Mesures fragiles | J'ai corrigé la règle des noms en regardant les questions de test de PopQA. La règle « moins de 1 point » est plus fine que le bruit (environ 1,4 point sur 955 questions) | Les jeux actuels deviennent des jeux de réglage. Chiffres finals sur des jeux jamais regardés, comparaisons par test apparié | |

### Idée écartée par la mesure

Présélectionner les candidats par les mots qu'ils partagent avec la question : 79,0 % de couverture avec 6 candidats par taille et 6 par recoupement, contre 85,2 % avec 12 par taille. Abandonné.

### Non mesuré à ce jour

Mode réflexion avec le système final, suivi de consignes et appels d'outils avec des notes, coût réel d'une note hors sujet sur un problème de calcul.

## 5. Prochaines étapes

1. **Mesure de bout en bout sur les jeux de réglage**, à travers le modèle intégré : tête v1 à 48 candidats contre tête v2 exportée, sur PopQA, TriviaQA, MKQA anglais et français, HotpotQA, GSM8K. C'est elle qui dira si le filtrage des termes coûte sur MKQA et TriviaQA. Mêmes passes pour régler le nombre de notes par nom (1, 2 ou 3) et la longueur de note lue.
2. **Mesure finale, une seule fois**, sur les jeux jamais regardés.
3. **Redirections de Wikipédia** dans l'index, pour les autres noms d'un même article.
4. **Questions en minuscules** (MKQA) : la règle des noms s'appuie sur les majuscules.
5. **Portage MLX**, mode réflexion à mesurer, suivi de consignes et appels d'outils avec des notes.

## 6. Limites de ces mesures

- PopQA pose des questions dont la réponse est souvent dans la première phrase de l'article. D'autres jeux seront moins favorables.
- La correction est jugée par présence de la réponse dans le texte généré, ce qui laisse passer quelques faux positifs (les 3 à 4 % du contrôle en donnent l'ordre de grandeur).
- Le mode principal force l'absence de réflexion. Le mode avec réflexion n'est mesuré que sur 120 questions.
- Le test à 4 passages les lit à la suite ; la version réelle les lirait indépendamment.
- D3 et D4 portent chacun sur une seule exécution, sans répétition.
