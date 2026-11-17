# LFM3-1.7B + mémoire : résultats des diagnostics et nouvelle architecture

17 novembre 2026. Ce document remplace les sections 5 à 7 de [PLAN.md](PLAN.md) : les diagnostics ont écarté l'Engram classique et fait émerger une autre architecture. Les chiffres bruts sont dans `results/*.json`.

## 1. Conclusion

- **L'Engram classique est écarté pour ce modèle gelé.** Un vecteur ajouté au flux interne, même optimisé exprès, corrige 5 % des réponses fausses (contrôle : 3 %).
- **Une mémoire lue par l'attention fonctionne sans aucun entraînement.** Le début de l'article de l'entité, rendu visible aux 6 couches d'attention globale, corrige 73,5 % des réponses fausses sur PopQA, contre 74,5 % quand on colle le texte dans le prompt.
- **De bout en bout, sans entraînement, PopQA passe de 15,2 % à 54,7 %** quand le système trouve seul l'entité (plafond avec la bonne entité fournie : 72,5 %).
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

## 4. Réserves : où on en est

| Réserve | État | Preuve ou correction |
|---|---|---|
| Un vecteur porte trop peu d'information | Fermée | D3 l'a confirmé, d'où le passage à un texte complet |
| Le modèle gelé ne sait pas lire le signal | Fermée | D4 : 73,5 % contre 74,5 % pour le texte dans le prompt |
| Budget d'entraînement | Fermée | Aucun entraînement |
| Stockage | Fermée | Texte au lieu des clés et valeurs, cache pour les entités fréquentes |
| Tient-il avec la réflexion ? | Fermée | D4 : 0,8 % → 59,2 %, et moins de réponses inachevées |
| Moteur local | Fermée | POC : on modifie MLXL3 ou on écrit un moteur |
| Dérive de style du modèle instruct | Sans objet | On n'entraîne rien sur du texte brut |
| **Un passage hors sujet fait des dégâts** | **Réduite, pas fermée** | D5 : 71 % → 79 % de réponses justes conservées avec en-tête et consigne ; de bout en bout, 91 % |
| **Choisir la bonne entité automatiquement** | **Ouverte sur les homonymes** | D5 : 83 % de bonnes réponses quand le nom est unique, 37 % quand il a 5 candidats ou plus |
| La consigne système perturbe le modèle sans mémoire | Ouverte | D5 : 84 % des réponses justes conservées avec la consigne seule |
| La mise en page du test perturbe le modèle | À comprendre | D5 : 88 % conservées avec une mémoire rendue invisible |
| Couverture du passage | À améliorer | 83 % à 64 tokens, 90 % à 128 |
| Bon passage qui casse 4,7 % des réponses justes | À comprendre | Non analysé |
| Raisonnement en plusieurs sauts | Non testé | Demande le déclenchement en cours de génération |

Limites qu'aucune mémoire ne lève :

- Le raisonnement pur (logique, calcul).
- Les questions sans nom à quoi se raccrocher, tant que le modèle n'a pas écrit un candidat.
- Les faits absents de la mémoire.

## 5. Prochaines étapes

1. **D5, précision du déclenchement.** Liste réelle des homonymes (titres Wikipédia partageant un nom), choix du candidat par popularité et par proximité avec le contexte, mesure de bout en bout sur PopQA sans donner la bonne entité. Tester aussi un en-tête qui présente le passage comme une note de référence.
2. **D6, plusieurs sauts.** Déclenchement pendant la réflexion, sur des questions à deux sauts.
3. **Portage MLX.** Ajout des clés et valeurs aux 6 couches globales dans le cache, pour tourner en local.

## 6. Limites de ces mesures

- PopQA pose des questions dont la réponse est souvent dans la première phrase de l'article. D'autres jeux seront moins favorables.
- La correction est jugée par présence de la réponse dans le texte généré, ce qui laisse passer quelques faux positifs (les 3 à 4 % du contrôle en donnent l'ordre de grandeur).
- Le mode principal force l'absence de réflexion. Le mode avec réflexion n'est mesuré que sur 120 questions.
- Le test à 4 passages les lit à la suite ; la version réelle les lirait indépendamment.
- D3 et D4 portent chacun sur une seule exécution, sans répétition.
