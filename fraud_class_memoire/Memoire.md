---
title: "Des Silos aux Signaux : Stratification du Risque de Fraude et Centralisation Réglementaire des Paiements"
author:
  - Noah Brouard
date: "31/08/2026"
lang: fr
toc: false
toc-depth: 2
numbersections: false
geometry: margin=2.5cm
fontsize: 12pt
mainfont: "Times New Roman"
linestretch: 1.5
colorlinks: true
linkcolor: black
citecolor: black
urlcolor: black
toccolor: black
header-includes: |
  \renewcommand{\maketitle}{}
  \usepackage{etoolbox}
  \usepackage{booktabs}
  \usepackage{array}
  \usepackage{calc}
  \AtBeginEnvironment{longtable}{\footnotesize}
include-before: |
  \begin{titlepage}
  \centering
  \vspace*{3cm}
  {\Large Université de Lorraine -- IAE de Metz\par}
  \vspace{2.5cm}
  {\huge\bfseries Des Silos aux Signaux : Stratification du Risque de Fraude et Centralisation Réglementaire des Paiements\par}
  \vspace{3cm}
  {\Large Noah Brouard\par}
  \vspace{1cm}
  {\large Sous la direction de [personal info]\par}
  \vfill
  {\large Août 2026\par}
  \end{titlepage}

  \thispagestyle{empty}
  \mbox{}

  \renewcommand*\contentsname{Sommaire}
  \begingroup
  \footnotesize
  \setcounter{tocdepth}{2}
  \tableofcontents
  \endgroup
  \newpage
---

# Remerciements

*[personal info]*

\newpage

# I – Introduction générale

## I.1 – Introduction

La lutte contre la fraude aux paiements en Europe s'est longtemps heurtée à une contrainte structurelle : le secret bancaire interdit aux établissements de partager entre eux les IBAN identifiés comme frauduleux, permettant aux fraudeurs récidivistes d'opérer à travers plusieurs institutions sans être détectés. La loi Labaronne (2025) lève cette contrainte en France en créant le FNC-RF, un fichier national centralisant les signalements de comptes frauduleux entre PSP. Ce mémoire interroge la portée de cette centralisation : un scoring de risque par apprentissage automatique appliqué à une base de ce type permet-il de stratifier de manière robuste les fraudeurs récidivistes des faux positifs, et les signaux inter-institutionnels qu'elle rend accessibles justifient-ils empiriquement la centralisation réglementaire face à l'alternative structurellement privilégiée par la littérature, l'apprentissage fédéré (FL).
En l'absence d'accès au FNC-RF réel, une base proxy (Fake-RF) est construite à partir du jeu de données synthétique créé par IBM *AML world* [3], en simulant un processus de déclaration par banque combinant modèle de scoring et investigation. Trois hypothèses sont testées : la robustesse de la stratification par des modèles supervisés (H1, régression logistique, Random Forest, XGBoost, LightGBM, MLP) ; la supériorité d'un entraînement centralisé sur des schémas fédérés simulés (H2, FedAvg, FedAdam, FedAdam avec confidentialité différentielle, Fed-XGBoost); et la valeur prédictive des variables inter-institutionnelles, évaluée par SHAP (H3).
Les résultats supportent avec réserve H1, H2 et H3. L'ensemble des architectures discrimine les Vrai et faux positifs avec un ROC-AUC supérieur ou égal à 0,89, avec une précision globale comprise entre 0.74 et 0.92 et 0.63 et 0.83 sur Vrai-Positifs (H1). Chaque variante fédérée affiche une dégradation de performance par rapport au modèle centralisé équivalent (H2). L'attribution de cette dégradation est en partie due aux variables inter-institutionnelles, qui figurent parmi les prédicteurs les plus importants selon SHAP, avec un effet sur la performance (H3). Ces résultats, obtenus sur une base proxy et donc à interpréter avec prudence, suggèrent que la perte de performance du FL ne relève pas uniquement de contraintes de convergence, mais reflète une perte structurelle de signal, un argument empirique en faveur de la centralisation réglementaire de l'intelligence anti-fraude à l'échelle européenne.

\newpage

## I.2 – Problématique et hypothèses

Nous ébauchons donc la problématique suivante :

*Dans quelle mesure un scoring de risque par apprentissage automatique basé sur une classification entre Fraudeurs et Faux Positifs, appliqué à une base de données réglementaire centralisée de type FNC-RF, permet-il de distinguer de manière robuste les fraudeurs récidivistes des faux positifs et les signaux inter-institutionnels qu'elle rend accessibles démontrent-ils empiriquement la supériorité structurelle de la centralisation réglementaire sur l'apprentissage fédéré pour le renseignement anti-fraude européen ?* 

De cette problématique découlent les hypothèses suivantes :

H1 : La classification appliquée à une base centralisée de fraude stratifie de manière robuste les fraudeurs récidivistes des faux positifs.

H2 : Les modèles entraînés sur l’ensemble de la base centralisée présentent une performance supérieure aux modèles entraînés avec différents algorithmes de Federated learning, par PSP.

H3 : Les variables inter-institutionnelles (nombre d'institutions signalantes, délai inter-signalements, statistiques sur les institutions déclarantes et sous lesquels les comptes sont domiciliées) sont des prédicteurs significatifs de récidive, et ces signaux ne sont pas reproductibles de façon équivalente sous un schéma d'apprentissage fédéré (FL) ce qui constituerait l'argument empirique central en faveur de la centralisation réglementaire.
 
Afin de répondre à cette problématique, nous structurons notre réflexion en quatre parties.
La première contextualise notre démarche : nous y retraçons l'évolution de la fraude aux paiements en Europe, en soulignant notamment comment les avancées récentes des modèles générateurs (LLM, MLLM) ont élargi l'arsenal des fraudeurs, avant d'examiner le cadre réglementaire qui y répond et de présenter la base FNC-RF comme innovation centrale de ce dispositif.
La deuxième partie propose une revue de littérature articulée autour des cinq piliers théoriques de notre approche : le ML appliqué à la classification de risque dans des domaines variés (fraude, AML, risque de crédit et probabilité de défaut); le Federated Learning et ses limites structurelles ; la gestion du déséquilibre de classe en statistique et en ML ; et enfin l'explicabilité des modèles et l'interprétation de l'impact des variables.
La troisième partie détaille notre méthodologie : les protocoles expérimentaux, les données et les modèles retenus pour chaque hypothèse.
La quatrième et dernière partie présente nos résultats, les confronte à nos trois hypothèses, et discute les limites de notre démarche ainsi que les implications pour la conception d'une infrastructure de renseignement anti-fraude à l'échelle européenne.
 
# II – Partie 1 : Contextualisation

## II.1 – Évolution de la fraude en Europe

Simon Harris, Tánaiste et Ministre des Finances irlandais, révélait en 2025 que son identité avait été utilisée pour promouvoir plusieurs fonds d'investissement factices, dans le but de soutirer des fonds à des particuliers. Les fraudeurs avaient eu recours à des deepfakes générés par des modèles d'IA générative pour imiter sa voix et son visage avec un réalisme troublant [40]. Cet épisode illustre une convergence entre deux dynamiques structurelles : la progression continue des paiements électroniques et la démocratisation des outils d'intelligence artificielle générative [24], qui ensemble alimentent une hausse soutenue de la fraude aux paiements en Europe.

### II.1.1 – Une fraude en hausse absolue

Selon le rapport conjoint EBA/BCE sur la fraude aux paiements publié en 2025, le montant total de la fraude dans l'Espace Économique Européen a atteint 4,2 milliards d'euros en 2024, contre 3,5 milliards en 2023 et 3,4 milliards en 2022 — soit une progression de 17% en un an [20]. Cette hausse est principalement portée par deux catégories : les virements frauduleux (2,5 milliards d'euros, +24%) et les paiements par carte (1,3 milliard, +4%), ce sont précisément les typologies que cible la base de données FNC-RF. Il convient cependant de nuancer : cette progression est davantage imputable à une hausse de la valeur des transactions frauduleuse qu'à une explosion du nombre d'actes frauduleux [20].

### II.1.2 – Le tournant de l'ingénierie sociale

De nombreuses typologies de fraude aux paiements coexistent, des malwares bancaires aux faux messages SMS, en passant par les interceptions physiques d'identifiants. Mais la fraude qui enregistre la plus forte progression est la manipulation. Les virements frauduleux initiés par manipulation du payeur sont passés de 65% à 74% en valeur, et de 55% à 71% en volume, entre 2023 et 2024 [20]. Le fraudeur ne contourne plus le système, il s'attaque au jugement de la victime. On parle de social engineering et de fraude APP. 
Si les avancées réglementaires comme l'authentification forte (SCA) imposée par la PSD2 ont réduit certaines formes de fraude technique, elles ont simultanément orienté les fraudeurs vers des vecteurs moins contraints. L'EPC recense plusieurs typologies en forte croissance : usurpation d'identité de conseillers bancaires, fraude au "compte sécurisé", arnaques au support informatique, escroqueries d'urgence ou de récupération de fonds [21]. L'IA générative amplifie l'ensemble de ces vecteurs; phishing, smishing, deepfakes audio et vidéo; avec un réalisme qui dépasse les barrières linguistiques et culturelles [21]. L'affaire Simon Harris en est une illustration visible [40].

### II.1.3 – L'émergence des paiements agentiques : une frontière à surveiller

Au-delà des tendances actuelles, une évolution mérite d'être signalée, même si elle reste hors du périmètre direct de ce mémoire : l'essor des paiements agentiques. Ces systèmes délèguent à des agents IA autonomes la capacité d'initier, valider ou gérer des transactions pour le compte d'un utilisateur [5]. Les gains d'efficacité sont théoriquement réels. Or les moyens de compromission sont nombreux (données d'entraînement contaminées, prompt injections ou fuite de données personnelles par exemple). Donc les implications en matière de fraude sont profondes : un agent compromis devient un vecteur d'initiation frauduleuse à grande échelle, sans intervention humaine identifiable. Les cadres réglementaires existants comme la SCA mis en avant par PSD2 qui ont été conçus autour du consentement du payeur humain ne sont pas adaptés à ce paradigme, et les infrastructures comme le FNC-RF devront l'anticiper.

## II.2 – Le contexte réglementaire

Ce mémoire s’inscrit dans un contexte réglementaire européen et français particulier, celui de la Loi Labaronne en France, mais aussi des régulations sur les paiements PSD et PSD2 et de la directive européenne anti-fraude VOP. Nous examinerons celles-ci dans un ordre chronologique ci-dessous.

### II.2.1 – PSD / PSD2

Suite à l’essor des paiements électroniques au début du XXIe siècle, l’Union européenne a cherché à réguler l’espace des paiements d’abord avec la Directive sur les services de paiements, PSD [14], en 2007. Cette directive pose les bases des règles applicables aux *PSP*, *Payment Service Providers*. Suite à l’essor du commerce en ligne, l’émergence des Fintech et le développement des paiements mobiles, la Commission a révisé son cadre en adoptant la Directive sur les services de paiements révisée, PSD2 [15], en 2016, qui exigeait une transposition nationale en 2018. Les objectifs clés de PSD2 sont alignés sur les objectifs classiques de la Commission européenne, soit la promotion de la compétition et la protection des consommateurs. Pour PSD2, ils se déclinent en trois axes : le renfort de la sécurité des transactions, l’accroissement de la concurrence dans le secteur bancaire et auprès des tiers ainsi que l’harmonisation des pratiques au sein de l’espace européen.

Sur le plan de la lutte anti-fraude, PSD2 met en place la SCA, c’est-à-dire l’authentification forte qui exige la double validation des transactions. Ces évolutions ont eu un impact mesurable sur la fraude aux paiements en Europe. En effet, la fraude a diminué de près de 50 % entre 2020 et 2021 à la suite de son déploiement [55]. Néanmoins ses régulations se sont révélées insuffisantes pour lutter contre les fraudes dites APP par manipulation du paiement et social engineering plus ou moins complexe, comme exposé dans la partie II.1.

Une révision est en cours et la Commission prépare PSD3, dont un accord provisoire a été trouvé en novembre 2025 et dont la pleine application est attendue pour 2027-2028. Entre autres, elle renforce explicitement les exigences pour la détection de la fraude et introduit un régime de responsabilité élargi avec des limites de dépense, des authentifications sécurisées et remboursement pour les fraudes à l'impersonation décrites précédemment. 
 
### II.2.2 – VOP

L'adoption du Règlement (UE) 2024/886 sur les Paiements Instantanés (Instant Payments Regulation, IPR) [16], le 13 mars 2024, marque une étape décisive dans la sécurisation des paiements européens. Ce règlement impose à l'ensemble des *PSPs* de la zone euro d'offrir des virements instantanés en euros 24h/24, 7j/7, à un tarif équivalent aux virements classiques, obligation qui vise à généraliser l'usage des paiements en temps réel tout en encadrant les risques qui y sont associés.
La mesure phare du règlement, applicable depuis le 9 octobre 2025, est l'obligation de Verification of Payee (VoP) [23]. Ce dispositif impose aux PSPs de vérifier, avant toute exécution d'un virement SEPA, que le nom du bénéficiaire renseigné par le payeur correspond bien à celui associé à l'IBAN destinataire. Le résultat de cette vérification est communiqué sous la forme d'un statut (match, close match, no match) avant l'autorisation du paiement.
L'architecture du VoP repose sur un schéma interopérable défini par le European Payments Council (EPC) via son SEPA Verification of Payee Scheme Rulebook, entré en vigueur le 5 octobre 2025 (version 1.0) et mis à jour en mars 2026 (version 1.1). Concrètement, la banque du payeur envoie en temps réel une requête à la banque du bénéficiaire, qui renvoie le résultat de la correspondance. Le VoP représente la première infrastructure IBAN à portée européenne conçue explicitement pour contrer la fraude et constitue, à ce titre, le précurseur direct des mécanismes de centralisation qui suivront au niveau national.

### II.2.3 – La loi Labaronne

Malgré les avancées apportées par la PSD2 et le VoP, une faille structurelle persistait dans le dispositif de lutte contre la fraude aux virements : l'impossibilité juridique pour les établissements bancaires de partager entre eux les informations relatives aux comptes suspects. Le secret bancaire, principe fondamental du droit français, empêchait qu'un IBAN identifié comme frauduleux par une banque A soit signalé aux banques B, C ou D, qui pouvaient donc continuer à exécuter des virements vers ce compte pendant plusieurs jours, voire plusieurs semaines. Les Fraudeurs peuvent ainsi continuer à opérer sans avoir forcément à ouvrir de nouveaux comptes.

Cette faille est documentée par les données de l'Observatoire de la Sécurité des Moyens de Paiement (OSMP) de la Banque de France [11] : au premier semestre 2025, les fraudes par manipulation ont représenté environ 245 millions d'euros de pertes, soit près de 40 % de l'ensemble de la fraude aux moyens de paiement. Par ailleurs, selon l'exposé des motifs de la proposition de loi initiale, près de 48 % des fraudes aux virements bancaires en 2023 étaient liées aux arnaques aux faux IBAN [8], pour un préjudice de 149,76 millions d'euros. 

La proposition de loi n°884 est finalement adoptée : la Loi n° 2025-1058 du 6 novembre 2025, dite Loi Labaronne du nom de son rapporteur à l'Assemblée nationale Daniel Labaronne, apporte une réponse directe à cette faille en créant une dérogation expresse au secret bancaire au bénéfice d'un nouveau dispositif centralisé : le Fichier National des Comptes bancaires Signalés pour Risques de Fraude (FNC-RF). Adoptée à l'Assemblée nationale le 31 mars 2025, puis au Sénat le 29 octobre 2025 dans le cadre de la procédure de législation en commission, la loi instaure une obligation de signalement quotidien des IBAN suspects par l'ensemble des PSPs et institutions bancaires français. Les modalités de collecte, conservation et consultation des données sont définies par arrêté ministériel, après avis de la Commission Nationale de l'Informatique et des Libertés (CNIL), afin d’assurer la confidentialité et la protection des données relatives à des individus. Nous nous pencherons plus précisément sur le fonctionnement détaillé de la base FNC-RF dans le chapitre suivant [35]. 
En somme, SCA, VoP et FNC-RF s'inscrivent dans une trajectoire de sécurisation des paiements à l'échelle nationale et européenne, plaçant la protection du consommateur au cœur des dispositifs réglementaires. Il convient toutefois de souligner que cette logique de sécurisation implique, en contrepartie, une réduction progressive de l'anonymat transactionnel : chaque couche réglementaire supplémentaire exige une identification plus fine des acteurs du paiement, des noms aux IBANs, jusqu'aux comportements suspects centralisés. Cette tension entre protection et vie privée constitue l'un des défis structurels que tout système d'intelligence anti-fraude devra résoudre.
C'est précisément dans cette perspective que s'inscrit le projet FRIDA (Fraud Information Distribution Arrangement) du Conseil Européen des Paiements (EPC) [22]. Anticipant les obligations du futur Règlement sur les Services de Paiement (PSR), attendu pour 2028, l'EPC a établi une task force chargée de concevoir un schéma permettant aux PSPs d'échanger des informations sur la fraude selon des règles et standards communs à travers l'ensemble de la zone SEPA. En janvier 2026, l'EPC a lancé un appel à information pour identifier les opérateurs d'une plateforme centrale FRIDA, marquant le passage du projet à une phase opérationnelle concrète. Le FNC-RF français, dans cette lecture, n'est pas une exception nationale, il est un précédent : la démonstration empirique qu'une centralisation réglementaire de l'intelligence anti-fraude est juridiquement faisable et opérationnellement viable. C'est cette thèse que le présent mémoire entend évaluer et étayer sous l’angle empirique de la classification des comptes entre faux positifs — c'est-à-dire les comptes déclarés comme frauduleux mais innocentés — et fraudeurs, en comparant un système décentralisé et un système centralisé. 

## II.3 – La base FNC-RF

Comme vu dans le chapitre 2, le Fichier National Centralisé des Comptes de Paiement Frauduleux a été créé suite à l'adoption de la Loi Labaronne, reposant sur une dérogation spécifique au secret bancaire permettant la mutualisation des IBAN frauduleux entre PSP. 
 
### II.3.1 – Architecture et opération

Ces signalements sont centralisés et consolidés par la Banque de France, qui joue le rôle d'opérateur neutre et de tiers de confiance. La loi Labaronne instaure à cet effet une dérogation explicite au secret bancaire, condition nécessaire à la légalité du partage inter-institutionnel des données [35], le dispositif étant placé sous le contrôle de la CNIL afin de garantir la conformité au RGPD. Une fois un IBAN inscrit, les PSPs connectés peuvent l'interroger en temps réel afin d'informer leurs clients du risque associé à un virement ou, le cas échéant, de le bloquer préventivement. Concrètement, les PSPs sont tenus de créer un événement horodaté pour chaque déclaration de fraude, renseignant diverses caractéristiques : type et source de la fraude, canal d'origine, nature de l'opération, identifiants du compte et statut de l'événement. Chaque événement se voit attribuer un identifiant unique (UUID), ainsi que les dates de publication, d'occurrence et de mise à jour. Les seules données à caractère personnel mobilisées sont l'IBAN du compte présumé frauduleux, lequel ne sert que de clé d'identification et peut être chiffré afin d'assurer une conformité avec le RGPD. Ce sont les événements qui servent de base pour bloquer les IBANs. Néanmoins, un grand nombre d'événements sont des faux positifs, c'est-à-dire des IBANs finalement innocentés après investigation. En effet seulement 23% des IBANs seraient véritablement frauduleux à cette date [63].

Nous nous garderons de détailler davantage la base FNC-RF. En effet, n'ayant pas eu accès aux données, nous utilisons la base créée par IBM et leur générateur *AML World* [3].
 
\newpage
### II.3.2 – Positionnement par rapport à la littérature

La littérature en matière de détection de fraude inter-institutionnelle a largement privilégié l'apprentissage fédéré (FL) comme solution au problème fondamental du partage de données sous contraintes de confidentialité [9, 28]. Dans ce paradigme, chaque institution entraîne un modèle local sur ses propres données, seuls les gradients ou paramètres agrégés étant partagés, préservant ainsi la confidentialité des données brutes. Si cette approche constitue une réponse élégante aux contraintes légales, elle induit structurellement une perte de signal : les relations inter-PSP, par exemple un même IBAN signalé par plusieurs entités ou le temps entre deux déclarations d'un fraudeur, ne sont pas observables dans un schéma fédéré sans mécanismes d'agrégation complexes et potentiellement bruités. 

# III – Partie 2 : Revue de littérature

Nous verrons ici une revue de littérature de ML appliqué à la classification et plus précisément à la classification des risques. Ensuite nous nous pencherons plus précisément sur le federated learning, la gestion des déséquilibres de classes et l'applicabilité des modèles, centrale à l'expérimentation présentée dans ce mémoire.


## III.4 – ML appliqué à la classification de risques

### III.4.1 – Introduction

La détection et la classification des risques financiers constituent l'un des domaines d'application les plus anciens des méthodes statistiques en général [19]. Depuis les premiers travaux sur la notation de crédit dans les années 1990, en passant par des méthodes non paramétriques classiques comme les *KNN* [17], jusqu'aux architectures de deep learning du début du siècle [32],
la littérature a progressivement affiné sa capacité à identifier des patterns anormaux dans des données financières massives et hétérogènes. Ce chapitre présente un panorama structuré des approches retenues dans ce mémoire, choisies par souci de complétude et pertinence dans le cadre du problème de stratification du risque IBAN posé par le FNC-RF, tout en préservant une certaine interprétabilité afin de pouvoir conduire une démarche expérimentale.

Plus récemment, cette littérature s'est diversifiée selon les domaines d'application couverts par le *Machine Learning* (*ML*) a la classification de risque : la prédiction d'approbation de carte de crédit combinant ML et DL [50], le scoring de crédit adaptatif en prêt numérique [6], la détection de blanchiment d'argent sur transactions mobiles par deep learning [25], ou encore la gestion du risque interne par scoring adaptatif et détection de menaces assistée par LLM [30]. Concernant plus spécifiquement la fraude aux paiements, des approches alternatives comme l'apprentissage par renforcement profond [51] ou les autoencodeurs [47] ont également été explorées dans la littérature, bien qu'elles ne soient pas retenues dans le cadre de ce mémoire.
Il convient de souligner d'emblée que le problème traité dans ce mémoire se distingue de la détection de fraude classique : l'ensemble des observations est déjà composé d'IBANs signalés comme frauduleux. L'objectif n'est donc pas de séparer transactions frauduleuses et légitimes, mais de stratifier le risque au sein d'une population homogènement suspecte en distinguant les récidivistes systémiques des faux positifs et des cas isolés. Nous commencerons par une revue simple des processus de base du Machine Learning dans le cadre de problèmes de classifications.

### III.4.2 – Régression logistique

La régression logistique, introduite par Cox [18], constitue le point de départ naturel de toute tâche de classification binaire ou multiclasse en contexte financier. Elle modélise la probabilité d'appartenance à une classe par une fonction sigmoïde appliquée à une combinaison linéaire des variables explicatives. Son interprétabilité est intrinsèque, chaque coefficient est directement interprétable comme un log-odds, ce qui en fait un modèle de référence privilégié dans les environnements réglementaires et académiques. Elle est définie comme : 

$$P(Y=1\mid X) = \frac{1}{1 + e^{-z}}$$
$$\ln\left(\frac{P}{1-P}\right) = \beta_0 + \beta_1 x_1 + \dots + \beta_n x_n$$

où $z = \beta_0 + \beta_1 x_1 + \dots + \beta_n x_n$.

Dans le contexte du scoring de risque IBAN, la régression logistique servira de baseline interprétable permettant d'établir une performance de référence et de valider la pertinence des variables construites avant d'introduire des modèles plus complexes. Ses limites sont bien documentées : elle suppose une relation linéaire entre les features et le log-odds, ce qui la rend inadaptée à la capture d'interactions non-linéaires entre variables [44]. On note que dans ce mémoire lorsqu'on parle de modèle linéaire on fait référence à une régression logistique.


### III.4.3 – Descente de gradient et bases du ML

Lorsque l'on a un problème de classification comme celui-ci, la résolution du modèle revient à chercher à minimiser la fonction d'objectif choisie, c'est-à-dire la fonction de perte (ou fonction de coût) qui mesure l'écart entre les prédictions du modèle et les étiquettes réelles. Pour une classification binaire le cas qui nous intéresse ici sur notre base Fake-RF, où l'on distingue fraudeurs et faux positifs, la perte la plus couramment utilisée est l'entropie croisée binaire *(binary cross-entropy, ou log-loss)*, directement liée à la log-vraisemblance négative d'un modèle probabiliste :

$$L(\theta) = -\frac{1}{N}\sum_{i=1}^{N}\left[ y_i \log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i) \right]$$

où $y_i \in \{0,1\}$ est l'étiquette réelle de l'observation i, $\hat{y}_i = f_\theta(x_i)$ la probabilité prédite par le modèle paramétré par θ, et N le nombre d'observations. Contrairement à la régression linéaire, dont les coefficients s'obtiennent par une solution analytique fermée (équations normales), cette fonction n'admet pas de solution fermée pour la régression logistique du fait de la non-linéarité introduite par la fonction sigmoïde ; on a donc recours à une solution itérative. On utilise la descente de gradient, la *Stochastic Gradient Descent* ou *SGD*, particulièrement lorsque les bases de données sont de taille importante. On met à jour les paramètres des modèles dans le sens inverse du gradient de la perte (ici le *cross-entropy loss*), avec la mise à jour contrôlée par le taux d'apprentissage ou learning rate. 

Pour un mini-batch $B_t$ d'observations échantillonnées uniformément :

$$ \hat{g}_t = \frac{1}{|B_t|} \sum_{i \in B_t} \nabla_\theta L_i(\theta_t) $$

$$ \theta_{t+1} = \theta_t - \eta \,\hat{g}_t $$


*SGD* est issue des travaux fondateurs de Robbins & Monro [62], qui approxime le gradient sur un sous-ensemble aléatoire de B observations. La quantité B et le taux d'apprentissage η deviennent dès lors deux hyperparamètres, ceux-ci deviennent important pour assurer la convergence du modèle [56]. Ce qui nous mène donc à la partie suivante. 

#### III.4.3.1 – Hyperparamètres

Contrairement aux paramètres du modèle, comme les poids et biais d'un réseau de neurones ou les poids des feuilles d'un arbre de décision, qui sont appris au cours de l'entraînement, les hyperparamètres sont les paramètres de configuration du modèle qui sont fixés avant que le processus d'apprentissage ne commence. Ils ne sont pas appris mais conditionnent directement la manière dont l'apprentissage se déroule. En sont des exemples canoniques le taux d'apprentissage, la taille de batch, le nombre d'itérations ou de rounds (nombre d'arbres T pour les méthodes de boosting [13][29]), la profondeur maximale des arbres, les coefficients de régularisation L1/L2, ainsi que la pondération de classe​, introduite en section (III.6.2).
Ils permettent entre autres de gérer la problématique bias-variance. Un taux d'apprentissage trop élevé provoque une divergence de l'optimisation. Une profondeur d'arbre trop grande favorise le surapprentissage, trop petite, un biais élevé. La qualité du modèle final dépend donc en partie du choix de ces hyperparamètres. Comme l'ont montré Bergstra & Bengio [57], l'espace de recherche est en pratique de dimension élevée et non homogène, ce qui rend son exploration difficile. Dans le cadre de ce mémoire, ces hyperparamètres sont déterminés de manière systématique par optimisation bayésienne. 

#### III.4.3.2 – Optimisation bayésienne

L'optimisation bayésienne (BO) est un algorithme efficace pour explorer l'espace des hyperparamètres, dans le cas où l'évaluation de la fonction d'objectif est coûteuse et où celle-ci est considérée comme une boîte noire, sans expression analytique connue ni gradient disponible [59]. Elle est particulièrement adaptée au problème qui nous occupe : la performance du modèle en fonction des hyperparamètres ne peut être évaluée qu'empiriquement, et chaque entraînement sur la base Fake-RF demande de la puissance de calcul et du temps. Deux choses dont nous ne disposons pas dans ce mémoire.

Le principe repose sur deux composantes. D'une part, un *surrogate model* est entraîné sur les points déjà évalués afin de modéliser la distribution a posteriori de la fonction objectif sur l'espace des hyperparamètres. D'autre part, une fonction d'acquisition, telle que l'Expected Improvement (EI) ou l'Upper Confidence Bound (UCB), pilote la sélection du prochain point à évaluer en arbitrant entre exploitation (c'est-à-dire rechercher près des zones déjà prometteuses) et exploration (où l'on va plutôt visiter les régions de forte incertitude) [58, 59]. Cette stratégie permet de trouver un bon jeu d'hyperparamètres en un nombre d'évaluations très réduit, typiquement de l'ordre de quelques dizaines à quelques centaines d'essais. De plus, il existe une implémentation fonctionnelle, *scikit-optimize* (*skopt*), compatible avec le *stack* utilisé dans le cadre de ce mémoire. 

Le package skopt supporte les espaces continus, entiers et catégoriels, ainsi que plusieurs mécanismes de substitution (processus gaussien, *TP Estimator* de Bergstra et al. [60], arbres de forêt aléatoire), ce qui est essentiel pour explorer des espaces mixtes tels que ceux de XGBoost et LightGBM (III.4.4) [61]. C'est cet outil qui est utilisé pour tuner chaque architecture de H1 (ainsi que le méta-learner de H2) sur une durée fixée, afin de garantir la comparabilité des résultats (cf. annexe VII.1).

### III.4.4 – Arbres de décision et boosting

#### III.4.4.1 – Arbres de décision et Random Forest

Introduit par Breiman (2001) [12], le Random Forest est un algorithme d'ensemble par *bagging*. Il entraîne un ensemble de $B$ arbres de décision indépendants $\{T_b\}$ sur des échantillons bootstrap des données. Pour une nouvelle observation $x$, la prédiction finale $\hat{y}$ est obtenue par agrégation :

$$\hat{y} = \frac{1}{B}\sum_{b=1}^{B} T_b(x) \quad \text{(pour la régression)}$$
$$\hat{y} = \operatorname{mode}\{T_b(x)\} \quad \text{(pour la classification)}$$

Cette double randomisation sur les données et sur les variables réduit la variance du modèle final sans augmenter significativement le biais, rendant le Random Forest naturellement robuste au surapprentissage.
Dans le contexte du FNC-RF, cette architecture offre un double intérêt. D'une part, elle constitue un point d'entrée conceptuel aux méthodes d'ensemble avant l'introduction du *boosting*. Ses limitations par rapport aux méthodes de *boosting*; moindre capacité à corriger le biais, performances inférieures sur données déséquilibrées comme dans la base FNC-RF justifient toutefois la progression vers XGBoost, LightGBM et CatBoost [41].

#### III.4.4.2 – XGBoost

XGBoost (*Extreme Gradient Boosting*), introduit par Chen et Guestrin (2016) [13], représente une avancée majeure dans les méthodes de Gradient Boosting, particulièrement pour les données tabulaires. Sa supériorité repose sur une fonction objectif optimisée, qui combine une fonction de perte dérivable L et un terme de régularisation Ω pour contrôler la complexité du modèle :
$$\text{Obj}(\Theta) = \sum_i l(y_i, \hat{y}_i) + \sum_k \Omega(f_k)$$

où la régularisation est définie comme $\Omega(f) = \gamma T + \frac{1}{2}\lambda \|w\|^2$, $T$ étant le nombre de feuilles et $w$ le vecteur des scores des feuilles. Cette structure intègre nativement la régularisation L1/L2, permettant de prévenir le surapprentissage tout en gérant efficacement les valeurs manquantes et le parallélisme de calcul. Cette architecture a été largement validée dans les applications de scoring de risque et de détection de fraude [44] grâce à sa capacité à modéliser des interactions non linéaires complexes entre variables, couplée à une interprétabilité renforcée via SHAP ainsi qu’une grande efficience en termes de puissance de calcul.

#### III.4.4.3 – LightGBM

LightGBM (Ke et al., 2017) [29] optimise le processus de *Gradient Boosting* via deux innovations algorithmiques majeures qui améliorent significativement l'efficacité computationnelle sur des bases volumineuses. Premièrement, le *Gradient-based One-Side Sampling* (*GOSS*) sélectionne les instances pour l'apprentissage en conservant celles ayant les gradients les plus élevés — supposées apporter le plus d'information pour la convergence du modèle — tout en conservant un sous-ensemble aléatoire des instances à gradient faible pour maintenir la distribution. Cette technique réduit la complexité de l’entraînement de O(n) à O(log n).
Deuxièmement, *l'Exclusive Feature Bundling* (*EFB*) réduit la dimensionnalité en regroupant les *features* mutuellement exclusives (dont la valeur est rarement non-nulle simultanément) en un seul "bundle", diminuant ainsi le coût de construction des arbres. Ces optimisations, couplées à une approche Leaf-wise (plutôt que Tree-wise pour XGBoost) visant à minimiser la perte globale par la division de la feuille avec le gain le plus élevé, permettent une convergence rapide sans perte de performance. Dans le cadre de ce mémoire, cette efficacité computationnelle — supérieure à celle de XGBoost — constitue un atout crucial pour la simulation des multiples modèles en environnement de Federated Learning.

### III.4.5 – Deep learning

#### III.4.5.1 – Concepts fondamentaux : réseaux de neurones

Un réseau de neurones de type *feed-forward* (*FFN*) peut être défini mathématiquement comme un approximateur universel de fonctions transformant une entrée x en une sortie y par une succession de couches de neurones empilées. À chaque couche l, le réseau effectue d'abord une transformation linéaire des activations de la couche précédente :
$$z^{(l)} = W^{(l)} a^{(l-1)} + b^{(l)}$$

Une fonction d'activation non linéaire $\sigma$ est ensuite appliquée pour obtenir les activations de la couche courante, permettant au modèle de capturer des relations complexes, nous choisirons ReLU dans le cadre de notre mémoire :

$$a^{(l)} = \sigma\!\left(z^{(l)}\right)$$

avec, 

$$\sigma(x) =\max(0,x)$$

Le processus de propagation avant (*forward pass*) se définit par la chaîne suivante, partant de l'entrée $a^{(0)} = x$ :

$$a^{(l)} = \sigma\!\left(W^{(l)} a^{(l-1)} + b^{(l)}\right), \qquad \hat{y} = \sigma_{out}\!\left(W^{(L)} a^{(L-1)} + b^{(L)}\right)$$

L'apprentissage du réseau consiste à optimiser les poids $W$ et les biais $b$ en minimisant une fonction de perte $L(\hat{y}, y)$ via des algorithmes de descente de gradient et la rétropropagation de l'erreur.

#### III.4.5.2 – FFN & MLP

Comme évoqué en section 4.1 avec les travaux de Rumelhart, Hinton et Williams (1986) [45], les réseaux de neurones feedforward constituent le fondement du deep learning appliqué aux données tabulaires. Leur capacité à empiler des couches dotées de fonctions d'activation non linéaires permet l'apprentissage de représentations complexes. Chaque couche successive transforme les données en un hidden state de plus en plus abstrait, enrichissant ainsi la capacité prédictive du modèle. Nous intégrons ces architectures dans ce mémoire afin d'assurer un test exhaustif de l'hypothèse H1.
Pour cette expérimentation, nous déployons un *Multi-Layer Perceptron* (MLP) doté d'une architecture spécifique incluant des couches d'*embeddings*, de normalisation et de blocs de type *Gated MLP*. Le traitement des variables catégorielles s'appuie sur des *embeddings* [10] : chaque catégorie est associée à un vecteur de dimension fixe (par exemple 8) appris par le modèle. Ces vecteurs, une fois projetés, sont concaténés aux variables continues pour former le vecteur d'entrée global du modèle, distinguant les fraudeurs des faux positifs selon la différence normalisée entre le reporting d'un événement et son follow-up.
Ce vecteur d'entrée traverse ensuite une couche de *LayerNorm* [10] qui applique une normalisation centrée-réduite par échantillon, stabilisant l'apprentissage et évitant la dégradation des gradients. Le cœur de notre architecture repose sur des blocs *Gated MLP* [46] : chaque bloc exploite un mécanisme de porte qui combine linéairement deux projections distinctes du vecteur d'entrée, permettant une sélection adaptative des signaux avant application d'une fonction d'activation ReLU. Enfin, une tête de sortie linéaire projette la représentation apprise vers l'espace des classes, générant les logits nécessaires au calcul des probabilités.
Des architectures plus récentes fondées sur les *Transformers*, à l'image de Tabformer [39] pour la modélisation de séries temporelles tabulaires multivariées ou de FraudTransformer [7] pour la détection de fraude transactionnelle time-aware, illustrent l'évolution récente du domaine ; elles ne sont toutefois pas retenues dans ce mémoire, en raison de leur coût computationnel et de la taille limitée des sous-ensembles par banque dans Fake-RF.

### III.4.6 – Stacking et ensembles

Au-delà des modèles individuels, les approches de *stacking*, où l’on entraîne un méta-modèle sur les prédictions des modèles de base entraînés à prédire l’objectif sur les données d’entraînement, ont démontré des gains de performance systématiques sur les tâches de détection de fraude [2, 26]. Le méta-modèle, typiquement une régression logistique ou un XGBoost léger, apprend à pondérer optimalement les prédictions des modèles de base en fonction de leurs forces et faiblesses respectives, ce qui renforce la régularisation et réduit de facto le surapprentissage, dans une logique similaire à celle d’une *Random Forest*, qui, tout comme celle-ci, présente l’avantage supplémentaire de réduire la variance des prédictions, problème particulièrement pertinent dans le cadre d’une base de données en cours de maturation, où le risque de surapprentissage sur des patterns non-généralisables est non négligeable. Nous l’utiliserons dans le cadre des tests de H2 avec l’architecture proposée par Zhang et al. (2024) [53].

### III.4.7 – Métriques d'évaluation
 
Le choix des métriques d’évaluation est déterminant dans un contexte de classification de risque asymétrique. L’exactitude globale (*accuracy*) est une métrique trompeuse lorsque les classes sont déséquilibrées ou lorsque les coûts d’erreur sont hétérogènes. Les conséquences d’un compte bloqué à tort ne sont pas les mêmes qu’un compte sous investigation classé comme risque faible qui récidive.
Les méthodes d’évaluation suivantes sont retenues pour mesurer la performance des modèles :

**1. Précision et Rappel**

La précision ($P$) mesure la proportion de prédictions positives correctes :

$$P = \frac{VP}{VP + FP}$$

Le rappel ($R$, ou sensibilité) mesure la proportion de cas positifs réels correctement identifiés :

$$R = \frac{VP}{VP + FN}$$

**2. F1-Score**

Le F1-Score est la moyenne harmonique de la précision et du rappel :

$$F1 = \frac{2 \cdot P \cdot R}{P + R}$$

**3. AUROC (*Area Under the Receiver Operating Characteristic*)**

AUROC évalue la capacité de discrimination du modèle sur l'ensemble des seuils de décision. La courbe ROC trace le Taux de Vrais Positifs (TPR) en fonction du Taux de Faux Positifs (FPR) :

$$TPR = \frac{VP}{VP + FN}, \qquad FPR = \frac{FP}{FP + VN}$$

AUROC correspond à l’aire sous cette courbe, où 0,5 indique une classification aléatoire et 1,0 une séparation parfaite.
La validation croisée k-fold stratifiée, c’est-à-dire que l’on segemente les données d’entraînement en 3 sous-parties avec chacune leur donnée de test, tient compte du déséquilibre résiduel entre les deux classes et des variances temporelles, afin de s’assurer de créer un modèle robuste et représentatif. Elle est appliquée uniformément aux trois conditions expérimentales pour l’entraînement (H1, H2, H3) afin de garantir la comparabilité des résultats. 
Enfin, la recherche des hyperparamètres optimaux par optimisation bayésienne s’appuie sur la métrique *Predict proba*, c’est-à-dire la confiance des prédictions du modèle en moyenne pour les différentes classes.

## III.5 – Federated Learning

L’introduction du federated learning (FL) par McMahan et al. (2017) [38], qui formalisent un paradigme d'entraînement permettant à plusieurs institutions de construire collaborativement un modèle global sans jamais centraliser leurs données d'entraînement. Ce cadre répond à une contrainte structurelle bien documentée dans la littérature : la tension entre la nécessité d'agréger des signaux distribués pour améliorer la robustesse des modèles et les impératifs légaux, secret bancaire, RGPD, souveraineté des données, qui s'opposent donc à leur centralisation [52]. 

Dans le domaine financier, chaque prestataire de services de paiement (PSP) ou institution bancaire détient des données de signalement propres à sa clientèle, protégées réglementairement et commercialement sensibles. En l'absence d'un cadre légal autorisant leur mise en commun, le FL constitue, selon la littérature, la solution de référence pour entraîner des modèles collaboratifs inter-institutionnels, notament des parties tierces qui ne peuvent agréger les données de leur clients,  tout en respectant ces contraintes. Awosika et al. (2024) [9] illustrent ce cas d'usage en montrant qu'un dispositif FL appliqué à la détection de fraude financière permet d'améliorer les performances prédictives par rapport aux modèles entraînés en silo, tout en maintenant la confidentialité des données de chaque participant. Ce résultat constitue le point de départ empirique du présent mémoire, dans la mesure où il établit les conditions sous lesquelles le FL représente un progrès par rapport à l'absence de coopération inter-institutionnelle, avant l'émergence de bases centralisées comme FNC-RF ou la future FRIDA. Nous présenterons ci-dessous deux algorithmes de référence que nous comptons tester, ainsi que les solutions existantes afin de limiter la fuite de données par les gradients d'entraînement du modèle comme le démontrent Zhu et al. [64].

### III.5.1 – FedAvg

L'algorithme *Federated-Averaging* (*FedAvg*) de McMahan et al. (2017) [38] permet une mise en place viable du *federated learning*. Son fonctionnement repose sur un schéma itératif en cycles de communication coordonnés par un serveur central :le serveur diffuse le modèle global courant à un sous-ensemble aléatoire de clients.
Chaque client entraîne localement ce modèle sur ses données pendant plusieurs époques avec une taille de batch, produisant une mise à jour locale.
Le serveur agrège les mises à jour par moyenne pondérée selon le volume de données de chaque client :
$$w_{t+1} \leftarrow \sum_{k=1}^{K} \frac{n_k}{n} \, w_{t+1}^{k}$$

Où $w$ est le poids du modèle, $K$ le nombre de clients et $n$ le nombre total de données.
McMahan et al. (2017) démontrent empiriquement que FedAvg réduit les coûts de communication, tout en maintenant des performances comparables sur des tâches de classification d'images. De manière notable, les auteurs montrent que l'algorithme reste robuste aux distributions non-IID, c'est-à-dire aux situations où les distributions de données diffèrent significativement entre clients. Cette propriété est directement pertinente dans un contexte multi-PSP, où les profils de fraude varient selon les institutions et leurs clientèles respectives. Cependant, Li et al. (2020) [33] contestent fortement cette notion, montrant que la convergence finale vers le minimum global est biaisée par une erreur correspondant à l'hétérogénéité des distributions de chaque base de données clients. En effet, intuitivement, lorsque chaque PSP entraîne son modèle local sur ses propres données de fraude, celui-ci converge progressivement vers la solution optimale pour ce PSP spécifiquement et non vers le minimum global. Plus les profils de fraude diffèrent entre PSPs, plus les modèles locaux divergent les uns des autres, et plus leur agrégation produit un modèle global dégradé.

### III.5.2 – FedAdam

Comme vu dans le paragraphe précédent, les travaux postérieurs à FedAvg ont mis en évidence plusieurs limites de convergence dans les contextes fortement hétérogènes, comme le *client-drift*.
En réponse, Reddi et al. (2020) [42] proposent le cadre général FedOpt, dont l'idée centrale est de distinguer ce que fait chaque client localement (toujours SGD) de la manière dont le serveur agrège les mises à jour. Dans FedAvg standard, le serveur se contente de faire une moyenne des modèles reçus, ce qui revient à appliquer SGD avec un taux d'apprentissage fixe de 1 sur tous les paramètres sans distinction. FedOpt ouvre la possibilité d'utiliser un optimiseur plus sophistiqué au niveau de l'agrégation serveur :
$$x_{t+1} = \text{ServerOpt}(x_t, -\Delta_t, \eta, t)$$

où $\Delta_t$ est la moyenne pondérée des mises à jour envoyées par chaque client, c'est-à-dire la différence entre le modèle local après entraînement et le modèle global de départ, et non les gradients bruts eux-mêmes.
FedAdam spécialise ce cadre en remplaçant la moyenne simple par un optimiseur de type Adam (*Adaptive Moment Estimation*) [65] côté serveur. La différence pratique est significative : là où FedAvg applique le même taux d'apprentissage à toutes les variables du modèle, FedAdam ajuste automatiquement ce taux variable par variable selon l'historique des mises à jour. Une variable qui reçoit des mises à jour fortes et cohérentes entre PSPs se voit appliquer un taux plus faible étant déjà bien intégrée dans les poids du modèle. À l'inverse, une variable rare mais informative, par exemple un type de fraude peu fréquente mais discriminante comme un écart temporel entre deux événements dans deux banques distinctes, reçoit un taux plus élevé lui permettant d'être mieux capturée malgré sa rareté. C'est précisément le type de signal présent dans des données de fraude IBAN à distribution asymétrique.
Reddi et al. (2020) [42] établissent théoriquement et empiriquement que cette adaptativité améliore significativement la convergence sur des tâches à gradients épars, et confère une plus grande robustesse au réglage des hyperparamètres. Dans le cadre expérimental du mémoire, FedAdam constitue donc l’implémentation FL retenue pour la simulation multi-PSP (H2), pour sa supériorité documentée en contexte hétérogène non-IID et sa pertinence pour les cas de fraudes situées dans la longue queue des distributions. Or elle présente des limites, ce qui conforte théoriquement H2. En effet selon FedAdam [34], premièrement les gradients hétérogènes reçus de chaque participant créent de l’instabilité dans l’agrégation de données non-IID. Deuxièmement, la tendance de l’optimiseur Adam au surapprentissage sur des données locales lors de sa mise à jour pourrait renforcer le client drift. A chaque communication vers le serveur la mise à jour des estimations réduit la vitesse de convergence et pourrait biaiser le minimum global obtenu. Cette thèse est soutenue par une étude comparative récente [37] qui met en lumière que FedAdam a des difficultés sur des bases de données plus complexes et hétérogènes.

### III.5.3 – Federated XGBoost

L'application du FL aux modèles à base d'arbres de décision, et en particulier à XGBoost, soulève une difficulté algorithmique fondamentale : contrairement aux réseaux de neurones, XGBoost ne dispose pas de paramètres continus agrégeables par moyenne pondérée. L'application directe de FedAvg ou FedAdam à un modèle à gradient *boosting* est donc structurellement impossible, dans la mesure où ces algorithmes opèrent sur des vecteurs de poids différentiables, hypothèse que les ensembles d'arbres ne satisfont pas.
Zhang et al. (2024) [53] documentent cette contrainte dans le cadre de la prévision de risque de crédit sous FL, en adoptant un schéma d’agrégation distinct pour XGBoost. On agrège des arbres indépendants lors du premier round, puis on renvoie l’arbre agrégée aux clients. Pour les rounds suivants, on crée un ensemble par agrégation des prédictions de cet arbre agrégée avec un Meta-Learner (dans leur cas un CNN qui utilise les prédictions pour chaque dataset privé et devient « learnable » avec les algorithmes de Federated Learning classique). 
Il convient toutefois de souligner que ce schéma constitue un paradigme fédéré structurellement différent de FedAdam appliqué aux FNNs : FedAdam agrège des gradients et met à jour un modèle global unique par optimisation adaptative. Or FedXGBoost n'offre pas de correction du *client-drift*. Zhang et al. (2024) [53] observent d’ailleurs que les modèles XGBoost fédérés présentent, sur certaines configurations non-IID, des performances inférieures au modèle centralisé, là où les modèles de DL fédérés tendent à s’en rapprocher davantage. Cette distinction est documentée explicitement dans le présent mémoire plutôt que traitée comme équivalente, afin de ne pas biaiser la comparaison expérimentale lors des tests en H2. Enfin nous notons qu’ils enregistrent une dégradation de 2 à 3% par rapport aux modèles centralisés.

### III.5.4 – Federated learning et protection des données

Une limite structurelle du FL est que les mises à jour de modèle elles-mêmes peuvent révéler des informations sur les données d'entraînement locales, via des attaques par inférence ou reconstruction de gradient [64]. Abadi et al. (2016) [1] répondent à ce problème en introduisant la *differential privacy (DP)* appliquée à l'entraînement des réseaux de neurones. Le principe est d'arrêter le gradient de chaque exemple individuel et d'ajouter un bruit gaussien calibré :
$$\tilde{g}_t \leftarrow \frac{1}{L}\sum_i \bar{g}_t(x_i) + \mathcal{N}\!\left(0,\, \sigma^2 C^2 I\right)$$

où $C$ est le seuil d'écrêtage et $\sigma$ le niveau de bruit. Abadi et al. formalisent la garantie obtenue sous la notion de $(\varepsilon, \delta)$-differential privacy : deux bases de données adjacentes (différant d'un seul exemple) produisent des distributions d'outputs statistiquement indiscernables à un facteur $e^{\varepsilon}$ près, avec probabilité $1-\delta$.
L’introduction du bruit DP représente un compromis inévitable : plus la garantie de confidentialité est forte (ε faible), plus la performance du modèle se dégrade. Dans le contexte du FL pour la fraude financière, ce compromis peut être coûteux : les signaux de fraude sont rares et précis, et le DP peut masquer exactement les patterns discriminants que le modèle cherche à apprendre. Néanmoins les auteurs mettent en avant qu’ils n’ont pas eu de perte significative de performance.

*Voir annexe (VII.3) pour le détail des algorithmes testés ici.*

## III.6 – Gestion du déséquilibre de classe

### III.6.1 – Un déséquilibre structurel inversé

Contrairement aux problèmes classiques de détection de fraude où les cas frauduleux représentent une infime minorité des transactions, le dataset FNC-RF présente une structure inversée : l'ensemble des entrées correspond à des IBANs déjà signalés comme suspects. La tâche est donc une classification binaire entre fraudeurs confirmés (récidivistes ou cas avérés), qui représentent environ 23% de la base FNC-RF et 15,93% de notre base Fake-RF, et les faux positifs.

Ce déséquilibre modéré, reste néanmoins suffisant pour biaiser les modèles vers la classe majoritaire si non traité, au détriment du rappel sur les faux positifs, dont l'identification correcte constitue précisément l'un des objectifs centraux de ce travail.

### III.6.2 – Approches retenues

Face à ce déséquilibre, deux familles de méthodes sont généralement mobilisées dans la littérature : le rééchantillonnage (SMOTE, undersampling) et la pondération des classes. Ce travail retient exclusivement la pondération des classes (class weighting), pour les raisons suivantes.

Les algorithmes retenus (XGBoost, LightGBM, Random Forest) intègrent nativement des paramètres de pondération, les régularisateurs L1 et L2, et d’autres hyperparamètres tels que la profondeur des arbres ou le poids maximum assigné à chaque feuille, ainsi que la proportion des données d’entraînement qu’utilise chaque arbre, permettant de pénaliser davantage les erreurs sur la classe minoritaire sans altérer la distribution des données. Le rééchantillonnage synthétique (SMOTE) introduit un risque de *data leakage* en phase de validation croisée et génère des observations artificielles dont la validité est discutable sur des données réglementaires réelles. Vu la structure théorique de la base FNC-RF avec ses 23% de vrais positifs et celle de la base artificielle que nous allons créer, ce déséquilibre demeure modéré. 

Les poids sont calculés de manière inversement proportionnelle à la fréquence de chaque classe :

$$w_c = \frac{N}{k \cdot N_c}$$

## III.7 – Explicabilité

### III.7.1 – Enjeux de l'explicabilité

L’essor du ML dans les systèmes de décision financière soulève une tension fondamentale : plus un modèle est performant ou complexe, plus il tend à être opaque. Cette opacité est problématique dans un contexte réglementaire où les décisions automatisées affectant les droits des individus, comme le blocage d’un virement ou la signalisation d’un IBAN, doivent pouvoir être justifiées. L’Article 22 du RGPD encadre explicitement les décisions automatisées, tandis que l’AI Act européen impose des exigences de transparence accrues pour les systèmes à haut risque déployés dans le secteur financier. L’explicabilité conditionne donc sa légitimité opérationnelle et réglementaire. Pour un système de scoring d’IBAN tel que celui étudié dans ce mémoire, un PSP qui bloque un virement doit être en mesure d’en justifier la raison, non seulement pour satisfaire ses obligations légales, mais aussi pour réduire la contestation client et les coûts opérationnels associés aux faux positifs liés à un blocage. Dans le cadre de ce mémoire, elle est nécessaire pour vérifier l’hypothèse H3.

### III.7.2 – Revue globale des méthodes d'XAI

On distingue deux grandes caractéristiques des méthodes d’explicabilité : intrinsèque vs *post-hoc* et locale vs global.
En effet certains modèles sont interprétables par construction comme la régression logistique, arbres de décision ou encore les règles de scoring heuristiques. D'autres, comme XGBoost ou les réseaux de neurones, nécessitent des méthodes d'explicabilité appliquées après entraînement (post-hoc). Dans le cadre de ce mémoire, les modèles retenus étant majoritairement des arbres *boostés* et des architectures *deep-learning* (DL), les méthodes *post-hoc* sont privilégiées.
Ensuite l'explicabilité globale décrit le comportement général du modèle (quelles variables sont globalement importantes), tandis que l'explicabilité locale explique une prédiction individuelle, pourquoi ce compte a-t-il été considéré comme faux positif par exemple. Les deux niveaux sont pertinents : le niveau global pour valider la cohérence du modèle et soutenir l'argument de H3 dans le cadre de ce mémoire tandis que le niveau local pour la justification opérationnelle PSP. 

### III.7.3 – Cadre normatif de l'explicabilité

Au-delà du choix de la méthode, la qualité d'une explication doit elle-même être évaluée. Lago et al. (2025) [31] proposent un cadre structuré en quatre critères applicables à tout système XAI :

- consistance : l'explication doit rester stable sous des variations mineures de l'input ;
- plausibilité : l'explication doit s'aligner avec la connaissance experte du domaine ;
- fidélité : l'explication doit refléter fidèlement les mécanismes internes du modèle, pas seulement y ressembler ;
- utilité : l'explication doit être actionnable pour l'utilisateur final.
Bien que développé dans un contexte médical, ce cadre est directement transposable au scoring de fraude. Pour une entité régulée ayant recours a ces modèles une explication est consistante si elle est reproductible entre deux audits, plausible si les *features* importantes correspondent à des signaux de fraude reconnus (fréquence de signalement, délai de récidive), fidèle si elle reflète effectivement le comportement du modèle (même si cet axiome est discutable comme nous le verrons ci-dessous) et utile si elle permet au PSP de décider en connaissance de cause de bloquer ou libérer un virement.

### III.7.4 – Revue des méthodes considérées pour ce mémoire

#### III.7.4.1 – LIME

LIME (Local Interpretable Model-agnostic Explanations), introduit par Ribeiro, Singh & Guestrin (2016) [43], adopte une approche différente : pour chaque observation à expliquer, LIME génère un ensemble de perturbations locales, entraîne une régression linéaire sur ces perturbations, et extrait les coefficients de ce modèle comme proxy de l'importance des *features*.
L'avantage de LIME est sa généralité : étant agnostique au modèle, il s'applique à n'importe quelle architecture sans accès aux gradients ou à la structure interne du modèle. Il est également intuitif, produisant des explications sous forme de poids linéaires facilement compréhensibles. Néanmoins il a plusieurs limitations : l'algorithme est instable. En effet, l'échantillonnage aléatoire des perturbations implique que deux appels à LIME sur la même observation peuvent produire des explications différentes. Alvarez-Melis & Jaakkola (2018) [4] ont documenté cette instabilité et montré que LIME peut produire des explications contradictoires pour des observations similaires, ce qui constitue une propriété inacceptable dans un contexte de recherche mais aussi d’audit. Le modèle *surrogate* de régression linéaire est une approximation, et la qualité de cette approximation dépend fortement du voisinage choisi. Pour des modèles non linéaires complexes comme XGBoost sur des données financières, l'approximation linéaire locale peut être trompeuse. Enfin l’explication n’est pas cohérente globalement, les explications LIME ne s'agrègent pas de manière cohérente au niveau global. Il est donc difficile d'utiliser LIME pour tirer des conclusions sur l'importance relative des features à l'échelle du dataset. 

#### III.7.4.2 – SHAP

SHAP (*SHapley Additive exPlanations*), introduit par Lundberg & Lee (2017) [36], repose sur les valeurs de Shapley issues de la théorie des jeux. Pour chaque prédiction, SHAP décompose la contribution de chaque feature de manière additive :

$$f(x) = \phi_0 + \sum_i \phi_i$$

où $\phi_0$ est la prédiction moyenne du modèle et $\phi_i$ la contribution marginale de la feature $i$, calculée en moyennant son impact sur toutes les coalitions possibles de features.

Cette propriété additive confère à SHAP deux avantages décisifs sur les méthodes concurrentes : Elle est déterministe, pour un modèle et une observation donnés, les valeurs SHAP sont uniques et reproductibles. Contrairement à LIME, qui repose sur un échantillonnage aléatoire de perturbations locales, SHAP produit des explications stables d'une exécution à l'autre. Dans un contexte d'audit réglementaire, cette reproductibilité est essentielle. De plus, la somme des contributions SHAP est exactement égale à la différence entre la prédiction et la valeur de base du modèle. Cette propriété de complétude garantit qu'aucune contribution n'est arbitrairement ignorée ou surestimée. Enfin SHAP satisfait trois propriétés formelles importantes : efficacité (somme des contributions = prédiction nette), symétrie (les caractéristiques identiques reçoivent des contributions identiques), et nullité (une variable sans impact reçoit une contribution nulle). Ces propriétés font de SHAP la méthode théoriquement la plus fondée pour l'attribution *post-hoc* d’importance. 

#### III.7.4.3 – Limites

SHAP présente tout de même des limites importantes. Slack et al. (2020) [48] démontrent en effet que ces limites concernent notamment l'auditabilité et la dissimulation des biais. Les méthodes *post-hoc* basées sur des perturbations, dont SHAP, sont vulnérables à des attaques adversariales ciblées. En construisant un classifieur, un acteur malveillant peut concevoir un modèle dont les prédictions sur les données réelles restent biaisées (par exemple discriminantes sur un attribut protégé), mais dont les explications SHAP générées sur des données perturbées paraissent parfaitement innocentes. Cette propriété exploite le fait que les perturbations utilisées par SHAP pour estimer les contributions sont souvent hors distribution. Dans les expériences menées sur des jeux de données de récidive criminelle (COMPAS) et de scoring de crédit, les auteurs montrent qu'un classifieur discriminant basé uniquement sur la race peut voir son biais entièrement masqué par SHAP dans 84% des cas. LIME s'avère encore plus vulnérable, le biais étant masqué dans 100% des cas sur le même dataset. De plus, cette méthode est coûteuse en termes de calcul ; ainsi, pour H3, nous utiliserons seulement XGBoost et LightGBM pour tester cette hypothèse, ces modèles nous permettant d'attribuer une place aux variables inter-institutionnelles dans les prédictions de ces modèles.

# IV – Partie 3 : Méthodologie

Les hypothèses H1-H3 sont formulées dans une optique d’application à la base FNC-RF. En l’absence d’accès à cette dernière avant l’échéance de rédaction, elles seront donc testées sur la base proxy Fake-RF dont nous allons expliquer la création en détail dans la section suivante, avec les adaptations suivantes : H1 et H2 testées telles quelles et H3 traitée en version exploratoire, ses variables inter-PSP étant approximées par des variables inter-banques dérivées du dataset IBM Transactions for Anti Money Laundering [3]. Les résultats sont interprétés à la lumière de cette contrainte, discutée en section IV. Enfin nous utiliserons une classification binaire (0,1) pour respectivement faux positif et vrais positifs, en effet la base AML d’IBM ne permet pas de créer une stratification en trois classes comme nous l'avions voulu originellement.

## IV.8 – Données

### IV.8.1 – Fake-FNCRF & IBM Transactions for Anti Money Laundering 

Les données sont issues de la base de données IBM AML créée en 2022 [3]. Cette base a été construite grâce au générateur AMLworld, développé conjointement par IBM Research et l'ETH Zurich pour créer des jeux de données de transactions financières synthétiques et réalistes destinés au développement et au benchmarking de modèles de lutte contre le blanchiment d'argent (AML). Ce choix méthodologique répond à une contrainte structurelle du domaine : les données financières réelles permettant d'entraîner des modèles de détection de blanchiment sont généralement indisponibles, et les générateurs synthétiques précédents présentaient des lacunes significatives, notamment l'absence de modélisation multi-institutionnelle réaliste. 

L’idée de base d’AML World est de créer une simulation multi-agent, avec un monde financier virtuel composé de particuliers, d’entreprises et de banques en interaction, mêlant activités légitimes et criminelles. Le modèle sous-jacent ne repose pas sur l’anonymisation ou l’obfuscation de données réelles, mais sur des individus virtuels qui recréent des distributions statistiques et des schémas observés. Ce monde présente des agents à la fois bienveillants et malveillants, ces derniers se livrant à des activités criminelles qui nécessitent un blanchiment des fonds illicites obtenus. Le simulateur représente le flux classique du blanchiment d’argent, placement, *layering* et intégration dans l’économie légale. Le générateur crée ainsi des traces complètes de blanchiments basés sur huit typologies de blanchiment courantes (voir Figure 1) en propageant une étiquette de blanchiment le long des chaînes de transactions afin de fournir un label de vérité-terrain complet, approche également utilisée dans des simulateurs précédents comme *AML Sim* [49]. 

![*Flux de blanchiment, source [4]*](patterns.png)

Le générateur a ainsi permis de générer 6 jeux de données différents. Ces six variantes se déclinent selon deux axes : un taux d'illicéité (HI, taux élevé ; LI, taux faible) et une taille (Small, Medium, Large), avec des volumes allant de 515 088 comptes et 5 078 345 transactions pour HI-Small à 2 116 168 comptes et 179 702 229 transactions pour HI-Large.
Chaque transaction est décrite par : un horodatage, l'identifiant de la banque et du compte émetteurs, l'identifiant de la banque et du compte récepteurs, le montant reçu et sa devise, le montant payé et sa devise, le format de paiement, ainsi qu'un label binaire indiquant s'il s'agit ou non d'une transaction de blanchiment. C'est la présence conjointe des identifiants de banque émettrice et réceptrice qui permet, dans le cadre de ce mémoire, la segmentation par institution nécessaire à la construction de Fake-RF. Ces données ont été utilisées extensivement : le papier a été cité 235 fois selon arXiv.

### IV.8.2 – Construction de Fake-RF

Confrontés à la lenteur institutionnelle de l'accès à FNC-RF, et disposant d'un jeu de données synthétique riche en structure multi-institutionnelle (IBM AML), nous avons conçu l'algorithme suivant afin de construire une base approximant le processus réglementaire visé. Nous partons du principe que chaque banque est tenue de déclarer les transactions et comptes qu'elle considère impliqués dans du blanchiment d'argent, et que cette déclaration résulte d'un dispositif de contrôle interne combinant modèle de scoring et investigation aboutissant à la déclaration des cas de blanchiment avérés confirmés après investigation, ainsi que d'une proportion de faux positifs détectés par les contrôles internes mais infirmés a posteriori.

#### IV.8.2.1 – Modélisation par banque

Les transactions sont regroupées par banque émettrice (*From Bank*). Seules les banques comptant au moins 30 cas de blanchiment labellisés sont retenues, ce seuil garantissant un volume minimal pour l'entraînement d'un modèle par institution.

Pour chaque compte, un ensemble de variables est calculé de façon strictement causale (uniquement à partir de l'historique disponible au moment de la transaction, afin d'éviter toute fuite d'information) : délai depuis la transaction précédente, incohérence de devise entre émission et réception, indicateur de transfert intra-bancaire ou vers soi-même, montant transformé (log, indicateur de montant rond), variables temporelles (heure, jour de la semaine, transaction hors heures ouvrées), ainsi que la diversité cumulative de contreparties, banques et formats de paiement rencontrés jusqu'à la transaction courante. À ces variables de compte s'ajoutent des statistiques agrégées par banque émettrice et par banque réceptrice (devise dominante, nombre de comptes et d'événements distincts), ainsi que des indicateurs de connectivité du compte dans le graphe de transactions (*fan-in*, *fan-out*, ratio des deux par exemple).

Afin d'éviter toute fuite d'information temporelle et de reproduire un scénario réaliste de détection (entraînement sur l'historique, détection sur les transactions futures), la séparation entraînement/test est effectuée par quantile temporel : les 80 % des transactions les plus anciennes constituent l'ensemble d'entraînement, les 20 % les plus récentes l'ensemble de test.

Pour chaque banque retenue, un modèle de régression logistique (pondération de classe équilibrée) est entraîné sur les features décrites ci-dessus. Le déséquilibre extrême de la classe positive (blanchiment) rend un seuil de décision standard (0,5) inopérant. Le seuil est donc calibré via la courbe précision-rappel sur l'ensemble d'entraînement, en ciblant une précision de 0,08, choix arbitré pour garantir un volume exploitable de faux positifs sans générer un excès de bruit, avec une valeur de repli à 0,15 en cas de cible non atteinte. On a testé ce seuil sur le dataset Medium HI puis l’avons étendu aux Large HI, avec ses 180 M de transactions 2.1 M de comptes bancaires et un taux de fraude de 1/807. 

#### IV.8.2.2 – Extraction des classes et agrégation

Sur l'ensemble de test de chaque banque, l'ensemble des cas de blanchiment avérés est conservé et étiqueté Fraudeur ; les transactions légitimes classées positives par le modèle au seuil calibré sont étiquetées Faux Positif. Les sous-ensembles Fraudeur et Faux Positif de chaque banque, associés à leur identifiant d'institution, sont concaténés pour former la base Fake-RF, utilisée pour l'ensemble des expériences présentées en Partie IV.

### IV.8.3 – Caractéristiques statistiques de la base

Après traitement par le procédé décrit ci-dessus, on a extrait une base constituée de 1636 banques, avec des taux de vrais positifs variés. Le tableau ci-dessous en présente un extrait.

| Code de la Banque | Faux Positif | Fraudeur | Ratio Faux Positif / Fraudeur |
|---|---|---|---|
| 138848 | 891 | 7 | 127.285714 |
| 76077 | 317 | 3 | 105.666667 |
| 63576 | 187 | 2 | 93.500000 |
| 211154 | 1441 | 17 | 84.764706 |
| 39382 | 324 | 4 | 81.000000 |
| 125437 | 235 | 3 | 78.333333 |
| 147527 | 221 | 3 | 73.666667 |
| 18526 | 1529 | 21 | 72.809524 |
| 214853 | 1310 | 18 | 72.777778 |
| 40692 | 280 | 4 | 70.000000 |
| 130596 | 276 | 4 | 69.000000 |
| 247191 | 318 | 5 | 63.600000 |
| 44188 | 179 | 3 | 59.666667 |
| 196461 | 848 | 15 | 56.533333 |
| 34562 | 1053 | 19 | 55.421053 |

 *Tableau 1 – Taux de faux positifs des 50 premières banques de la base de données et de fraudeurs par banque dans Fake-RF*

On crée d’abord un dataset d'entraînement et de validation. Comme nos données sont temporellement sensibles nous avons crée le protocole suivant.
 
Compte tenu de la nature intrinsèquement temporelle des transactions, une validation en k-fold avec un k-fold de 5, d’environ 15 jours chacun, s'avère méthodologiquement nécessaire. Afin de préserver l’intégrité causale de l’expérimentation, nous adoptons un protocole de validation croisée par fenêtre glissante. Les caractéristiques de chaque segment sont les suivantes : 

| Segment | Vrais positifs en % *Train* | Vrais positifs en % Val |
|---|---|---|
| 1 | 0.0642 | 0.1048 |
| 2 | 0.0769 | 0.8850 |
| 3 | 0.0897 | 0.9482 |
| 4 | 0.0940 | 0.9482 |
| 5 | 0.0952 | 0.9174 |

*Tableau 2 – Caractéristiques des segments de validation croisée par fenêtre glissante*


On observe une croissance des taux de fraude sur les fins de segments, ce qui est un biais connu de notre base de données. Les auteurs de [3] expliquent ce biais par le fait que plus de transactions sont flaggées *post hoc* : "Note that the "Date Range" provided is "primary" period of transaction activity. In the discussion Marco Pianta observed that there are some transactions after the specified date period, and that those transactions are all laundering. Please see the response to Marco for a fuller description of this situation and how to deal with it. We thank Marco for raising this issue.". D'où la nécessité de recourir à une validation en k-fold, renforcée quant à la robustesse par le fait que, comme présenté en introduction, les motifs de fraude sont changeants.

Pour une proportion finale de Fraude pour Train et val de :

| | Vrais Positifs | Faux Positifs |
|---|---|---|
| Train | 0.0854 | 0.9145 |
| Val | 0.1593 | 0.8404 |

*Tableau 4 – Proportion finale de fraude en Train et Validation*

Les données de validation sont donc structurellement proche de la base FNC-RF réelle avec ses 23% de vrais positifs estimés. 

Ce protocole segmenter le continuum temporel des données en n segments successifs. À chaque itération, le modèle est entraîné sur une fenêtre historique croissante et évalué sur une fenêtre de validation future immédiate. Pour garantir une indépendance stricte entre les ensembles d'apprentissage et de test et ainsi éviter toute contamination par des dépendances temporelles immédiates, nous introduisons un intervalle de garde d'une heure entre la fin de la période d'entraînement et le début de la période de validation.

Nous avons ensuite mené des tests exploratoires sur la base afin de voir si les structures temporelles, comme, par exemple arranger les données en matrice d’événement faisais une différence, c’est-à-dire n_transactions × (n features) par compte avec comme objectif de prédire si le dernier événement est une fraude. Cela ne c'est pas avéré probant et a donc justifié notre choix d’exclure certaines architectures comme LSTM [27] ou CNN de notre mémoire. Nos modèles utilisent donc les données suivant un vecteur par transaction composée de n variables. Voir détails en annexe (VII.2).

## IV.9 – Structure expérimentale

Ce chapitre détaille le protocole expérimental mis en œuvre pour chacune des trois hypothèses, sur la base de Fake-RF (cf. IV.8.2). Chaque section précise le modèle retenu, la procédure d'évaluation, et les limites propres au protocole.

### IV.9.1 – H1 : Robustesse de la stratification

Objectif. Vérifier si une classification supervisée entraînée sur Fake-RF parvient à discriminer de manière robuste la classe Fraudeur de la classe Faux Positif.

#### IV.9.1.1 – Protocole

Un modèle est entraîné sur l’ensemble agrégé de Fake-RF (toutes banques confondues), avec optimisation des hyperparamètres grâce à une optimisation bayésienne validée par inter-validation k-fold à 3 afin de s'assurer de la présence de vrais positifs dans tous les folds de test et d'entraînement. L’algorithme d’optimisation bayésienne cherche ainsi à minimiser l’AP score dans l’espace des hyperparamètres, avec une heure par modèle. Le meilleur modèle est ensuite évalué et comparé.

Les détails des *features* créées sont décrits en H3 et en annexe. On procède de manière classique a un *scaling* et à un encoding des features (voir annexe VII.2.1) : on centre et réduit les variables numériques et on encode les variables catégoriques.

#### IV.9.1.2 – Critère de validation de H1

H1 est considérée comme supportée si le modèle calibré (*tuned*) permet de distinguer de manière robuste les FP des cas de fraude, et surpasse un modèle de base comme la régression logistique. Une limite intrinsèque est que le signal mesuré est en partie circulaire : les faux positifs proviennent eux-mêmes d'un modèle de régression logistique par banque (IV.8.2.1). Un modèle H1 de même famille (régression logistique) risque donc de sur-performer artificiellement en ré-apprenant la frontière de décision qui a servi à générer les labels. Ce biais est discuté en IV.4 et motive le choix, pour H1, de tester différents modèles afin de vérifier que le signal se généralise au-delà des régressions logistiques utilisées pour créer le dataset.

### IV.9.2 – H2 : Centralisation vs. Federated Learning

Objectif. Comparer la performance d'un modèle entraîné sur l'ensemble centralisé de Fake-RF à celle de modèles entraînés selon un schéma d'apprentissage fédéré, où chaque banque constitue un client.

#### IV.9.2.1 – Protocole

Le schéma fédéré est approximé par FedAvg, FedAdam, FedAdam + DP et Fed-XGBoost, chaque banque de Fake-RF étant traitée comme un client local disposant de ses propres données. À chaque round, les clients entraînent localement puis transmettent leurs mises à jour, agrégées par le serveur selon la règle d'Adam côté serveur. Cette simulation reste simplifiée : elle ne modélise ni la latence réseau ni les contraintes de communication réelles. Néanmoins elle intègre bien les contraintes liées à la DP. Les catégories sont encodées au niveau de la base globale (i.e., serveur), mais les données numériques sont normalisées par banques. On segmente ensuite la base par banques et implémente les algorithmes décrits en III.5.

Plus précisément nous ménerons l'experience avec : Un MLP pour FedAdam, FedAvg et FedAdam avec DP, exactement la même architecture que pour H1 afin de s’assurer d’une comparaison juste des modèles. Ainsi qu'avec l'XGBoost couplé a un Meta-Learner Linéaire avec une projection dans l’espace des deux classes pour l’implémentation de la méthodologie de Zhang et al. (2024) [53]. Contrairement a eux nous n'avons pas utilisé un CNN. Celuis-ci ajoutais un coût de calcul élevé pour des résultats similaires après ablation sur une sous partie du dataset. Cette expérience se réalise sur les 1636 banques et leurs transactions, et dû à la nature de l’agrégation, elle est relativement plus lente que les entraînements sur l’ensemble de la base. On note que dû aux échantillons plus petits, nous entraînons des arbres sensiblement plus petits par client que sur la base centralisée, et non tunés, avec un seul round de boosting par souci de ressources. Afin de tester leur algorithme de manière plus robuste, nous testerons la procédure suivante:

le même algorithme qu’avec XGBoost mais avec un LightGBM d’un seul round, tuné ainsi qu’avec 100 rounds au lieu de 50, et un *Meta-Learner* (voir annexe pour détails en VII.1.2), afin d’avoir une comparaison plus juste pour le Fed-XGBoost et les méthodes de *boosting* fédérées.

#### IV.9.2.2 – Critère de validation

H2 est supportée si le modèle centralisé surpasse, sur les mêmes métriques qu'en 9.1, le meilleur modèle fédéré obtenu après convergence des algorithmes choisis. On note que nous utiliserons exactement la même architecture pour le MLP que pour le modèle centralisé afin de garantir une comparaison juste entre les modèles. 

Limite. Le nombre de rounds de communication et la convergence de FedAdam sont bornés par les ressources de calcul disponibles, particulièrement pour Fed-XGBoost où nous utilisons des modèles de seulement un round de boosting, ce qui limite fortement la qualité des prédictions par client. Une non-convergence du modèle fédéré ne doit pas être interprétée à tort comme une infériorité structurelle du FL, mais documentée comme une limite computationnelle du protocole expérimental, distinction discutée explicitement en IV.4. De plus ces modèles ne bénéficient pas des mêmes *features *inter-institutionnelles*. On applique à ce niveau le même protocole que lors de la construction de la base. D'où la nécessité de tester H3.

### IV.9.3 – H3 : Valeur prédictive des signaux inter-institutionnels

#### IV.9.3.1 – Objectif

Tester si les variables inter-PSP (nombre de banques signalantes, délai inter-signalements, statistiques par banque déclarante) sont des prédicteurs significatifs de récidive, et si leur reproduction est équivalente sous schéma centralisé et fédéré.

#### IV.9.3.2 – Protocole

Les variables inter-institutionnelles sont construites à partir des identifiants de banque disponibles dans Fake-RF. On compare la performance des différents modèles sur la base avec ou sans variables inter-institutionnelle. L'importance prédictive est évaluée par SHAP. Et on mène une comparaison de ces variables en situation de FL et centralisée. Voir annexe pour les détails des variables (VII.3).

On note une limite structurelle et parallèle avec FNC-RF. Fake-RF n'est constituée que des sous-ensembles Fraude et Faux Positif extraits du jeu de test de chaque banque (IV.8.2.2), et non de l'historique complet des comptes. Il est donc impossible de calculer des motifs de connectivité (*fan-in/fan-out*, chaînes de transactions) sur l'intégralité du graphe transactionnel sous-jacent. Seules les transactions retenues dans l'extraction sont visibles. Cette limite n'est pas un simple artefact de notre protocole : elle reproduit fidèlement une contrainte réelle de FNC-RF elle-même, qui ne centralise que les cas déclarés par les PSP et non l'ensemble des transactions du système de paiement.

#### IV.9.3.3 – Critère de validation

H3 est partiellement supportée si les variables inter-institutionnelles ressortent comme prédicteurs significatifs sous le modèle centralisé, et que leur reconstruction s'avère dégradée ou impossible sous le schéma fédéré simulé en 9.2 (les mises à jour locales par banques ne peuvent, par construction, accéder aux identifiants de banques tierces). Cette hypothèse est traitée en version exploratoire, sa validation reposant sur une base proxy et non sur la base réglementaire cible.

\newpage

# V – Partie 4 : Résultats

## V.10 – H1 : Performance de la base centralisée

Après avoir tuné chaque architecture sur la base centralisée pendant une heure, nous avons les résultats suivants :


| Modèle | Threshold | Précision sur VP | Recall sur VP | F1 sur VP | Précision globale | ROC-AUC |
|---|---|---|---|---|---|---|
| MLP | 0.50 | 0.65 | 0.76 | 0.70 | 0.86 | 0.907 |
| MLP | 0.836 (tuned) | 0.75 | 0.78 | 0.76 | 0.91 | 0.919 |
| Random Forest | 0.50 | 0.62 | 0.79 | 0.70 | 0.85 | 0.897 |
| LightGBM | 0.50 | 0.83 | 0.80 | 0.81 | 0.92 | 0.944 |
| XGBoost | 0.50 | 0.74 | 0.73 | 0.74 | 0.89 | 0.903 |
| Reg-Logistique | 50 | 0.25 | 0.98 | 0.4 | 0.36 | 0.871 |

*Tableau 5 – Performance des architectures sur la base centralisée (H1)*


Les résultats obtenus permettent de valider l'hypothèse H1 de manière convaincante. L'ensemble des modèles testés, FNN, Random Forest, LightGBM et XGBoost, atteignent un ROC-AUC compris entre 0.897 et 0.944 et battent la régresison logistique. Ce qui indique une capacité de discrimination nettement supérieure au hasard entre la classe Fraudeur et la classe Faux Positif. La convergence de ces performances à travers des architectures aussi différentes qu'un réseau de neurones et des méthodes d'ensemble arborescentes renforce la robustesse de ce constat : le signal discriminant présent dans les données stratifiées via Fake-RF n'est pas un artefact propre à un modèle particulier, mais reflète une séparabilité réelle entre les deux classes. Sur le plan opérationnel, LightGBM se distingue nettement des autres modèles, avec le meilleur ROC-AUC (0.944) et le meilleur compromis precision/recall dès le seuil par défaut (F1 = 0.81, precision = 0.83, recall = 0.80), tandis que XGBoost, bien que de la même famille de boosting, reste proche des performances du Random Forest (F1 = 0.74). Le MLP, quant à lui, nécessite un ajustement du seuil de décision pour atteindre un niveau de performance comparable ou supérieur à XGBoost, ce qui suggère un pouvoir discriminant correct mais une calibration native moins optimale. La précision relativement modérée observée pour la classe Fraudeur pour la majorité des modèles (0.62 à 0.75, hors LightGBM) indique toutefois que la séparation entre les deux classes, bien que robuste statistiquement, demeure imparfaite en pratique pour ces modèles, laissant subsister une zone de confusion résiduelle entre fraudeurs et faux positifs. On note que la surperformance du LightGBM peut être due au fait qu'étant très rapide, il a pu être tuné plus efficacement par l'algorithme BO durant les 1h de calcul attribuées, et donc bénéficier de meilleurs hyperparamètres, une des forces de cette architecture.

H1 ayant été établie, on peut maintenant passer à H2.

## V.11 – H2 : Performance du Federated Learning simulée

Après avoir implémenté le protocole présenté précédentement, on obtient les résultats suivants : 


| Modèle | Threshold | Précision sur VP | Recall sur VP | F1 sur VP | Accuracy | ROC-AUC |
|---|---|---|---|---|---|---|
| MLP (centralisée + Full) | 0.50 | 0.65 | 0.76 | 0.70 | 0.86 | 0.911 |
| LightGBM (centralisée + Full) | 0.50 | 0.83 | 0.80 | 0.81 | 0.92 | 0.944 |
| FedAvg | 0.50 | 0.26 | 0.69 | 0.38 | 0.76 | 0.807 |
| FedAdam | 0.50 | 0.23 | 0.88 | 0.36 | 0.68 | 0.896 |
| FedAdam + DP | 0.50 | 0.23 | 0.88 | 0.36 | 0.68 | 0.893 |
| Fed-XGBoost + Meta-Learner | 0.50 | 0.52 | 0.44 | 0.48 | 0.90 | 0.830 |
| Fed-LightGBM + Meta-Learner | 0.50 | 0.68 | 0.16 | 0.26 | 0.90| 0.789 |

*Tableau 6 – Performance des méthodes fédérées comparées aux modèles centralisés (H2)*

En comparant d’abord le MLP centralisé avec accès aux variables inter-institutionnelles aux variantes fédérées reposant sur la même architecture (FedAvg, FedAdam, FedAdam+DP), on observe une dégradation systématique du F1 : le MLP centralisé atteint 0.70, contre 0.38 pour FedAvg et seulement 0.36 pour FedAdam et FedAdam+DP. FedAvg conserve *l’accuracy* la plus proche du centralisé (0.76 vs 0.86) mais au prix d’un recall limité (0.69) et d’une précision faible (0.26). FedAdam, à l’inverse, obtient un ROC-AUC proche du centralisé (0.896 vs 0.907) et un recall supérieur (0.88 vs 0.76), mais au prix d’une précision très dégradée (0.23) et d’une *accuracy* en retrait (0.68). L’ajout de la DP (FedAdam+DP) ne modifie pas cette tendance et confirme la stabilité du compromis et de l’algorithme proposé par Abadi et al. (2016) [1]. Ainsi, à architecture égale, le MLP centralisé conserve un meilleur équilibre précision/rappel (F1 le plus élevé) et une meilleure *accuracy* que toutes les variantes fédérées, même si FedAdam le dépasse sur le rappel seul. On précise que tous les modèles ont été évalués avec un seuil de décision à 0.5.

En élargissant la comparaison au meilleur modèle centralisé avec accès aux variables inter-institutionnelles, toutes architectures confondues (LightGBM, F1 = 0.81, ROC-AUC = 0.944), l’écart avec le meilleur modèle fédéré (Fed-XGBoost + Meta-Learner, F1 = 0.48, ROC-AUC = 0.830) se creuse encore davantage, confirmant que la contrainte de fédération dégrade sensiblement la performance par rapport à la base centralisée, même sans les problématiques de communication avec les approches de DL et *boosting*. H2 est donc supportée. La question est maintenant d’assigner cette dégradation de performance au FL ou à la perte des signaux inter-institutionnels. Nous nuançons cependant la performance du LightGBM dans la comparaison, en effet il a bénéficié des meilleurs hyperparamètres.

## V.12 – H3 : Impact des features centralisées et inter-PSP

Après avoir construit les *features*, détaillées en annexe, ainsi que les basse de données avec à la fois les signaux inter-institutionnels et les statistiques calculées sur l'ensemble de la base, on obtient les résultats suivants : 

| Modèle | Threshold | Accuracy | ROC-AUC | Précision sur VP | Recall sur VP | F1 sur VP |
|---|---|---|---|---|---|---|
| XGBoost Full | 0.5 | 0.88 | 0.90 | 0.74 | 0.73 | 0.74 |
| XGBoost Reduced | 0.5 | 0.87 | 0.91 | 0.67 | 0.78 | 0.72 |
| LightGBM Full | 0.5 | 0.92 | 0.94 | 0.83 | 0.80 | 0.81 |
| LightGBM Reduced | 0.5 | 0.86 | 0.91 | 0.81 | 0.64 | 0.71 |
| MLP Full | 0.83 *(tuned)* | 0.91 | 0.91 | 0.75 | 0.78 | 0.76 |
| MLP Reduced | 0.83 *(tuned)* | 0.89 | 0.90 | 0.75 | 0.75 | 0.75 |

*Tableau 7 – Performance des modèles XGBoost, LightGBM et MLP (variantes Full et Reduced) sur H3.*

On note ici que « Reduced » s'applique aux modèles ne bénéficiant pas des variables inter-institutionnelles. On observe un delta clair pour le LightGBM entre le Full et le Reduced, de +3% sur le ROC-AUC du complet par rapport à l'incomplet, +2% sur la précision sur la fraude, +6% sur la précision globale, +16% sur le rappel et donc +10% sur le F1. Néanmoins les gains sont moins importants pour le XGBoost, potentiellement car il est moins bien tuné. On a -1% sur le ROC-AUC, +7% sur la détection de fraude, +1% sur la précision globale. Le rappel est plus mauvais (-5%) et donc un gain modeste sur le F1 de +2%. Tandis que pour les MLP tuned la perte de performance est minime.
Les résultats semblent tendre en partie vers H3, même si les gains semblent en partie liés aux choix architecturaux et au tuning pour la performance. La figure 1 ci-dessous illustre ce résultat :

![*Courbes ROC des modèles XGBoost et LightGBM (Full vs Reduced) sur H3*](ROC-H3.png)

\newpage

Or la seconde partie de H3 est l’utilisation des features inter-institutionnelles par les modèles. On s’intéresse aux variables explicatives agrégées par SHAP par les modèles complets. On obtient pour le XGBoost :

| Top 15 des variables explicatives| Variables inter-institutionnelle |
|---|---|
| Fan-out associé au compte | Faux |
| Banque déclarante / à l'origine de la transaction | Faux |
| Une autre banque a déclaré ce compte comme frauduleux de la transaction | Vrai |
| Montant de la transaction (log, normalisé) | Faux |
| Format de paiement | Faux |
| Banque destinataire de la transaction | Faux |
| Taux de fraude du corridor | Vrai |
| Nombre de comptes déclarés dans la base pour la banque destinataire | Vrai |
| Nombre de devises utilisées par le compte dans la base | Faux |
| Delta temporel (log, normalisé) depuis la dernière transaction dans la base | Faux |
| Nombre de transactions déclarées pour la banque déclarante | Vrai |
| Taux de fan-out associé au compte | Vrai |
| Nombre de transactions déclarées pour la banque destinataire | Vrai |
| Devise de la transaction | Faux |
| Heure de la transaction dans la journée | Faux |

*Tableau 8 – Top 15 des variables explicatives (SHAP) du modèle XGBoost Full sur H3.*

On a donc 6 des *features* les plus importantes qui sont issues de variables inter-institutionnelles.

\newpage

Pour le LightGBM, le tableau est le suivant : 

| Top 15 des variables explicatives | Variables inter-institutionnelle |
|---|---|
| Fan-out associé au compte | Faux |
| Montant de la transaction (log, normalisé) | Faux |
| Format de paiement | Faux |
| Banque déclarante / à l'origine de la transaction | Faux |
| Taux de fraude du corridor | Vrai |
| Banque destinataire de la transaction | Faux |
| Nombre de devises utilisées par le compte dans la base | Vrai |
| Une autre banque a déclaré ce compte comme frauduleux de la transaction | Vrai |
| Delta temporel (log, normalisé) depuis la dernière transaction dans la base | Vrai |
| Jour de la semaine de la transaction | Faux |
| Taux de fan-out associé au compte | Vrai |
| Heure de la transaction dans la journée | Faux |
| Nombre de comptes déclarés dans la base pour la banque destinataire | Vrai |
| Nombre de transactions déclarées pour la banque destinataire | Vrai |
| Nombre de comptes déclarés pour la banque déclarante | Vrai |

*Tableau 9 – Top 15 des variables explicatives (SHAP) du modèle LightGBM Full sur H3.*

8 des variables les plus importantes pour LightGBM sont inter-institutionnelles. Ce modèle étant le plus performant, cela valide notre hypothèse sur les variables inter-institutionnelles. Enfin, lorsqu’on approfondit l'analyse de certaines de ces variables inter-institutionnelles, on trouve une explication à l’importance des features inter-institutionnelles. Premièrement, les historiques des banques destinataires, c'est-à-dire du blanchiment commis vers celles-ci, ne sont visibles par client que dans 85,67% des cas. Deuxièmement, la variable de déclaration d’un compte avéré frauduleux en t-1 (9,34% des cas) a un taux de fraude 6 fois supérieur aux autres, ce qui explique son importance dans les variables explicatives des modèles. On considère donc que les features inter-institutionnelles ont un intérêt pour la classification du risque de récidive et que H3 est partiellement supportée mais que les gains sont hétérogènes selon les architectures.

\newpage

# VI – Conclusion

Ce mémoire interrogeait la capacité d'un scoring de risque à stratifier fraudeurs récidivistes et faux positifs au sein d'une base réglementaire centralisée, et cherchait à établir si les signaux inter-institutionnels qu'elle rend accessibles justifient empiriquement une centralisation plutôt qu'un apprentissage fédéré. Les trois hypothèses sont soutenues, mais à des degrés de robustesse inégaux. 

La capacité des algorithmes de ML à stratifier les vrais et faux positifs est validée (H1) sans ambiguïté : toutes les architectures testées (MLP, Random Forest, XGBoost, LightGBM) discriminent les deux classes avec un ROC-AUC supérieur à 0,89. Cette convergence entre familles de modèles aussi différentes rend le résultat robuste, peu dépendant du choix architectural. La dégradation des performances des algorithmes de FL est également supportée (H2): à architecture égale, chaque variante fédérée (FedAvg, FedAdam, FedAdam+DP, Fed-XGBoost) dégrade systématiquement par rapport au centralisé équivalent. Une réserve s'impose toutefois : l'essentiel de l'écart, lorsqu'on compare les meilleurs modèles toutes architectures confondues, vient de LightGBM, qui a bénéficié d'un tuning bayésien plus efficace du fait de sa rapidité d'entraînement. Cependant, dans le contexte de faibles ressources de calcul (puce M3 avec 16 Go de RAM), la performance de cet algorithme et l'importance du tuning bayésien constituent des résultats à ne pas négliger. Une partie de l'écart centralisé/fédéré est donc probablement imputable à ce déséquilibre de tuning, pas seulement à la contrainte de fédération. La comparaison à architecture égale (MLP centralisé vs. FedAvg/FedAdam), néanmoins moins exposée à ce biais, est également en faveur de H2, même si la perte de performance reste moindre. L'importance des features inter-institutionnelles n'est que partiellement supportée (H3). Les variables inter-institutionnelles comptent bien parmi les prédicteurs les plus importants selon SHAP (6 des 15 premières pour XGBoost, 8 pour LightGBM), et leur retrait coûte jusqu'à 10 points de F1 pour LightGBM. Mais le gain est hétérogène : marginal pour XGBoost, quasi nul pour le MLP tuné. L'apport mesuré dépend donc autant de la capacité du modèle à exploiter ce signal que du signal lui-même. Nous n'avons par ailleurs pas exploité les représentations en graphe de transactions pour construire ces features, ce qui laisse ouverte la possibilité qu'une partie du signal inter-institutionnel reste sous-captée plutôt qu'absente. Pris ensemble, ces résultats suggèrent que la perte de performance du FL n'est pas qu'un artefact de convergence ou de calcul, mais reflète en partie une perte structurelle de signal. On ne peut cependant pas, avec ce protocole, isoler entièrement cette part de celle due au tuning inégal et à la spécification incomplète des variables inter-PSP. Nous montrons que la centralisation permet d'atteindre une performance que le FL, tel qu'implémenté ici, n'atteint pas. 

Trois limites pèsent sur ces conclusions, toutes dans le même sens. Fake-RF reste une base proxy : ses labels sont issus d'un scoring et d'une investigation simulés, sur une base synthétique (IBM AML World) calibrée sur des distributions statistiques des années 2020. Le protocole fédéré est simplifié et borné par les ressources de calcul, pour les rounds comme pour le tuning. Les variables inter-institutionnelles ne sont reconstruites que partiellement, à l'image des limites réelles d'accès au FNC-RF. Ces trois limites tendent à sous-estimer ce que le FL pourrait atteindre avec plus de moyens, et à surestimer la marge réelle de la centralisation. Les résultats sont donc une indication de direction, pas une mesure d'écart généralisable. Sur le plan régulatoire, ce travail apporte un argument empirique, mais partiel, en faveur de la centralisation des signalements de fraude : le partage de signaux inter-institutionnels a un intérêt réel pour la stratification du risque, à mettre en balance avec son coût en matière de vie privée; un arbitrage que ce mémoire documente sans le trancher. 

Cela n'implique pas que la centralisation intégrale soit la seule voie : des schémas intermédiaires (par exemple FL enrichi de statistiques inter-PSP anonymisées) pourraient capter une partie du signal de H3 sans reproduire toutes les contraintes du FNC-RF. Nous n'avons pas testé ces alternatives ; leur évaluation est une extension naturelle de ce travail. Enfin, les avancées récentes de modèles fondationnels et transformers pour la détection de fraude, chez Revolut, NVIDIA avec PRAGMA [54], posent la question de leur place dans les stratégies régulatoires futures ; on note cependant qu'ils surperforment sur toutes les tâches sauf l'AML à ce jour.

\newpage 

# VII – Annexe

## VII.1 – Hyperparamètres

### VII.1.1 – H1

Les algorithmes de *BO* pour 100 itérations utilisaient, respectivement par modèle, l'espace de recherche et les meilleurs paramètres suivants :

**XGBoost**

| Hyperparamètre | Espace de recherche | Description | Meilleure valeur |
|---|---|---|---|
| learning_rate | Real(0.01, 0.5, log-uniform) | | 0.2034 |
| max_depth | Integer(2, 10) | Profondeur maximale | 10 |
| min_child_weight | Integer(1, 5) | Plus bas = meilleur pour les classes minoritaires | 5 |
| subsample | Real(0.5, 1.0) | | 0.8371 |
| colsample_bytree | Real(0.3, 1.0) | Fraction des données d'entraînement utilisée pour construire chaque arbre | 1.0 |
| reg_lambda | Real(1e-9, 100, log-uniform) | | 1e-09 |
| reg_alpha | Real(1e-9, 100, log-uniform) | | 1e-09 |
| n_estimators | Integer(50, 2000) | Nombre d'arbres | 2000 |
| max_delta_step | Integer(3, 7) | Poids maximal assigné à chaque feuille d'arbre, utile pour les classes déséquilibrées et la régression | 3 |
\newpage
**Random Forest**

| Hyperparamètre | Espace de recherche | Description | Meilleure valeur |
|---|---|---|---|
| n_estimators | Integer(50, 2000) | Nombre d'arbres | 858 |
| max_depth | Integer(2, 30) | Profondeur maximale de l'arbre | 28 |
| min_samples_leaf | Integer(1, 10) | Plus bas pour mieux détecter les minorités | 7 |
| class_weight | Categorical(['balanced', 'balanced_subsample']) | Meilleurs résultats testés pour la moyenne macro | 'balanced_subsample' |
| max_features | Categorical(['sqrt', 'log2']) | | 'sqrt' |
| bootstrap | Categorical([True, False]) | | True |

**LightGBM**

| Hyperparamètre | Espace de recherche | Meilleure valeur |
|---|---|---|
| num_leaves | Integer(15, 1000) | 1000 |
| max_depth | Integer(3, 12) | 12 |
| learning_rate | Real(1e-3, 0.3, log-uniform) | 0.2166 |
| n_estimators | Integer(50, 1000) | 1000 |
| min_child_samples | Integer(5, 100) | 74 |
| subsample | Real(0.5, 1.0) | 0.7598 |
| colsample_bytree | Real(0.5, 1.0) | 0.5450 |
| reg_alpha | Real(1e-8, 10.0, log-uniform) | 1.1515 |
| reg_lambda | Real(1e-8, 10.0, log-uniform) | 0.2049 |

**FNN**

| Hyperparamètre | Espace de recherche | Meilleure valeur |
|---|---|---|
| lr | Real(1e-4, 1e-3, log-uniform) | 9e-4 |
| module__dim | Categorical([64, 128, 256]) | 128 |
| module__emb_dim | Categorical([8, 16]) | 8 |
| module__mlp_mult | — | 2 |
| module__depth | — | 1 |
| max_epochs | — | 50 |
\newpage
### VII.1.2 – H2

**FNN**

| Hyperparamètre | Valeur |
|---|---|
| module__dim | 128 |
| module__mlp_mult | 2 |
| module__depth | 1 |
| module__emb_dim | 8 |
| max_rounds | 50 |
| lr | 1e-2 (FedAdam et FedAdam+DP, et le méta-learner FNN pour Fed-XGBoost) ; 1e-3 (FedAvg) |

**XGBoost (un round par arbre)**

| Hyperparamètre | Valeur |
|---|---|
| objective | multi:softprob |
| num_class | num_class |
| max_depth | 4 |
| eta | 0.3 |

**Espace de recherche de BO pour les LightGBM**

| Hyperparamètre | Espace de recherche |
|---|---|
| num_leaves | Integer(7, 255) |
| max_depth | Integer(3, 10) |
| learning_rate | Real(1e-2, 0.3, log-uniform) |
| min_child_samples | Integer(5, 50) |
| subsample | Real(0.6, 1.0) |
| colsample_bytree | Real(0.6, 1.0) |
| reg_alpha | Real(1e-6, 10.0, log-uniform) |
| reg_lambda | Real(1e-6, 10.0, log-uniform) |

Le budget de recherche est plafonné à 45 secondes de calcul par client, avec 3 arbres. La métrique optimisée est l'*average precision* (aire sous la courbe précision-rappel) sur les probabilités prédites. En cas d'échec de la recherche, les valeurs de repli sont : *num_leaves=31*, *max_depth=6*, *learning_rate=0.1*, *n_estimators=3*, *min_child_samples=10*.

Le nombre d'itérations de BO est adapté à la taille du client; moins d'itérations pour les banques avec beaucoup de transactions, plus lentes à entraîner, afin de garder un temps de calcul par client comparable.

| Taille du client (nb. lignes) | Itérations BO |
|---|---|
| < 300 | 10 |
| 300 – 1 000 | 8 |
| 1 000 – 5 000 | 6 |
| 5 000 – 20 000 | 4 |
| ≥ 20 000 | 3 |

Après tests, un taux d'apprentissage significativement plus élevé a produit de meilleurs résultats. Les algorithmes ne convergeant pas après 50 epochs/rounds autrement, on a aussi augmenté les epochs à 100 pour donner plus de chances au *bagging* fédéré.

## VII.2 – Features

### VII.2.1 – Pour la construction de la base et H2

Ce script construit la base Fake-RF/FNC-RF elle-même (labellisation proxy par régression logistique par banque) et fournit le jeu de variables utilisé tel quel pour les expériences H2. Il ne contient aucune variable inter-institutionnelle causale, comme détaillé dans la méthodologie. Chaque banque n'a accès qu'à ses propres transactions au moment de la construction, conformément à la contrainte de silo simulée en Federated Learning.

| Feature | Description courte |
|---|---|
| nb.currency | Rang cumulatif de la transaction dans la séquence du compte |
| delta.t | Temps écoulé (secondes, log-transformé) depuis la transaction précédente du compte |
| currency.mismatch | Indicateur binaire : devise de réception différente de la devise d'envoi |
| is.self.transfer | Indicateur binaire : compte émetteur identique au compte récepteur |
| is.intra.bank | Indicateur binaire : banque émettrice identique à la banque destinataire |
| log.amount | Transformation log(1+montant) du montant payé |
| is.round.amount | Indicateur binaire : montant multiple de 100 |
| hour.of.day | Heure de la transaction |
| day.of.week | Jour de la semaine de la transaction |
| is.off.hours | Indicateur binaire : transaction entre 0h et 5h |
| nb.distinct.to.bank_cum | Nombre cumulatif de banques destinataires distinctes utilisées par le compte |
| nb.distinct.from.bank_cum | Nombre cumulatif de banques émettrices distinctes utilisées par le compte |
| nb.distinct.payfmt_cum | Nombre cumulatif de formats de paiement distincts utilisés par le compte |
| top.1.holder.RC / top.1.holder.SC | Devise de réception / d'envoi la plus fréquente chez les contreparties destinataires vues par la banque |
| nb.iban.holder, nb.events.holder | Nombre de comptes / d'événements distincts associés aux contreparties destinataires |
| top.1.declaring.RC / top.1.declaring.SC | Devise de réception / d'envoi la plus fréquente chez les contreparties émettrices vues par la banque |
| nb.iban.declaring, nb.events.declaring | Nombre de comptes / d'événements distincts associés aux contreparties émettrices |
| fan.out | Nombre de comptes destinataires distincts atteints par le compte |
| fan.in | Nombre de comptes émetteurs distincts ayant envoyé vers le compte |
| fan.ratio | Ratio fan.in / (fan.out + 1) |

Ce jeu de variables reprend les signaux transactionnels standards du scoring de fraude/AML (montant, délai, horaire, format, degré fan-in/fan-out), calculables par un PSP à partir de ses seules données. Il sert de base commune à H2, sans aucun signal partagé entre banques, afin d'isoler l'effet de la fédération elle-même sur la performance.

### VII.2.2 – Pour H1 et H3

On part de la base Fake-RF déjà construite et y ajoute les variables inter-institutionnelles causales testées en H3, en plus de reprendre les variables transactionnelles de VII.2.1. Toutes les variables sont calculées de manière strictement causale (historique antérieur à la transaction courante uniquement) puis décalées d'un lag par compte, afin que le modèle prédise le label courant à partir de l'état connu à la transaction précédente.

**Variables inter-institutionnelles ajoutées (déclarant / détenteur / corridor)**

| Feature | Description courte |
|---|---|
| declaring.fraud_rate | Taux de fraude historique cumulé des transactions émises par la banque déclarante |
| declaring.fp_rate | Taux de faux positifs historique cumulé de la banque déclarante |
| declaring.nb.prior.txn | Nombre de transactions antérieures déjà traitées par la banque déclarante |
| declaring.has.history | Indicateur binaire : historique non-nul pour la banque déclarante |
| holding.fraud_rate | Taux de fraude historique cumulé des transactions reçues par la banque détentrice |
| holding.fp_rate | Taux de faux positifs historique cumulé de la banque détentrice |
| holding.nb.prior.txn | Nombre de transactions antérieures déjà reçues par la banque détentrice |
| holding.has.history | Indicateur binaire : historique non-nul pour la banque détentrice |
| corridor.fraud_rate | Taux de fraude historique cumulé du corridor (banque émettrice a banque destinataire) |
| corridor.nb.prior.txn | Nombre de transactions antérieures observées sur ce corridor |

Les variables transactionnelles/intra-compte (nb.currency, delta.t, currency.mismatch, is.self.transfer, is.intra.bank, log.amount, hour.of.day, day.of.week, is.off.hours, nb.distinct._cum, top.1.holder/declaring., nb.iban., nb.events., fan.out, fan.in, fan.ratio) sont identiques à VII.2.1.

Les variables `.fraud_rate`, `.fp_rate` et `corridor.*` sont directement liées à H3 et la base FNCRF : elles n'existent que si un signal de fraude est partagé entre PSP, contrairement au reste du jeu de variables déjà disponible en H1/H2. Leur construction causale (exclusion de la transaction courante, agrégation strictement antérieure) évite toute fuite d'information et reste fidèle au scénario FNC-RF où une institution ne dispose que des signalements déjà remontés par les autres PSP.

![*Corrélation des variables globales sur Fake-RF*](correlation_global_features.png)

![*Corrélation des variables locales utilisées lors de la création de la base Fake-RF*](correlation_local_features.png)

## VII.3 – Détails des Algorithmes

Les quatre algorithmes ci-dessous partagent la même boucle client : à chaque round, chaque banque de Fake-RF (traitée comme un client local) initialise son modèle avec l'état global courant, effectue 3 pas de descente de gradient stochastique (SGD, lr = 1e-2 pour FedAdam/FedAdam+DP et le méta-learner Fed-XGBoost/Fed-LightGBM, 1e-2 pour FedAvg) sur sa perte locale avec comme objectif le *cross-entropy loss*, pondérée par les poids de classe calculés une fois sur l'ensemble de la base d'entrainement et partagés par tous les clients; on part du principe qu'ils partagent les taux de Fraude puis renvoie son état local au serveur. Le serveur agrège ensuite les mises à jour selon la règle propre à chaque variante, sur 50 rounds afin de rester cohérent avec les MLP centralisés (100 pour le méta-learner Fed-LightGBM au vu des coûts de calcul inférieurs). L'évaluation se fait à chaque round sur un jeu de test tenu à l'écart, encodé avec les mêmes pipelines globaux que les clients, comme décrit dans la méthodologie.

### VII.3.1 – FedAvg

Implémentation directe de McMahan et al. (2017) [38] : le nouvel état global est la moyenne des états locaux, pondérée par la taille de chaque client:

$$w_{t+1} = \sum_{k=1}^{K} \frac{n_k}{n} w_{t+1}^{k}$$

où $n_k$ est le nombre d'observations du client $k$ et $n = \sum_k n_k$. Aucune mémoire de moment n'est conservée côté serveur d'un round à l'autre ; c'est la variante la plus simple testée, servant de référence basse pour les autres schémas d'agrégation.

### VII.3.2 – FedAdam

Variante de FedOpt (Reddi et al., 2020) [42] où le serveur traite la moyenne des deltas clients comme un pseudo-gradient et applique une mise à jour Adam plutôt qu'une simple moyenne. Pour chaque client $k$, le delta $\Delta_k = w_{t+1}^{k} - w_t$ est d'abord *clippé* en norme L2, maximum 1.0, afin d'assurer la stabilité et la cohérence avec FedAdam + DP, puis moyenné (non pondéré par $n_k$):

$$\Delta_t = \frac{1}{K}\sum_{k=1}^{K}\text{clip}(\Delta_k, C)$$

Le serveur met ensuite à jour ses moments d'ordre 1 et 2 ($\beta_1$ = 0.9, $\beta_2$ = 0.99, $\epsilon$ = 1e-2 — des valeurs classiques recommandées par la littérature) ainsi que l'état global, exactement comme Adam le ferait sur un gradient :

$$m_t = \beta_1 m_{t-1} + (1-\beta_1)\Delta_t \qquad v_t = \beta_2 v_{t-1} + (1-\beta_2)\Delta_t^2$$
$$w_{t+1} = w_t + \eta \frac{\hat m_t}{\sqrt{\hat v_t} + \epsilon}$$

avec $\hat m_t$, $\hat v_t$ corrigés du biais d'initialisation comme dans Adam standard. 

### VII.3.3 – FedAdam + DP

Extension de FedAdam intégrant un mécanisme de confidentialité différentielle inspiré de Abadi et al. (2016) [1] côté agrégation serveur (FedAdamDPServer). Deux mécanismes s'ajoutent à FedAdam :

- *Clipping* de la norme L2 de chaque delta client à un seuil 1.0, comme pour FedAdam, ce qui borne la sensibilité de la contribution de chaque client à l'agrégat ;
- bruit gaussien ajouté à la somme des deltas clippés avant division par le nombre de clients, avec un multiplicateur de bruit $\sigma$ = 0.1, comme préconisé par [1] :

$$\tilde\Delta_t = \frac{1}{K}\left(\sum_{k=1}^{K}\text{clip}(\Delta_k, C) + \mathcal{N}(0, (\sigma C)^2 I)\right)$$

$\tilde\Delta_t$ remplace ensuite $\Delta_t$ dans la même mise à jour Adam. Le *clipping* et le bruit calibré sur ce seuil constituent le mécanisme standard de garantie de confidentialité différentielle au niveau de l'agrégation serveur DP-FedAdam, au prix d'un bruit supplémentaire avec un effet moindre dans nos observations.

### VII.3.4 – Fed-XGBoost / Fed-LightGBM (méthodologie Zhang et al., 2024)

Pour les modèles à base d'arbres, FedAvg/FedAdam ne s'appliquent pas directement (III.5.3) : on suit à la place le schéma en deux phases de Zhang et al. (2024) [53].

Phase 1 — agrégation par empilement d'arbres (*tree bagging*).

Chaque client entraîne localement un *booster* indépendant sur ses propres données, 1-3 arbres par client. Les arbres de tous les clients sont ensuite concaténés (et renumérotés) en un unique booster global vérifié pour produire des prédictions en score brut strictement égales à la somme des boosters individuels.

Phase 2 — méta-apprentissage fédéré.
 
Pour chaque observation, on extrait la contribution marginale (poids de la feuille atteinte) de chaque arbre du booster global gelé, formant un vecteur de *features* de dimension *rounds* d'agrégation x classes (1,0) pour le XGBoost ou le nombre total d'arbres (LightGBM) x classes (0,1). Un *Méta-Learner* linéaire est ensuite entraîné sur ces variables par FedAdam. 50 rounds pour le XGBoost et 100 rounds pour le LightGBM, et une learning rate de 1e-3 et 1e-2 respectivement. Chaque client entraînant localement le méta-learner sur ses propres marges d'arbres puis transmettant son état au serveur pour agrégation.

## VII.4 – Packages et stack utilisée

L'ensemble des expériences (construction de la base, entraînement centralisé H1, simulation du Federated Learning H2, ablation des variables inter-institutionnelles H3, interprétabilité) repose sur la *stack* Python suivante :

**Manipulation de données**

| Package | Usage |
|---|---|
| pandas | Manipulation tabulaire, feature engineering, fenêtres temporelles glissantes |
| numpy | Calcul vectoriel, transformations |
| scipy | Matrices creuses pour les features encodées |
\newpage
**Machine learning classique et pipelines**

| Package | Usage |
|---|---|
| scikit-learn | Pipelines de prétraitement, modèles de bagging et régression, métriques, pondération de classe |
| xgboost | XGBoost |
| lightgbm | LightGBM |
| statsmodels | Diagnostic de colinéarité |

**Optimisation d'hyperparamètres**

| Package | Usage |
|---|---|
| scikit-optimize (skopt) | Optimisation Bayésienne des hyperparamètres |

**Deep learning et Federated Learning**

| Package | Usage |
|---|---|
| torch (PyTorch) | Réseaux de neurones (FNN/MLP centralisé et fédéré), optimisation SGD/Adam côté client et serveur, tenseurs des features par client |
| skorch | Interface scikit-learn pour les modules PyTorch, callbacks d'entraînement utilisés pour le tuning du MLP en H1 |

Les algorithmes de *Federated Learning* eux-mêmes (FedAvg, FedAdam, FedAdam+DP, agrégation d'arbres Fed-XGBoost/Fed-LightGBM) sont implémentés directement en PyTorch/NumPy (cf. VII.3) plutôt que via un *framework* FL dédié (type *Flower* ou *TensorFlow Federated*), afin de garder un contrôle explicite sur l'agrégation, du *clipping* et le bruit différentiel.

**Interprétabilité**

| Package | Usage |
|---|---|
| shap | Valeurs de Shapley pour l'analyse d'importance des variables |
\newpage

**Utilitaires**

| Package | Usage |
|---|---|
| joblib | Sérialisation des modèles, pipelines et boosters globaux fusionnés |
| matplotlib | Visualisation  |
| tqdm | Suivi de progression des boucles d'entraînement|
| re, gc, glob, os | Utilitaires système |

## VII.5 – Glossaire

AI Act - Règlement européen sur l'intelligence artificielle

AML - Anti Money Laundering (lutte anti-blanchiment)

APP - Authorised Push Payment (fraude par virement autorisé)

AUROC / ROC-AUC - Area Under the Receiver Operating Characteristic curve

BCE / ECB - Banque Centrale Européenne / European Central Bank

BO - Optimisation Bayésienne (Bayesian Optimization)

CNIL - Commission Nationale de l'Informatique et des Libertés

CNN - Convolutional Neural Network

DB - Data base (base de données)

DL - Deep Learning

DP - Differential privacy (confidentialité différentielle)

EBA - European Banking Authority

EPC - European Payments Council (Conseil Européen des Paiements)

FedAvg - Federated Averaging

FedAdam / FedOpt - Federated Adam / Federated Optimization

FL - Federated Learning (apprentissage fédéré)

FN - Faux Négatif

FNC-RF - Fichier National Centralisé des Comptes de Paiement Frauduleux

FNN - Feed-Forward Neural Network

FP - Faux Positif

FPR - False Positive Rate

FRIDA - Fraud Information Distribution Arrangement

IBAN - International Bank Account Number

IPR - Instant Payments Regulation

LIME - Local Interpretable Model-agnostic Explanations

LSTM - Long Short Term Memory

ML - Machine Learning

MLP - Multi Layer Perceptron

OSMP - Observatoire de la Sécurité des Moyens de Paiement

PSD / PSD2 / PSD3 - Payment Services Directive 

PSP - Prestataire de Service de Paiement

PSR - Payment Services Regulation

RGPD / GDPR - Règlement Général sur la Protection des Données

SCA - Strong Customer Authentication 

SEPA - Single Euro Payments Area

SGD - Stochastic Gradient Descent

SHAP - SHapley Additive exPlanations

SMOTE - Synthetic Minority Oversampling Technique

TPR - True Positive Rate

UUID - Universally Unique Identifier

VN - Vrai Négatif

VOP / VoP - Verification of Payee

VP - Vrai Positif

XAI - eXplainable Artificial Intelligence

XGBoost - Extreme Gradient Boosting

# VIII – Sources

[1] Abadi, M., Chu, A., Goodfellow, I., McMahan, H. B., Mironov, I., Talwar, K., & Zhang, L. (2016). Deep Learning with Differential Privacy. arXiv:1607.00133.

[2] Almalki, F., & Masud, M. (2025). Financial Fraud Detection Using Explainable AI and Stacking Ensemble Methods. arXiv:2505.10050.

[3] Altman, E., Blanuša, J., von Niederhäusern, L., Egressy, B., Anghel, A., & Atasu, K. (2023). Realistic Synthetic Financial Transactions for Anti-Money Laundering Models. *Advances in Neural Information Processing Systems 36 (NeurIPS 2023), Datasets and Benchmarks Track*. arXiv:2306.16424.

[4] Alvarez-Melis, D., & Jaakkola, T. S. (2018). On the Robustness of Interpretability Methods. *ICML 2018 Workshop on Human Interpretability in Machine Learning (WHI)*. arXiv:1806.08049.

[5] Amariles, D. R., Charlotin, D., & He-Guelton, L. (2026). AI Agents in Payments: Applications, Risks and Regulations. *European Journal of Risk Regulation*, published online 2026, 1-24. doi:10.1017/err.2026.10103

[6] Amed, S., Hang, C. Y., & Banerjee, S. (2025). PDx — Adaptive Credit Risk Forecasting Model in Digital Lending using Machine Learning Operations. arXiv:2512.22305.

[7] Aminian, G., Elliott, A., Li, T., Wong, T. C. H., Dehon, V. C., Szpruch, L., Maple, C., Read, C., Brown, M., Reinert, G., & Mamouei, M. (2025). FraudTransformer: Time-Aware GPT for Transaction Fraud Detection. Workshop paper, *ACM International Conference on AI in Finance (ICAIF '25)*, Singapore. arXiv:2509.23712.

[8] Assemblée nationale. *Proposition de loi n° 884 visant à renforcer la lutte contre la fraude bancaire*, exposé des motifs.

[9] Awosika, T., Shukla, R. M., & Pranggono, B. (2024). Transparency and Privacy: The Role of Explainable AI and Federated Learning in Financial Fraud Detection. *IEEE Access*, 12, 64551-64560. doi:10.1109/ACCESS.2024.3394528. arXiv:2312.13334.

[10] Ba, J. L., Kiros, J. R., & Hinton, G. E. (2016). Layer Normalization. arXiv:1607.06450.

[11] Banque de France — Observatoire de la Sécurité des Moyens de Paiement (OSMP). *Note statistiques de fraude du premier semestre 2025*, 27 janvier 2026.

[12] Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5-32.

[13] Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD '16)*, 785-794. arXiv:1603.02754.

[14] Parlement européen et Conseil de l'Union européenne. *Directive 2007/64/CE du 13 novembre 2007 concernant les services de paiement dans le marché intérieur (PSD)*, JO L 319 du 5.12.2007.

[15] Parlement européen et Conseil de l'Union européenne. *Directive (UE) 2015/2366 du 25 novembre 2015 concernant les services de paiement dans le marché intérieur (PSD2)*, JO L 337 du 23.12.2015, pp. 35-127.

[16] Parlement européen et Conseil de l'Union européenne. *Règlement (UE) 2024/886 du 13 mars 2024 relatif aux virements et aux prélèvements instantanés en euros (Instant Payments Regulation, IPR)*. Disponible sur : <https://www.ecb.europa.eu/paym/retail/instant_payments/html/instant_payments_regulation.en.html>

[17] Cover, T., & Hart, P. (1967). Nearest Neighbor Pattern Classification. *IEEE Transactions on Information Theory*, 13(1), 21-27.

[18] Cox, D. R. (1958). The Regression Analysis of Binary Sequences. *Journal of the Royal Statistical Society: Series B*, 20(2), 215-242.

[19] Durand, D. (1941). *Risk Elements in Consumer Installment Financing*. National Bureau of Economic Research.

[20] EBA (European Banking Authority) & ECB (European Central Bank). (2025). *2025 Report on Payment Fraud*, EBA/REP/2025/40, décembre 2025.

[21] EPC (European Payments Council). (2025). *Payment Threats and Fraud Trends Report 2025*.

[22] EPC (European Payments Council) / ABBL. *FRIDA: a future framework for fraud intelligence sharing in Europe*. Disponible sur : <https://www.abbl.lu/frida-a-future-framework-for-fraud-intelligence-sharing-in-europe/> et <https://www.europeanpaymentscouncil.eu/what-we-do/other-epc-activities/fraud-prevention-and-payment-security>

[23] EPC (European Payments Council). *SEPA Verification of Payee Scheme Rulebook*, version 1.0 (5 octobre 2025) et version 1.1 (mars 2026). Disponible sur : <https://www.europeanpaymentscouncil.eu/what-we-do/other-schemes/verification-payee>

[24] eucrim — The European Criminal Law Associations' Forum. *Europol report: criminal use of deepfake technology*. Disponible sur : <https://eucrim.eu/news/europol-report-criminal-use-of-deepfake-technology/>

[25] Fan, J., Shar, L. K., Zhang, R., Liu, Z., Yang, W., Niyato, D., Mao, B., & Lam, K.-Y. (2025). Deep Learning Approaches for Anti-Money Laundering on Mobile Transactions: Review, Framework, and Directions. arXiv:2503.10058.

[26] Fazel, R. E., Bakhtiary, A., & Bigdeli, S. A. (2026). Improving Credit Card Fraud Detection with an Optimized Explainable Boosting Machine. arXiv:2602.06955.

[27] Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory. *Neural Computation*, 9(8), 1735-1780.

[28] Kanagavelu, R., Nepal, M., Peiyan, N., Kangning, C., Jiming, X., Gao, F., Liu, Y., Rick, G. S. M., & Wei, Q. (2026). DPxFin: Adaptive Differential Privacy for Anti-Money Laundering Detection via Reputation-Weighted Federated Learning. arXiv:2603.19314.

[29] Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., & Liu, T.-Y. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree. *Advances in Neural Information Processing Systems*, 30.

[30] Koli, L., Kalra, S., Thakur, R., Saifi, A., & Singh, K. (2025). AI-Driven IRM: Transforming Insider Risk Management with Adaptive Scoring and LLM-Based Threat Detection. arXiv:2505.03796.

[31] Lago, M. A., et al. (2025). Evaluating Explainability: A Framework for Systematic Assessment and Reporting of Explainable AI Features in Medical Imaging. arXiv:2506.13917.

[32] Lee, T.-S., Chiu, C.-C., Lu, C.-J., & Chen, I.-F. (2002). Credit scoring using the hybrid neural discriminant technique. *Expert Systems with Applications*, 23(3), 245-254. doi:10.1016/S0957-4174(02)00044-1

[33] Li, X., Huang, K., Yang, W., Wang, S., & Zhang, Z. (2020). On the Convergence of FedAvg on Non-IID Data. *International Conference on Learning Representations (ICLR 2020)*. arXiv:1907.02189.

[34] Liu, J., Shang, F., Liu, H., Tian, Y., Liu, Y., Liu, J., Zhu, K., & Lin, Z. (2025). FedAdamW: A Communication-Efficient Optimizer with Convergence and Generalization Guarantees for Federated Large Models. *AAAI 2026*. arXiv:2510.27486.

[35] République française. *Loi n° 2025-1058 du 6 novembre 2025 visant à renforcer la lutte contre la fraude bancaire* (dite « Loi Labaronne »), *Journal officiel de la République française*.

[36] Lundberg, S. M., & Lee, S.-I. (2017). A Unified Approach to Interpreting Model Predictions. *Advances in Neural Information Processing Systems*, 30. arXiv:1705.07874.

[37] Makris, A., Dousis, C., Kritharakis, E., Bouras, S., & Tserpes, K. (2026). A Comparative Study of Federated Learning Aggregation Strategies under Homogeneous and Heterogeneous Data Distributions. arXiv:2605.11010.

[38] McMahan, H. B., Moore, E., Ramage, D., Hampson, S., & Arcas, B. A. y. (2017). Communication-Efficient Learning of Deep Networks from Decentralized Data. *Proceedings of the 20th International Conference on Artificial Intelligence and Statistics (AISTATS 2017)*.

[39] Padhi, I., Schiff, Y., Melnyk, I., Rigotti, M., Mroueh, Y., Dognin, P., Ross, J., Nair, R., & Altman, E. (2021). Tabular Transformers for Modeling Multivariate Time Series. *ICASSP 2021, IEEE*, 3565-3569. arXiv:2011.01843.

[40] Presse : *Irish Times* (29 avril 2026), *AML Intelligence* (avril 2026) et *TheJournal.ie* (octobre 2025). Reportages sur la vidéo deepfake usurpant l'identité du Tánaiste irlandais Simon Harris pour promouvoir un faux produit d'investissement.

[41] Prokhorenkova, L., Gusev, G., Vorobev, A., Dorogush, A. V., & Gulin, A. (2018). CatBoost: Unbiased Boosting with Categorical Features. *NeurIPS 2018*. arXiv:1706.09516.

[42] Reddi, S., Charles, Z., Zaheer, M., Garrett, Z., Rush, K., Konečný, J., Kumar, S., & McMahan, H. B. (2020). Adaptive Federated Optimization. arXiv:2003.00295 (FedOpt / FedAdam ; publié à ICLR 2021).

[43] Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). "Why Should I Trust You?": Explaining the Predictions of Any Classifier. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD '16)*, 1135-1144. arXiv:1602.04938.

[44] Rida, A. (2024). Machine and Deep Learning for Credit Scoring: A Compliant Approach. arXiv:2412.20225.

[45] Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). Learning representations by back-propagating errors. *Nature*, 323(6088), 533-536.

[46] Shazeer, N. (2020). GLU Variants Improve Transformer. arXiv:2002.05202.

[47] Sharma, M. A., Raj, B. R. G., Ramamurthy, B., & Bhaskar, R. H. (2022). Credit Card Fraud Detection Using Deep Learning Based on Auto-Encoder. *ITM Web of Conferences*, 50, 01001. doi:10.1051/itmconf/20225001001

[48] Slack, D., Hilgard, S., Jia, E., Singh, S., & Lakkaraju, H. (2020). Fooling LIME and SHAP: Adversarial Attacks on Post hoc Explanation Methods. *AAAI/ACM Conference on AI, Ethics, and Society (AIES 2020)*. arXiv:1911.02508.

[49] Suzumura, T., & Kanezashi, H. (2021). AMLSim: A multi-agent based simulator generating synthetic banking transaction data with known money laundering patterns. IBM Research. Disponible sur : <https://github.com/IBM/AMLSim>

[50] Tong, K., Han, Z., Shen, Y., Long, Y., & Wei, Y. (2024). An Integrated Machine Learning and Deep Learning Framework for Credit Card Approval Prediction. arXiv:2409.16676.

[51] Vimal, S., Kayathwal, K., Wadhwa, H., & Dhama, G. (2021). Application of Deep Reinforcement Learning to Payment Fraud. Presented at Marble-KDD '21, Singapore. arXiv:2112.04236.

[52] Yurdem, B., Kuzlu, M., Gullu, M. K., Catak, F. O., & Tabassum, M. (2024). Federated learning: Overview, strategies, applications, tools and future directions. *Heliyon*, 10(19), e38137. doi:10.1016/j.heliyon.2024.e38137

[53] Zhang, S., Tay, J., & Baiz, P. (2024). The Effects of Data Imbalance Under a Federated Learning Approach for Credit Risk Forecasting. arXiv:2401.07234.

[54] Ostroukhov, M., Mikhailov, R., Iashin, V., Sokolov, A., Akshonov, A., Protasov, V., Beloborodov, D., Mullin, V., Enzmann, R. Y., Kolovos, G., Renders, J., Nesterov, P., & Repushko, A. (2026). *PRAGMA: Revolut Foundation Model*. arXiv:2604.08649.

[55] European Banking Authority. (2022). *Discussion paper on the EBA’s preliminary observations on selected payment fraud data under PSD2, as reported by the industry for the years 2019 and 2020* (EBA/DP/2022/01). https://www.eba.europa.eu/sites/default/files/document_library/About%20Us/Annual%20Reports/2021/1035237/EBA%202021%20Annual%20Report.pdf

[56] Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning. MIT Press.

[57] Bergstra, J., & Bengio, Y. (2012). Random Search for Hyper-Parameter Optimization. Journal of Machine Learning Research, 13, 281-305.

[58] Snoek, J., Larochelle, H., & Adams, R. P. (2012). Practical Bayesian Optimization of Machine Learning Algorithms. Advances in Neural Information Processing Systems 25 (NeurIPS 2012). arXiv:1206.2944.

[59] Shahriari, B., Swersky, K., Wang, Z., Adams, R. P., & de Freitas, N. (2016). Taking the Human Out of the Loop: A Review of Bayesian Optimization. Proceedings of the IEEE, 104(1), 148-175.

[60] Bergstra, J., Yamins, D., & Cox, D. D. (2013). Making a Science of Model Search: Hyperparameter Optimization in Hundreds of Dimensions for Vision Architectures. Proceedings of the 30th International Conference on Machine Learning (ICML 2013).

[61] Head, T., et al. (2021). scikit-optimize: Sequential model-based optimization in Python (v0.9.0). Zenodo. doi:10.5281/zenodo.5574484.

[62] Robbins, H., & Monro, S. (1951). A Stochastic Approximation Method. The Annals of Mathematical Statistics, 22(3), 400-407.

[63] ACPR — Banque de France. (2026). Premiers bilans du Fichier National des Comptes Signalés (FNC-RF) présentés aux Rencontres Anti-Blanchiment du 16 juin 2026, rapporté par mind Fintech, 16 juin 2026.

[64] Zhu, L., Liu, Z., & Han, S. (2019). Deep Leakage from Gradients. Advances in Neural Information Processing Systems (NeurIPS 2019), 32.

[65] Kingma, D. P., & Ba, J. (2015). Adam: A Method for Stochastic Optimization. 3rd International Conference on Learning Representations (ICLR 2015).