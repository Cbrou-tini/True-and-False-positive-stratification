This project was conducted during my studies, you will find below is the abstract in English and French. 

- Scripts : contains experimental tests & correlation plot scripts.
- Dataset : dataset creation script with no DB due to size but the code to recreate it.
- Memoire in english and french in pdf refer to the french document if unvcertain as the english one is machine translated.

--- 

## Abstract 

The fight against payment fraud in Europe has long faced a structural constraint: banking secrecy prohibits institutions from sharing IBANs identified as fraudulent among themselves, allowing repeat fraudsters to operate across multiple institutions without being detected. The Labaronne law (2025) lifts this constraint in France by creating the FNC-RF, a national file centralizing reports of fraudulent accounts between PSPs. This dissertation examines the scope of this centralization: does a machine-learning risk scoring model applied to a database of this type enable a robust stratification of repeat fraudsters from false positives, and do the inter-institutional signals it makes accessible empirically justify regulatory centralization against the alternative structurally favored by the literature, federated learning (FL). In the absence of access to the real FNC-RF, a proxy database (Fake-RF) is constructed from the synthetic dataset created by IBM AML World [3], by simulating a per-bank reporting process combining a scoring model and investigation. Three hypotheses are tested: the robustness of stratification by supervised models (H1, logistic regression, Random Forest, XGBoost, LightGBM, MLP); the superiority of centralized training over simulated federated schemes (H2, FedAvg, FedAdam, FedAdam with differential privacy, Fed-XGBoost); and the predictive value of inter-institutional variables, assessed via SHAP (H3). The results support H1, H2, and H3, albeit with reservations. All architectures discriminate between true and false positives with a ROC-AUC of 0.89 or higher, with overall accuracy ranging between 0.74 and 0.92, and between 0.63 and 0.83 on true positives (H1). Each federated variant shows a performance degradation relative to the equivalent centralized model (H2). This degradation is attributable in part to inter-institutional variables, which rank among the most important predictors according to SHAP, with a measurable effect on performance (H3). These results, obtained on a proxy database and therefore to be interpreted with caution, suggest that the FL performance loss is not solely a matter of convergence constraints, but reflects a structural loss of signal—an empirical argument in favor of regulatory centralization of anti-fraud intelligence at the European scale.

## Results & Conclusion

This dissertation examined the capacity of a risk-scoring model to stratify repeat fraudsters and false positives within a centralized regulatory database, and sought to establish whether the inter-institutional signals it makes accessible empirically justify centralization over federated learning. All three hypotheses are supported, but to unequal degrees of robustness.

The ability of ML algorithms to stratify true and false positives is validated (H1) unambiguously: all the architectures tested (MLP, Random Forest, XGBoost, LightGBM) discriminate between the two classes with a ROC-AUC above 0.89. This convergence across such different model families makes the result robust, largely independent of the architectural choice. The degradation in performance of FL algorithms is also supported (H2): for equal architecture, each federated variant (FedAvg, FedAdam, FedAdam+DP, Fed-XGBoost) systematically degrades relative to its centralized equivalent. One caveat is nonetheless warranted: most of the gap, when comparing the best models across all architectures, comes from LightGBM, which benefited from more efficient Bayesian tuning owing to its fast training time. However, in the context of limited computing resources (an M3 chip with 16 GB of RAM), this algorithm's performance and the importance of Bayesian tuning are results not to be overlooked. Part of the centralized/federated gap is therefore likely attributable to this tuning imbalance, not solely to the federation constraint. The comparison at equal architecture (centralized MLP vs. FedAvg/FedAdam), which is less exposed to this bias, is nonetheless also in favor of H2, even though the performance loss is smaller. The importance of inter-institutional features is only partially supported (H3). The inter-institutional variables are indeed among the most important predictors according to SHAP (6 of the top 15 for XGBoost, 8 for LightGBM), and removing them costs up to 10 points of F1 for LightGBM. But the gain is heterogeneous: marginal for XGBoost, nearly nil for the tuned MLP. The measured contribution therefore depends as much on the model's ability to exploit this signal as on the signal itself. We did not, moreover, exploit transaction graph representations to construct these features, which leaves open the possibility that part of the inter-institutional signal remains under-captured rather than absent. Taken together, these results suggest that FL's performance loss is not merely an artifact of convergence or computation, but partly reflects a structural loss of signal. However, with this protocol we cannot fully isolate this component from that due to unequal tuning and the incomplete specification of inter-PSP variables. We show that centralization achieves a level of performance that FL, as implemented here, does not reach.

Three limitations weigh on these conclusions, all in the same direction. Fake-RF remains a proxy database: its labels stem from simulated scoring and investigation, on a synthetic database (IBM AML World) calibrated on statistical distributions from the 2020s. The federated protocol is simplified and bounded by computing resources, both for the rounds and for tuning. The inter-institutional variables are only partially reconstructed, mirroring the real access limitations of FNC-RF. These three limitations tend to underestimate what FL could achieve with more resources, and to overestimate the real margin of centralization. The results are therefore an indication of direction, not a generalizable measure of the gap. From a regulatory standpoint, this work provides an empirical, though partial, argument in favor of centralizing fraud reports: sharing inter-institutional signals has genuine value for risk stratification, to be weighed against its cost in terms of privacy — a trade-off that this dissertation documents without resolving.

This does not imply that full centralization is the only path forward: intermediate schemes (for example, FL enriched with anonymized inter-PSP statistics) could capture part of the H3 signal without reproducing all of FNC-RF's constraints. We did not test these alternatives; their evaluation is a natural extension of this work. Finally, recent advances in foundation models and transformers for fraud detection, at Revolut and NVIDIA with PRAGMA, raise the question of their place in future regulatory strategies; we note, however, that they outperform on all tasks except AML to date.

---

## Introduction 

La lutte contre la fraude aux paiements en Europe s'est longtemps heurtée à une contrainte structurelle : le secret bancaire interdit aux établissements de partager entre eux les IBAN identifiés comme frauduleux, permettant aux fraudeurs récidivistes d'opérer à travers plusieurs institutions sans être détectés. La loi Labaronne (2025) lève cette contrainte en France en créant le FNC-RF, un fichier national centralisant les signalements de comptes frauduleux entre PSP. Ce mémoire interroge la portée de cette centralisation : un scoring de risque par apprentissage automatique appliqué à une base de ce type permet-il de stratifier de manière robuste les fraudeurs récidivistes des faux positifs, et les signaux inter-institutionnels qu'elle rend accessibles justifient-ils empiriquement la centralisation réglementaire face à l'alternative structurellement privilégiée par la littérature, l'apprentissage fédéré (FL).
En l'absence d'accès au FNC-RF réel, une base proxy (Fake-RF) est construite à partir du jeu de données synthétique créé par IBM *AML world* [3], en simulant un processus de déclaration par banque combinant modèle de scoring et investigation. Trois hypothèses sont testées : la robustesse de la stratification par des modèles supervisés (H1, régression logistique, Random Forest, XGBoost, LightGBM, MLP) ; la supériorité d'un entraînement centralisé sur des schémas fédérés simulés (H2, FedAvg, FedAdam, FedAdam avec confidentialité différentielle, Fed-XGBoost); et la valeur prédictive des variables inter-institutionnelles, évaluée par SHAP (H3).
Les résultats supportent avec réserve H1, H2 et H3. L'ensemble des architectures discrimine les Vrai et faux positifs avec un ROC-AUC supérieur ou égal à 0,89, avec une précision globale comprise entre 0.74 et 0.92 et 0.63 et 0.83 sur Vrai-Positifs (H1). Chaque variante fédérée affiche une dégradation de performance par rapport au modèle centralisé équivalent (H2). L'attribution de cette dégradation est en partie due aux variables inter-institutionnelles, qui figurent parmi les prédicteurs les plus importants selon SHAP, avec un effet sur la performance (H3). Ces résultats, obtenus sur une base proxy et donc à interpréter avec prudence, suggèrent que la perte de performance du FL ne relève pas uniquement de contraintes de convergence, mais reflète une perte structurelle de signal, un argument empirique en faveur de la centralisation réglementaire de l'intelligence anti-fraude à l'échelle européenne.

## Résultats & Conclusions 


Ce mémoire interrogeait la capacité d'un scoring de risque à stratifier fraudeurs récidivistes et faux positifs au sein d'une base réglementaire centralisée, et cherchait à établir si les signaux inter-institutionnels qu'elle rend accessibles justifient empiriquement une centralisation plutôt qu'un apprentissage fédéré. Les trois hypothèses sont soutenues, mais à des degrés de robustesse inégaux. 

La capacité des algorithmes de ML à stratifier les vrais et faux positifs est validée (H1) sans ambiguïté : toutes les architectures testées (MLP, Random Forest, XGBoost, LightGBM) discriminent les deux classes avec un ROC-AUC supérieur à 0,89. Cette convergence entre familles de modèles aussi différentes rend le résultat robuste, peu dépendant du choix architectural. La dégradation des performances des algorithmes de FL est également supportée (H2): à architecture égale, chaque variante fédérée (FedAvg, FedAdam, FedAdam+DP, Fed-XGBoost) dégrade systématiquement par rapport au centralisé équivalent. Une réserve s'impose toutefois : l'essentiel de l'écart, lorsqu'on compare les meilleurs modèles toutes architectures confondues, vient de LightGBM, qui a bénéficié d'un tuning bayésien plus efficace du fait de sa rapidité d'entraînement. Cependant, dans le contexte de faibles ressources de calcul (puce M3 avec 16 Go de RAM), la performance de cet algorithme et l'importance du tuning bayésien constituent des résultats à ne pas négliger. Une partie de l'écart centralisé/fédéré est donc probablement imputable à ce déséquilibre de tuning, pas seulement à la contrainte de fédération. La comparaison à architecture égale (MLP centralisé vs. FedAvg/FedAdam), néanmoins moins exposée à ce biais, est également en faveur de H2, même si la perte de performance reste moindre. L'importance des features inter-institutionnelles n'est que partiellement supportée (H3). Les variables inter-institutionnelles comptent bien parmi les prédicteurs les plus importants selon SHAP (6 des 15 premières pour XGBoost, 8 pour LightGBM), et leur retrait coûte jusqu'à 10 points de F1 pour LightGBM. Mais le gain est hétérogène : marginal pour XGBoost, quasi nul pour le MLP tuné. L'apport mesuré dépend donc autant de la capacité du modèle à exploiter ce signal que du signal lui-même. Nous n'avons par ailleurs pas exploité les représentations en graphe de transactions pour construire ces features, ce qui laisse ouverte la possibilité qu'une partie du signal inter-institutionnel reste sous-captée plutôt qu'absente. Pris ensemble, ces résultats suggèrent que la perte de performance du FL n'est pas qu'un artefact de convergence ou de calcul, mais reflète en partie une perte structurelle de signal. On ne peut cependant pas, avec ce protocole, isoler entièrement cette part de celle due au tuning inégal et à la spécification incomplète des variables inter-PSP. Nous montrons que la centralisation permet d'atteindre une performance que le FL, tel qu'implémenté ici, n'atteint pas. 

Trois limites pèsent sur ces conclusions, toutes dans le même sens. Fake-RF reste une base proxy : ses labels sont issus d'un scoring et d'une investigation simulés, sur une base synthétique (IBM AML World) calibrée sur des distributions statistiques des années 2020. Le protocole fédéré est simplifié et borné par les ressources de calcul, pour les rounds comme pour le tuning. Les variables inter-institutionnelles ne sont reconstruites que partiellement, à l'image des limites réelles d'accès au FNC-RF. Ces trois limites tendent à sous-estimer ce que le FL pourrait atteindre avec plus de moyens, et à surestimer la marge réelle de la centralisation. Les résultats sont donc une indication de direction, pas une mesure d'écart généralisable. Sur le plan régulatoire, ce travail apporte un argument empirique, mais partiel, en faveur de la centralisation des signalements de fraude : le partage de signaux inter-institutionnels a un intérêt réel pour la stratification du risque, à mettre en balance avec son coût en matière de vie privée; un arbitrage que ce mémoire documente sans le trancher. 

Cela n'implique pas que la centralisation intégrale soit la seule voie : des schémas intermédiaires (par exemple FL enrichi de statistiques inter-PSP anonymisées) pourraient capter une partie du signal de H3 sans reproduire toutes les contraintes du FNC-RF. Nous n'avons pas testé ces alternatives ; leur évaluation est une extension naturelle de ce travail. Enfin, les avancées récentes de modèles fondationnels et transformers pour la détection de fraude, chez Revolut, NVIDIA avec PRAGMA [54], posent la question de leur place dans les stratégies régulatoires futures ; on note cependant qu'ils surperforment sur toutes les tâches sauf l'AML à ce jour.

---

## Sources :


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
