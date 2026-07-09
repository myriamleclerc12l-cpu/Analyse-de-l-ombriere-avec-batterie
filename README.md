# Analyse-de-l-ombriere-avec-batterie

Dashboard Autoconsommation Photovoltaïque

Ce projet contient deux outils Streamlit conçus pour analyser les performances énergétiques de sites équipés de panneaux solaires, en utilisant des données réelles de consommation et de production.

 Contenu du projet

1. app.py (Dashboard Avancé)

Un outil complet d'aide à la décision pour le dimensionnement de systèmes solaires avec stockage batterie. Il propose 4 onglets :

Données (Raw) : Visualisation brute des entrées.

Analyse Temporelle : Graphiques interactifs de la courbe de charge et de la production.

Indicateurs (KPIs) : Calcul automatique du Taux d'Autoproduction (TAP) et du Taux d'Autoconsommation (TAC).

Analyse Annuelle (Stockage) : Simulation avancée de l'impact de la capacité de la batterie sur vos gains énergétiques, avec choix du niveau de charge initial (SoC).


 Utilisation

Les outils acceptent des fichiers CSV ou Excel (.xlsx, .xls).

Les colonnes attendues dans vos fichiers sont : Date et Valeur (W).

Les applications convertissent automatiquement les données de Watts en kW pour vos analyses.

 Contribution

Ce projet est conçu pour évoluer. Vous pouvez ajouter de nouvelles fonctionnalités de simulation tarifaire ou d'export de données en modifiant directement le code dans les fichiers sources
