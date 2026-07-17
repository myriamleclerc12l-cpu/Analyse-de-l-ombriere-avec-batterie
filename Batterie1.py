# -*- coding: utf-8 -*-
"""
Dashboard Autoconsommation - Version Finale
Inclut les 3 onglets initiaux + l'onglet 4 Analyse Annuelle avec sélection d'hypothèses
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io # Ajout de la bibliothèque io pour lire le format spécial Enedis

# ==========================================
# 0. PARAMÈTRES ET CONFIGURATION
# ==========================================
st.set_page_config(page_title="Dashboard Autoconsommation", layout="wide")

st.markdown("""
    <style>
    [data-testid="stFileUploadDropzone"] > div > div > span {display: none;}
    [data-testid="stFileUploadDropzone"] > div > div::before {
        content: 'Glissez et déposez vos fichiers ici';
        font-weight: 500;
        display: block;
    }
    [data-testid="stFileUploadDropzone"] > div > div > small {display: none;}
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 1. FONCTIONS DE CHARGEMENT & SIMULATION
# ==========================================

@st.cache_data
def charger_donnees_reelles(file_conso, file_prod, file_bornes):
    if file_conso.name.endswith('.csv'):
        df_c = pd.read_csv(file_conso)
    else:
        df_c = pd.read_excel(file_conso)
        
    df_c.columns = ["date", "conso_W"] 
    df_c["date"] = pd.to_datetime(df_c["date"], utc=True)
    df_c.set_index("date", inplace=True)
    df_c["conso_W"] = pd.to_numeric(df_c["conso_W"], errors='coerce').fillna(0)
    df_c["conso_kW"] = df_c["conso_W"] / 1000.0
    
    if file_prod.name.endswith('.csv'):
        df_p = pd.read_csv(file_prod)
    else:
        df_p = pd.read_excel(file_prod)
        
    df_p.columns = ["date", "prod_W"]
    df_p["date"] = pd.to_datetime(df_p["date"], utc=True)
    df_p.set_index("date", inplace=True)
    df_p["prod_W"] = pd.to_numeric(df_p["prod_W"], errors='coerce').fillna(0)
    df_p["prod_kW"] = df_p["prod_W"] / 1000.0
    
    df = pd.concat([df_c["conso_kW"], df_p["prod_kW"]], axis=1)
    df = df.fillna(0)
    df.sort_index(inplace=True)
    
    # --- Ajout des bornes de recharge (Format Enedis) ---
    if file_bornes is not None:
       try:
        content = file_bornes.getvalue().decode('utf-8', errors='replace')
        
        skip_lines = 0
        for i, line in enumerate(content.split('\n')):
            if 'horodate' in line.lower() or 'date' in line.lower():
                skip_lines = i
                break
                
        df_b = pd.read_csv(io.StringIO(content), sep=';', skiprows=skip_lines)
        
        horodate_cols = [col for col in df_b.columns if 'horodate' in col.lower()]
        date_cols = [col for col in df_b.columns if 'date' in col.lower()]
            
        if horodate_cols:
                date_col = horodate_cols[0]
        elif date_cols:
                date_col = date_cols[0]
        else:
                st.error("Aucune colonne de date ou d'horodate trouvée.")
                return df
     
        val_col = [col for col in df_b.columns if 'valeur' in col.lower() or 'soutirage' in col.lower() or 'puissance' in col.lower()]
        if not val_col:
             st.error("Aucune colonne de valeur trouvée.")
             return df
        val_col = val_col[0]

        prm_cols = [col for col in df_b.columns if 'prm' in col.lower()]

        if prm_cols:
            prm_col = prm_cols[0]
            df_b = df_b[[prm_col, date_col, val_col]].copy()
            df_b.columns = ["prm", "date", "valeur"]
        else:
            df_b = df_b[[date_col, val_col]].copy()
            df_b.columns = ["date", "valeur"]
            df_b["prm"] = "unique"

        df_b["date"] = pd.to_datetime(df_b["date"], utc=True)

        if df_b["valeur"].dtype == object:
            df_b["valeur"] = df_b["valeur"].str.replace(',', '.').astype(float)

        # --- Ré-échantillonnage PAR BORNE sur la grille du dataframe principal ---
        pas_principal = df.index[1] - df.index[0]

        series_par_borne = []
        for prm, g in df_b.groupby("prm"):
            s = g.set_index("date")["valeur"].sort_index()
            s = s.resample(pas_principal).mean()
            series_par_borne.append(s)

        conso_bornes_totale = pd.concat(series_par_borne, axis=1, sort=False).sum(axis=1, skipna=True)
        df_b_final = conso_bornes_totale.to_frame("valeur")
        df_b_final["conso_bornes_kW"] = df_b_final["valeur"] / 1000.0

        # Fusion avec le dataframe principal
        df = df.join(df_b_final["conso_bornes_kW"], how='left')
        df["conso_bornes_kW"] = df["conso_bornes_kW"].fillna(0)
        df["conso_kW"] = df["conso_kW"] + df["conso_bornes_kW"]

       except Exception as e:
        st.error(f"Erreur lors de la lecture du fichier Enedis des bornes : {e}")
    return df

def simuler_systeme_avec_batterie(df, capacite_kwh, p_max_kw=3.0, soc_initial_pct=0.0):
    if len(df) > 1:
        dt = (df.index[1] - df.index[0]).total_seconds() / 3600.0
    else:
        dt = 1.0

    soc_kwh = capacite_kwh * (soc_initial_pct / 100.0)
    soc_history, autoconso_directe, autoconso_batterie, import_reseau, export_reseau = [], [], [], [], []
    
    for row in df.itertuples():
        conso = row.conso_kW
        prod = row.prod_kW
        
        flux_direct = min(conso, prod)
        surplus = prod - flux_direct
        deficit = conso - flux_direct
        
        charge_bat = 0.0
        decharge_bat = 0.0
        
        if surplus > 0 and capacite_kwh > 0:
            max_power_to_fill = (capacite_kwh - soc_kwh) / dt
            charge_bat = min(surplus, p_max_kw, max_power_to_fill)
            soc_kwh += charge_bat * dt 
            surplus -= charge_bat
            
        elif deficit > 0 and capacite_kwh > 0:
            max_power_to_empty = soc_kwh / dt
            decharge_bat = min(deficit, p_max_kw, max_power_to_empty)
            soc_kwh -= decharge_bat * dt 
            deficit -= decharge_bat
            
        soc_history.append(soc_kwh)
        autoconso_directe.append(flux_direct)
        autoconso_batterie.append(decharge_bat)
        import_reseau.append(deficit)
        export_reseau.append(surplus) 
        
    df_res = df.copy()
    df_res["SoC_kWh"] = soc_history
    df_res["SoC_pourcent"] = (df_res["SoC_kWh"] / capacite_kwh * 100) if capacite_kwh > 0 else 0
    df_res["Autoconso_Totale_kW"] = np.array(autoconso_directe) + np.array(autoconso_batterie)
    df_res["Import_Reseau_kW"] = import_reseau
    df_res["Export_Reseau_kW"] = export_reseau
    
    
    return df_res, dt

def trouver_capacite_ideale(df_res, type_methode, param):
    """
    Fonction centrale pour calculer la capacité requise selon l'hypothèse choisie.
    """
    caps = df_res["Capacité (kWh)"].values.astype(float)
    gains = df_res["Gain Énergétique (kWh)"].values.astype(float)
    
    # Gestion du TAP selon la structure du dataframe (Onglet 3 ou Onglet 4)
    if "TAP (%)" in df_res.columns:
        taps = df_res["TAP (%)"].values.astype(float)
    elif "TAP global (%)" in df_res.columns:
        taps = df_res["TAP global (%)"].values.astype(float)
    else:
        taps = np.zeros_like(caps) # Fallback sécurisé

    if type_methode == "kneedle":
        # Méthode Kneedle : Recherche de la distance max entre la courbe et la corde
        x_norm = (caps - caps.min()) / (caps.max() - caps.min()) if caps.max() > caps.min() else caps
        y_norm = (gains - gains.min()) / (gains.max() - gains.min()) if gains.max() > gains.min() else gains
        x1, y1, x2, y2 = x_norm[0], y_norm[0], x_norm[-1], y_norm[-1]
        dist_denom = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
        if dist_denom > 0:
            distances = np.abs((y2 - y1) * x_norm - (x2 - x1) * y_norm + x2 * y1 - y2 * x1) / dist_denom
            idx = int(np.argmax(distances))
            return caps[idx]
        return caps[0]

    elif type_methode == "plateau_gain":
        # Atteindre X% du gain max
        seuil = gains.max() * param
        idx = int(np.argmax(gains >= seuil))
        return caps[idx]

    elif type_methode == "marginal_abs":
        # Gain marginal sous un seuil absolu
        for i in range(1, len(caps)):
            if (gains[i] - gains[i - 1]) < param:
                return caps[i - 1]
        return caps[-1]

    elif type_methode == "plateau_tap":
        # Atteindre X% du TAP max possible
        seuil = taps.max() * param
        idx = int(np.argmax(taps >= seuil))
        return caps[idx]

# ==========================================
# TARIFS BPU OCTOPUS ENERGY — Année 2026
# ==========================================
TARIFS_BPU = {
    "C5 - Bâtiments et équipements": {
        "cadrans_disponibles": ["Base (1 cadran)", "HP/HC (2 cadrans)", "4 cadrans saisonniers"],
        "fourniture": {
            "Base (1 cadran)": {"Base": 75.81},
            "HP/HC (2 cadrans)": {"HP": 77.71, "HC": 63.42},
            "4 cadrans saisonniers": {"HPSh": 105.93, "HCSh": 80.69, "HPSb": 56.01, "HCSb": 50.34},
        },
        "capacite": {
            "Base (1 cadran)": {"Base": 0.59},
            "HP/HC (2 cadrans)": {"HP": 0.78, "HC": 0.13},
            "4 cadrans saisonniers": {"HPSh": 1.13, "HCSh": 0.15, "HPSb": 0.00, "HCSb": 0.00},
        },
    },
    "C5 - Éclairage public": {
        "cadrans_disponibles": ["Base (1 cadran)"],
        "fourniture": {"Base (1 cadran)": {"Base": 72.81}},
        "capacite": {"Base (1 cadran)": {"Base": 0.16}},
    },
    "C4": {
        "cadrans_disponibles": ["4 cadrans saisonniers"],
        "fourniture": {"4 cadrans saisonniers": {"HPSh": 106.73, "HCSh": 77.84, "HPSb": 55.81, "HCSb": 50.09}},
        "capacite": {"4 cadrans saisonniers": {"HPSh": 1.10, "HCSh": 0.07, "HPSb": 0.00, "HCSb": 0.00}},
    },
    "C2": {
        "cadrans_disponibles": ["5 cadrans (avec Pointe)"],
        "fourniture": {"5 cadrans (avec Pointe)": {"Pte": 128.36, "HPSh": 106.16, "HCSh": 75.62, "HPSb": 55.60, "HCSb": 47.24}},
        "capacite": {"5 cadrans (avec Pointe)": {"Pte": 2.79, "HPSh": 0.93, "HCSh": 0.00, "HPSb": 0.00, "HCSb": 0.00}},
    },
}
PRIX_GO = 1.49    # €/MWh — Garanties d'Origine
PRIX_CEE = 11.23  # €/MWh — obligations d'économies d'énergie


def classer_cadran(timestamp, structure_cadran):
    """
     Calendrier HP/HC et saisons par DÉFAUT, à confirmer avec le contrat Enedis réel du TE13 :
       - Heures Creuses : 22h-6h  /  Heures Pleines : 6h-22h
       - Saison haute : novembre à mars  /  Saison basse : avril à octobre
       - "Pointe" (C2) non définie précisément ici -> traitée comme HPSh/HPSb par défaut
    """
    heure = timestamp.hour
    mois = timestamp.month
    est_hc = (heure >= 22) or (heure < 6)
    est_saison_haute = mois in (11, 12, 1, 2, 3)
    if structure_cadran == "Base (1 cadran)":
        return "Base"
    elif structure_cadran == "HP/HC (2 cadrans)":
        return "HC" if est_hc else "HP"
    elif structure_cadran in ("4 cadrans saisonniers", "5 cadrans (avec Pointe)"):
        if est_saison_haute:
            return "HCSh" if est_hc else "HPSh"
        else:
            return "HCSb" if est_hc else "HPSb"
    return "Base"


def prix_moyen_pondere_ttc(series_puissance_kw, dt_heures, segment, structure_cadran,
                             inclure_go=True, accise_eur_mwh=25.0, taux_tva=0.20):
    """Prix moyen pondéré (€/kWh TTC) d'un profil de charge, pondéré par l'énergie réelle par cadran."""
    tarif_fourniture = TARIFS_BPU[segment]["fourniture"][structure_cadran]
    tarif_capacite = TARIFS_BPU[segment]["capacite"][structure_cadran]
    cadrans = series_puissance_kw.index.map(lambda t: classer_cadran(t, structure_cadran))
    energie_kwh = series_puissance_kw.values * dt_heures
    df_tmp = pd.DataFrame({"cadran": cadrans, "energie_kwh": energie_kwh})
    energie_par_cadran = df_tmp.groupby("cadran")["energie_kwh"].sum()
    cout_total_htt = 0.0
    for cadran, energie in energie_par_cadran.items():
        prix_fourniture = tarif_fourniture.get(cadran, 0) / 1000.0
        prix_capacite = tarif_capacite.get(cadran, 0) / 1000.0
        prix_go = (PRIX_GO / 1000.0) if inclure_go else 0.0
        prix_cee = PRIX_CEE / 1000.0
        prix_accise = accise_eur_mwh / 1000.0
        prix_htt = prix_fourniture + prix_capacite + prix_go + prix_cee + prix_accise
        cout_total_htt += energie * prix_htt
    energie_totale = energie_par_cadran.sum()
    prix_moyen_htt = cout_total_htt / energie_totale if energie_totale > 0 else 0.0
    return prix_moyen_htt * (1 + taux_tva), energie_par_cadran


def calculer_flux_et_indicateurs(gain_net_kwh_an1, capex, opex_annuel, prix_achat_evite, prix_vente_reseau,
                                   taux_actualisation, duree_vie_ans, degradation_pct_an=0.0):
    """Calcule VAN, TRI, LCOS, Payback et ratio B/C pour une capacité donnée."""
    marge_arbitrage = prix_achat_evite - prix_vente_reseau
    annees = np.arange(0, duree_vie_ans + 1)
    recettes = np.zeros(duree_vie_ans + 1)
    opex = np.zeros(duree_vie_ans + 1)
    energies = np.zeros(duree_vie_ans + 1)
    for annee in range(1, duree_vie_ans + 1):
        facteur_degradation = (1 - degradation_pct_an) ** (annee - 1)
        energies[annee] = gain_net_kwh_an1 * facteur_degradation
        recettes[annee] = energies[annee] * marge_arbitrage
        opex[annee] = opex_annuel
    flux = recettes - opex
    flux[0] = -capex
    facteurs = 1 / (1 + taux_actualisation) ** annees
    van = float(np.sum(flux * facteurs))

    def van_pour_taux(r):
        return np.sum(flux / (1 + r) ** annees)

    tri = None
    if van_pour_taux(-0.99) > 0 and van_pour_taux(10.0) < 0:
        lo, hi = -0.99, 10.0
        for _ in range(200):
            mid = (lo + hi) / 2
            if van_pour_taux(mid) > 0:
                lo = mid
            else:
                hi = mid
        tri = (lo + hi) / 2
    couts_actualises = capex + float(np.sum(opex[1:] * facteurs[1:]))
    energie_actualisee = float(np.sum(energies[1:] * facteurs[1:]))
    lcos = couts_actualises / energie_actualisee if energie_actualisee > 0 else float("nan")
    cumul = np.cumsum(flux)
    payback = None
    idx_positif = np.where(cumul >= 0)[0]
    if len(idx_positif) > 0 and idx_positif[0] > 0:
        i = idx_positif[0]
        payback = float((i - 1) + (-cumul[i - 1] / flux[i])) if flux[i] != 0 else float(i)
    recettes_actualisees = float(np.sum(recettes[1:] * facteurs[1:]))
    ratio_bc = recettes_actualisees / couts_actualises if couts_actualises > 0 else float("nan")
    return {"van": van, "tri": tri * 100 if tri is not None else None, "lcos": lcos,
            "payback": payback, "ratio_bc": ratio_bc}


def calculer_tableau_enolab(capex, opex_an1, taux_inflation_opex, gain_net_kwh_an1, prix_moyen_ttc_an1,
                              taux_inflation_energie, revenu_producteur_an1, taux_inflation_revenu_producteur,
                              duree_vie_ans, degradation_pct_an=0.0):
    lignes = [{
        "Année": "A0", "CAPEX (€ HT)": -capex, "OPEX (€ HT)": np.nan,
        "Economie ACI (€ TTC)": np.nan, "Revenu producteur (€)": np.nan,
        "Economie nette (€)": -capex, "Flux cumulés (€)": -capex,
    }]
    flux_cumule = -capex
    for annee in range(1, duree_vie_ans + 1):
        facteur_degradation = (1 - degradation_pct_an) ** (annee - 1)
        opex = -opex_an1 * (1 + taux_inflation_opex) ** (annee - 1)
        gain_kwh = gain_net_kwh_an1 * facteur_degradation
        prix_ttc = prix_moyen_ttc_an1 * (1 + taux_inflation_energie) ** (annee - 1)
        economie_aci = gain_kwh * prix_ttc
        revenu_producteur = -revenu_producteur_an1 * (1 + taux_inflation_revenu_producteur) ** (annee - 1)
        economie_nette = economie_aci + revenu_producteur + opex
        flux_cumule += economie_nette
        lignes.append({
            "Année": f"A{annee}", "CAPEX (€ HT)": np.nan, "OPEX (€ HT)": opex,
            "Economie ACI (€ TTC)": economie_aci, "Revenu producteur (€)": revenu_producteur,
            "Economie nette (€)": economie_nette, "Flux cumulés (€)": flux_cumule,
        })
    return pd.DataFrame(lignes)

# ==========================================
# 2. INTERFACE STREAMLIT
# ==========================================

st.title( "TE13 — Dimensionnement de la Batterie de stockage")

# --- Panneau latéral : Données ---
st.sidebar.header("Données réelles")

fichier_conso = st.sidebar.file_uploader("Courbe de charge - Siège TE13", type=["csv", "xlsx", "xls"])
fichier_bornes = st.sidebar.file_uploader("Courbe de charge - Bornes de recharge  :", type=["csv", "xlsx", "xls"])
fichier_prod = st.sidebar.file_uploader("Production PV - Ombrière", type=["csv", "xlsx", "xls"])



if fichier_conso is not None and fichier_prod is not None:
    st.sidebar.success("Fichiers réels chargés.")
    df_complet = charger_donnees_reelles(fichier_conso, fichier_prod, fichier_bornes)
    
    # --- Sélection de la période ---
    st.markdown("---")
    st.header("Période d'analyse")

    date_min = df_complet.index.min().date()
    date_max = df_complet.index.max().date()

    col_date1, col_date2 = st.columns(2)
    date_debut = col_date1.date_input("Date de début", value=date_min, min_value=date_min, max_value=date_max, format="DD/MM/YYYY")
    date_fin = col_date2.date_input("Date de fin (incluse)", value=date_min, min_value=date_min, max_value=date_max, format="DD/MM/YYYY")

    mask = (df_complet.index.date >= date_debut) & (df_complet.index.date <= date_fin)
    df = df_complet.loc[mask]

    if df.empty:
        st.error("Aucune donnée trouvée pour les dates sélectionnées.")
    else:
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Simulation Temporelle Courte Durée",
            "Simulation Temporelle Longue Durée",
            "Gain de la Batterie",
            "Analyse Annuelle",
            "Analyse Économique"
        ])
        
        # ----------------------------------------------------
        # ONGLET 1 : Simulation Temporelle
        # ----------------------------------------------------
        with tab1:
            st.markdown("---")
            st.header("Simulation temporelle courte durée (< 1 mois) ")
            st.markdown("""
       Cet onglet est adapté à l'analyse des courbes de charges, de production et de la capacité utilisé de la batterie sur 
       périodes courtes (D'une journée à quelques semaines) avec un pas de temps de 30 min. 
       
       """  )

            col_bat1, col_bat2 = st.columns(2)
            capacite_batterie = col_bat1.slider("Capacité (kWh)", min_value=0.0, max_value=300.0, value=50.0, step=1.0, help="Volume total d'énergie stockable.")
            soc_initial = col_bat2.slider("Charge initiale (%)", min_value=0, max_value=100, value=0, step=5, help="Niveau de la batterie au début de la période sélectionnée.")

            puissance_onduleur = capacite_batterie / 2.0
            st.info(f"Règle appliquée : La puissance de l'onduleur est fixée à {puissance_onduleur:.1f} kW (moitié de la capacité de stockage).")

            unite_batterie = st.radio("Affichage de la batterie :", ["Pourcentage (%)", "Énergie (kWh)"], horizontal=True)

            df_simu, dt = simuler_systeme_avec_batterie(df, capacite_batterie, puissance_onduleur, soc_initial)

            fig_bat = make_subplots(specs=[[{"secondary_y": True}]])
            fig_bat.add_trace(go.Scatter(x=df.index, y=df_simu["conso_kW"], mode="lines", name="Consommation", line=dict(color="blue", width=2)), secondary_y=False)
            fig_bat.add_trace(go.Scatter(x=df.index, y=df_simu["prod_kW"], mode="lines", name="Production PV", line=dict(color="orange", width=2, dash="dash")), secondary_y=False)
            fig_bat.add_trace(go.Bar(x=df.index, y=df_simu["Import_Reseau_kW"], name="Achat Réseau (Déficit)", marker_color="rgba(255, 0, 0, 0.5)"), secondary_y=False)
            fig_bat.add_trace(go.Bar(x=df.index, y=df_simu["Export_Reseau_kW"], name="Injection Réseau (Surplus)", marker_color="rgba(0, 255, 0, 0.5)"), secondary_y=False)

            if unite_batterie == "Pourcentage (%)":
                y_bat_col, bat_name, bat_y_title, bat_y_range = "SoC_pourcent", "Niveau de Charge (%)", "État de charge (%)", [0, 105]
            else:
                y_bat_col, bat_name, bat_y_title = "SoC_kWh", "Énergie stockée (kWh)", "Énergie (kWh)"
                bat_y_range = [0, capacite_batterie * 1.05] if capacite_batterie > 0 else [0, 1]

            fig_bat.add_trace(go.Scatter(x=df.index, y=df_simu[y_bat_col], mode="lines", name=bat_name, line=dict(color="purple", width=3), fill="tozeroy", fillcolor="rgba(128, 0, 128, 0.1)"), secondary_y=True)

            fig_bat.update_layout(
                title=dict(text="Analyse des flux d'énergie avec batterie", font=dict(size=18)),
                hovermode="x unified", barmode="overlay",
                legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5),
                margin=dict(t=60, b=100, l=40, r=40)
            )
            fig_bat.update_yaxes(title_text="Puissance (kW)", secondary_y=False)
            fig_bat.update_yaxes(title_text=bat_y_title, range=bat_y_range, secondary_y=True)
            
            fig_bat.update_xaxes(type="date", tickformat="%H:%M\n%d/%m", hoverformat="%d/%m/%Y %H:%M", gridcolor="rgba(200, 200, 200, 0.2)")

            st.plotly_chart(fig_bat, use_container_width=True)

            
            # KPIs
            st.subheader("Performances du système sur la période")
            col_kpi1, col_kpi2, col_kpi3 = st.columns(3)

            conso_totale = df_simu["conso_kW"].sum() * dt
            prod_totale = df_simu["prod_kW"].sum() * dt
            export_totale = df_simu["Export_Reseau_kW"].sum() * dt
            autoconso_totale = df_simu["Autoconso_Totale_kW"].sum() * dt 

            # Référence : autoconsommation SANS batterie (autoconsommation directe uniquement)
            autoconso_sans_bat = np.minimum(df_simu["conso_kW"], df_simu["prod_kW"]).sum() * dt
            gain_net = max(0, autoconso_totale - autoconso_sans_bat)

            energie_pv_valorisee = prod_totale - export_totale
            nouveau_tac = (energie_pv_valorisee / prod_totale * 100) if prod_totale > 0 else 0
            nouveau_tap = (autoconso_totale / conso_totale * 100) if conso_totale > 0 else 0

            col_kpi1.metric("Taux d'Autoconso. (TAC)", f"{nouveau_tac:.1f} %")
            col_kpi2.metric("Taux d'Autoprod. (TAP)", f"{nouveau_tap:.1f} %")
            col_kpi3.metric(
                "Énergie totale économisée (gain net)",
                f"{gain_net:.1f} kWh",
                help=f"Gain net apporté par la batterie par rapport à une installation sans stockage, "
                     f"qui autoconsommerait naturellement {autoconso_sans_bat:.1f} kWh sur cette période."
            )
            
         # ----------------------------------------------------
         # ONGLET 2 : simulation temporelle longue durée
         # ----------------------------------------------------
            
            
        with tab2:
            st.header("Simulation Longue Durée (> 1 mois)")
            st.markdown("""
            Cet onglet est adapté à l'analyse de périodes longues (plusieurs semaines à plusieurs mois), 
            pour lesquelles le graphique détaillé de l'onglet **« Simulation Temporelle »** devient illisible 
            (trop de points à pas de 30 minutes).

            Le calcul de la batterie est effectué à la résolution complète des données (pour garantir un bilan 
            énergétique exact), mais **l'affichage est resserré à 1 point par jour** : le pic de consommation 
            et le pic de production de chaque journée, ainsi que l'état de charge de la batterie.
            """)

            if (date_fin - date_debut).days < 30:
                st.warning("La période sélectionnée fait moins d'un mois. Pour une vision détaillée, "
                           "privilégiez plutôt l'onglet « Simulation Temporelle ».")

            col_bat1_ld, col_bat2_ld = st.columns(2)
            capacite_batterie_ld = col_bat1_ld.slider(
                "Capacité (kWh)", min_value=0.0, max_value=300.0, value=50.0, step=1.0,
                key="cap_longue_duree", help="Volume total d'énergie stockable."
            )
            soc_initial_ld = col_bat2_ld.slider(
                "Charge initiale (%)", min_value=0, max_value=100, value=0, step=5,
                key="soc_longue_duree", help="Niveau de la batterie au début de la période sélectionnée."
            )
            puissance_onduleur_ld = capacite_batterie_ld / 2.0
            st.info(f"Règle appliquée : puissance de l'onduleur fixée à {puissance_onduleur_ld:.1f} kW "
                    f"(moitié de la capacité de stockage).")

            afficher_reseau_ld = st.checkbox(
                "Afficher aussi les achats/injections réseau (cumul journalier, en kWh/j)",
                value=False,
                key="chk_reseau_ld",
                help="Ajoute l'énergie totale achetée et injectée au réseau chaque jour (kWh cumulés "
                "sur ~48 pas de 30 min). Attention : ce sont des valeurs d'énergie, à ne pas comparer "
                "directement aux courbes de puissance (kW) affichées par défaut."
            )

            affichage_soc_ld = st.radio(
                "Affichage de l'état de charge :",
                ["Pic quotidien", "Moyenne quotidienne"],
                horizontal=True,
                key="radio_soc_ld",
                help="Pic quotidien : la charge maximale atteinte dans la journée (permet de voir si la "
                "batterie a rejoint 100 %). Moyenne quotidienne : le niveau moyen sur la journée, plus "
                "représentatif du taux d'utilisation réel, mais n'atteint jamais 100 % même les jours où "
                "la batterie s'est pleinement chargée."
            )

            # --- Simulation à pleine résolution (nécessaire pour un bilan énergétique correct) ---
            df_simu_ld, dt_ld = simuler_systeme_avec_batterie(
                df, capacite_batterie_ld, puissance_onduleur_ld, soc_initial_ld
            )

            # --- Ré-échantillonnage : 1 point par jour (les 3 courbes de base) ---
            jours = df_simu_ld.index.date
            conso_pic_j = df_simu_ld.groupby(jours)["conso_kW"].max()
            prod_pic_j  = df_simu_ld.groupby(jours)["prod_kW"].max()

            if affichage_soc_ld == "Pic quotidien":
                soc_j = df_simu_ld.groupby(jours)["SoC_pourcent"].max()
                soc_nom = "Pic d'état de charge (%)"
            else:
                soc_j = df_simu_ld.groupby(jours)["SoC_pourcent"].mean()
                soc_nom = "État de charge moyen (%)"

            series_a_convertir = [conso_pic_j, prod_pic_j, soc_j]

            # --- Réseau : optionnel, cumul journalier en kWh ---
            if afficher_reseau_ld:
                import_j = df_simu_ld.groupby(jours)["Import_Reseau_kW"].sum() * dt_ld
                export_j = df_simu_ld.groupby(jours)["Export_Reseau_kW"].sum() * dt_ld
                series_a_convertir += [import_j, export_j]

            for serie in series_a_convertir:
                serie.index = pd.to_datetime(serie.index)

            # --- Graphique ---
            fig_ld = make_subplots(specs=[[{"secondary_y": True}]])

            if afficher_reseau_ld:
                fig_ld.add_trace(go.Bar(x=import_j.index, y=import_j.values,
                    name="Achat Réseau (kWh/j)", marker_color="rgba(255, 0, 0, 0.3)"), secondary_y=False)
                fig_ld.add_trace(go.Bar(x=export_j.index, y=export_j.values,
                    name="Injection Réseau (kWh/j)", marker_color="rgba(0, 255, 0, 0.3)"), secondary_y=False)

            fig_ld.add_trace(go.Scatter(x=conso_pic_j.index, y=conso_pic_j.values, mode="lines",
                name="Pic de consommation (kW/j)", line=dict(color="royalblue", width=2)), secondary_y=False)

            fig_ld.add_trace(go.Scatter(x=prod_pic_j.index, y=prod_pic_j.values, mode="lines",
                name="Pic de production (kW/j)", line=dict(color="#FF8C00", width=2)), secondary_y=False)

            fig_ld.add_trace(go.Scatter(x=soc_j.index, y=soc_j.values, mode="lines",
                name=soc_nom, line=dict(color="purple", width=2),
                fill="tozeroy", fillcolor="rgba(128, 0, 128, 0.1)"), secondary_y=True)

            fig_ld.update_layout(
                title=dict(text="Simulation longue durée — 1 point par jour", font=dict(size=18)),
                hovermode="x unified", barmode="overlay",
                legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5),
                margin=dict(t=60, b=100, l=40, r=40)
            )
            axe_titre = "Puissance (kW)" + (" / Énergie cumulée (kWh/j)" if afficher_reseau_ld else "")
            fig_ld.update_yaxes(title_text=axe_titre, secondary_y=False)
            fig_ld.update_yaxes(title_text=soc_nom, range=[0, 105], secondary_y=True)
            fig_ld.update_xaxes(type="date", title_text="Jour", gridcolor="rgba(200, 200, 200, 0.2)")

            st.plotly_chart(fig_ld, use_container_width=True)

            # --- KPIs sur la période complète (calculés à pleine résolution, inchangés) ---
            st.subheader("Performances du système sur la période")
            col_kpi1_ld, col_kpi2_ld, col_kpi3_ld = st.columns(3)

            conso_totale_ld = df_simu_ld["conso_kW"].sum() * dt_ld
            prod_totale_ld = df_simu_ld["prod_kW"].sum() * dt_ld
            export_totale_ld = df_simu_ld["Export_Reseau_kW"].sum() * dt_ld
            autoconso_totale_ld = df_simu_ld["Autoconso_Totale_kW"].sum() * dt_ld

            # Référence : autoconsommation SANS batterie (autoconsommation directe uniquement)
            autoconso_sans_bat_ld = np.minimum(df_simu_ld["conso_kW"], df_simu_ld["prod_kW"]).sum() * dt_ld
            gain_net_ld = max(0, autoconso_totale_ld - autoconso_sans_bat_ld)

            energie_pv_valorisee_ld = prod_totale_ld - export_totale_ld
            tac_ld = (energie_pv_valorisee_ld / prod_totale_ld * 100) if prod_totale_ld > 0 else 0
            tap_ld = (autoconso_totale_ld / conso_totale_ld * 100) if conso_totale_ld > 0 else 0

            col_kpi1_ld.metric("Taux d'Autoconso. (TAC)", f"{tac_ld:.1f} %")
            col_kpi2_ld.metric("Taux d'Autoprod. (TAP)", f"{tap_ld:.1f} %")
            col_kpi3_ld.metric(
                "Énergie totale économisée (gain net)",
                f"{gain_net_ld:.1f} kWh",
                help=f"Gain net apporté par la batterie par rapport à une installation sans stockage, "
                     f"qui autoconsommerait naturellement {autoconso_sans_bat_ld:.1f} kWh sur cette période."
            )
        # ----------------------------------------------------
        # ONGLET 3 : Isoler le Gain de la Batterie
        # ----------------------------------------------------
        with tab3:
            st.header("Gain de la Batterie")
            st.info(f"Période d'analyse : Du {date_debut.strftime('%d/%m/%Y')} au {date_fin.strftime('%d/%m/%Y')} ({(date_fin - date_debut).days + 1} jours)")
            st.markdown("""
            L'objectif ici est de visualiser uniquement l'apport de la batterie par rapport à une installation solaire simple sans stockage.

            **Qu'est-ce que le "gain net" ?** C'est la quantité d'énergie solaire supplémentaire autoconsommée grâce à la batterie,
            par rapport à une installation identique fonctionnant sans stockage (autoconsommation directe uniquement, sans surplus stocké ni redistribué).
            Concrètement : *Gain net = Énergie autoconsommée avec batterie − Énergie autoconsommée sans batterie*.
            Il représente donc les kWh de production solaire qui, sans la batterie, auraient été perdus (renvoyés au réseau) et qui sont désormais utilisés sur place.
            """)
            
            max_cap_test = 300
            soc_init_test = st.number_input("Charge initiale au départ (%) :", value=0, min_value=0, max_value=100, key="num_soc_t2", help="Niveau de charge de la batterie au début de la période sélectionnée.")

            st.info("Règle appliquée : Pour chaque capacité testée, la puissance de l'onduleur (kW) sera automatiquement égale à la moitié de la capacité (kWh).")

            if st.button("Lancer l'analyse de sensibilité", key="btn_run_t2"):
                with st.spinner('Calcul en cours...'):
                    dt_val = (df.index[1] - df.index[0]).total_seconds() / 3600.0 if len(df) > 1 else 1.0
                    autoconso_sans_bat = np.minimum(df["conso_kW"], df["prod_kW"]).sum() * dt_val
                    
                    resultats = []
                    capacites_testees = np.arange(0, max_cap_test + 5, 5) 
                    
                    for cap in capacites_testees:
                        p_ond = cap / 2.0
                        df_res, dt_local = simuler_systeme_avec_batterie(df, cap, p_max_kw=p_ond, soc_initial_pct=soc_init_test)
                        
                        conso_tot = df_res["conso_kW"].sum() * dt_local
                        autoconso_tot = df_res["Autoconso_Totale_kW"].sum() * dt_local
                        gain_batterie = max(0, autoconso_tot - autoconso_sans_bat)
                        tap_val = (autoconso_tot / conso_tot * 100) if conso_tot > 0 else 0
                        
                        resultats.append({
                            "Capacité (kWh)": cap,
                            "Gain de la Batterie (kWh)": gain_batterie,
                            "TAP global (%)": tap_val
                        })
                    
                    df_resultats_t2 = pd.DataFrame(resultats)
                    
                    fig_opti = make_subplots(specs=[[{"secondary_y": True}]])
                    fig_opti.add_trace(go.Scatter(x=df_resultats_t2["Capacité (kWh)"], y=df_resultats_t2["Gain de la Batterie (kWh)"], mode="lines+markers", name="Gain net apporté par la batterie (kWh)", fill='tozeroy', line=dict(color='green', width=3)), secondary_y=False)
                    fig_opti.add_trace(go.Scatter(x=df_resultats_t2["Capacité (kWh)"], y=df_resultats_t2["TAP global (%)"], mode="lines", name="Taux d'Autoproduction (TAP)", line=dict(color='blue', width=2, dash='dash')), secondary_y=True)
                    
                    fig_opti.update_layout(title="Gain Énergétique Net de la Batterie", xaxis_title="Taille de la batterie simulée (kWh)", hovermode="x unified")
                    fig_opti.update_yaxes(title_text="Énergie Supplémentaire Économisée (kWh)", secondary_y=False)
                    fig_opti.update_yaxes(title_text="Taux d'Autoproduction Global (%)", range=[0, 105], secondary_y=True)
                    
                    st.plotly_chart(fig_opti, use_container_width=True)
                    st.success(f"Sans batterie, l'installation autoconsomme naturellement {autoconso_sans_bat:.1f} kWh sur cette période.")

        # ----------------------------------------------------
        # ONGLET 4 : Analyse Annuelle
        # ----------------------------------------------------
        with tab4:
            st.header("Analyse Annuelle : Gain, Autoproduction et Autoconsommation")
            st.markdown("""
            Cette analyse simule une **année complète glissante démarrant le 1er janvier** (indépendamment de la période
            sélectionnée plus haut), et calcule pour chaque capacité de batterie testée :
            - le **gain énergétique**,
            - le **taux d'autoproduction (TAP)**,
            - le **taux d'autoconsommation (TAC)**.

            **Qu'est-ce que le "gain énergétique" ?** C'est la quantité d'énergie solaire supplémentaire autoconsommée sur l'année grâce à la batterie,
            par rapport à la même installation sans stockage (autoconsommation directe uniquement). Autrement dit :
            *Gain énergétique = Énergie autoconsommée avec batterie sur l'année − Énergie autoconsommée sans batterie sur l'année*.
            Il représente les kWh de production solaire qui, sans la batterie, auraient été perdus (renvoyés au réseau) et qui sont désormais valorisés sur place.
            """)

            candidats_1er_janvier = df_complet.index[(df_complet.index.month == 1) & (df_complet.index.day == 1)]

            if len(candidats_1er_janvier) == 0:
                st.warning("Aucun 1er janvier n'a été trouvé dans les données importées. Impossible de construire l'année complète glissante.")
            else:
                date_debut_annee = candidats_1er_janvier.normalize().unique()[0]
                date_fin_annee_theorique = date_debut_annee + pd.Timedelta(days=365)
                date_fin_annee = min(date_fin_annee_theorique, df_complet.index.max())

                df_annee = df_complet.loc[(df_complet.index >= date_debut_annee) & (df_complet.index <= date_fin_annee)]
                nb_jours_dispo = (date_fin_annee - date_debut_annee).days
                
                if nb_jours_dispo < 360:
                    st.warning(f"Seulement {nb_jours_dispo} jours de données sont disponibles à partir du 1er janvier ({date_debut_annee.strftime('%d/%m/%Y')}). L'année n'est pas complète : les résultats seront calculés sur la période disponible.")

                st.info(f"Année analysée : du {date_debut_annee.strftime('%d/%m/%Y')} au {date_fin_annee.strftime('%d/%m/%Y')} ({nb_jours_dispo} jours).")

                max_cap_test_t4 = 300
                soc_initial_janvier_t4 = 0

                st.info("Règle appliquée : Pour chaque capacité testée, la puissance de l'onduleur (kW) sera automatiquement égale à la moitié de la capacité (kWh).")

                if st.button("Lancer l'analyse annuelle", key="btn_run_t4"):
                    with st.spinner("Calcul en cours..."):

                        dt_annee = (df_annee.index[1] - df_annee.index[0]).total_seconds() / 3600.0 if len(df_annee) > 1 else 1.0
                        autoconso_sans_bat_annee = np.minimum(df_annee["conso_kW"], df_annee["prod_kW"]).sum() * dt_annee

                        resultats_t4 = []
                        capacites_testees_t4 = np.arange(0, max_cap_test_t4 + 5, 5)

                        for cap in capacites_testees_t4:
                            p_ond = cap / 2.0
                            df_res_t4, dt_t4 = simuler_systeme_avec_batterie(df_annee, cap, p_max_kw=p_ond, soc_initial_pct=soc_initial_janvier_t4)

                            conso_tot_t4 = df_res_t4["conso_kW"].sum() * dt_t4
                            prod_tot_t4 = df_res_t4["prod_kW"].sum() * dt_t4
                            export_tot_t4 = df_res_t4["Export_Reseau_kW"].sum() * dt_t4
                            autoconso_tot_t4 = df_res_t4["Autoconso_Totale_kW"].sum() * dt_t4

                            gain_batterie_t4 = max(0, autoconso_tot_t4 - autoconso_sans_bat_annee)
                            energie_pv_valorisee_t4 = prod_tot_t4 - export_tot_t4
                            tac_val_t4 = (energie_pv_valorisee_t4 / prod_tot_t4 * 100) if prod_tot_t4 > 0 else 0
                            tap_val_t4 = (autoconso_tot_t4 / conso_tot_t4 * 100) if conso_tot_t4 > 0 else 0

                            resultats_t4.append({
                                "Capacité (kWh)": cap,
                                "Autoconso Totale (kWh)": autoconso_tot_t4,
                                "Gain Énergétique (kWh)": gain_batterie_t4,
                                "TAP (%)": tap_val_t4,
                                "TAC (%)": tac_val_t4
                            })

                        df_resultats_t4 = pd.DataFrame(resultats_t4)

                        fig_t4 = make_subplots(specs=[[{"secondary_y": True}]])
                        fig_t4.add_trace(go.Scatter(x=df_resultats_t4["Capacité (kWh)"], y=df_resultats_t4["Gain Énergétique (kWh)"], mode="lines+markers", name="Gain Énergétique (kWh)", fill='tozeroy', line=dict(color='green', width=3)), secondary_y=False)
                        fig_t4.add_trace(go.Scatter(x=df_resultats_t4["Capacité (kWh)"], y=df_resultats_t4["TAP (%)"], mode="lines", name="Taux d'Autoproduction (TAP)", line=dict(color='blue', width=2, dash='dash')), secondary_y=True)
                        fig_t4.add_trace(go.Scatter(x=df_resultats_t4["Capacité (kWh)"], y=df_resultats_t4["TAC (%)"], mode="lines", name="Taux d'Autoconsommation (TAC)", line=dict(color='red', width=2, dash='dot')), secondary_y=True)

                        fig_t4.update_layout(title="Gain Énergétique, TAP et TAC sur une Année Complète", xaxis_title="Taille de la batterie simulée (kWh)", hovermode="x unified", legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5))
                        fig_t4.update_yaxes(title_text="Gain Énergétique (kWh)", secondary_y=False)
                        fig_t4.update_yaxes(title_text="Taux (%)", range=[0, 105], secondary_y=True)

                        st.plotly_chart(fig_t4, use_container_width=True)
                        st.success(f"Sans batterie, l'installation solaire autoconsomme naturellement {autoconso_sans_bat_annee:.1f} kWh sur cette année complète.")

                        st.session_state["df_resultats_t4"] = df_resultats_t4

                # ZONE DE SÉLECTION DES HYPOTHÈSES (Cocher les cases)
                if "df_resultats_t4" in st.session_state:
                    df_res_t4 = st.session_state["df_resultats_t4"]

                    st.markdown("---")
                    st.subheader(" Validation des Hypothèses Techniques")
                    st.markdown("Cochez les critères que vous souhaitez imposer à votre batterie. L'outil calculera la capacité nécessaire pour valider **simultanément** toutes vos hypothèses.")

                    options_methodes = {
                        "Coude géométrique de rentabilité": ("kneedle", None),
                        "Atteindre le plateau à 90 % du gain net maximal": ("plateau_gain", 0.90),
                        "Atteindre le plateau à 95 % du gain net maximal": ("plateau_gain", 0.95),
                        "Gain marginal < 200 kWh (pour chaque ajout de 5 kWh)": ("marginal_abs", 200),
                        "Gain marginal < 100 kWh (pour chaque ajout de 5 kWh)": ("marginal_abs", 100),
                        "Gain marginal < 50 kWh (pour chaque ajout de 5 kWh)": ("marginal_abs", 50),
                        "Atteindre le plateau à 90 % du TAP maximal possible": ("plateau_tap", 0.90),
                        "Atteindre le plateau à 95 % du TAP maximal possible": ("plateau_tap", 0.95),
                    }
                    
                    

# Texte affiché au survol du petit "?" à côté de chaque case
                    aide_methodes = {
    "Coude géométrique de rentabilité":
        "Détecte automatiquement le point d'inflexion (le « coude ») de la courbe de gain : "
        "la capacité où l'écart entre la courbe et la droite reliant le premier et le dernier "
        "point testés (0 et 300 kWh) est maximal. Méthode géométrique (Kneedle), sans seuil à choisir.",
    "Atteindre le plateau à 90 % du gain net maximal":
        "Retient la plus petite capacité testée qui atteint déjà 90 % du gain énergétique "
        "maximal observé sur toute la plage testée (0 à 300 kWh).",
    "Atteindre le plateau à 95 % du gain net maximal":
        "Même principe que l'option 90 %, avec un seuil plus exigeant : 95 % du gain "
        "énergétique maximal possible.",
    "Gain marginal < 200 kWh (pour chaque ajout de 5 kWh)":
        "Parcourt les capacités testées par palier de 5 kWh et retient la dernière capacité "
        "pour laquelle le palier suivant apportait encore au moins 200 kWh de gain annuel. "
        "Dès qu'un palier rapporte moins que ce seuil, l'algorithme s'arrête et garde la capacité juste avant.",
    "Gain marginal < 100 kWh (pour chaque ajout de 5 kWh)":
        "Parcourt les capacités testées par palier de 5 kWh et retient la dernière capacité "
        "pour laquelle le palier suivant apportait encore au moins 100 kWh de gain annuel. "
        "Dès qu'un palier rapporte moins que ce seuil, l'algorithme s'arrête et garde la capacité juste avant.",
    "Gain marginal < 50 kWh (pour chaque ajout de 5 kWh)":
        "Parcourt les capacités testées par palier de 5 kWh et retient la dernière capacité "
        "pour laquelle le palier suivant apportait encore au moins 50kWh de gain annuel. "
        "Dès qu'un palier rapporte moins que ce seuil, l'algorithme s'arrête et garde la capacité juste avant.Conduit généralement à une capacité recommandée plus élevée que les deux options précédentes. ",
    "Atteindre le plateau à 90 % du TAP maximal possible":
        "Retient la plus petite capacité testée qui atteint déjà 90 % du Taux d'Autoproduction "
        "(TAP) maximal observé sur toute la plage testée.",
    "Atteindre le plateau à 95 % du TAP maximal possible":
        "Même principe que l'option 90 %, avec un seuil plus exigeant : 95 % du TAP "
        "maximal possible.",
                    }


                    col_c1, col_c2 = st.columns(2)
                    methodes_selectionnees = []
                    
                    for i, (label, params) in enumerate(options_methodes.items()):
                        # Cocher par défaut certaines options pour l'exemple
                        default_check = False
                        with (col_c1 if i % 2 == 0 else col_c2):
                            if st.checkbox(label, value=default_check, key=f"chk_hyp_{i}", help=aide_methodes[label]):
                                methodes_selectionnees.append(label)

                    if methodes_selectionnees:
                        caps_calculees = []
                        details_hypotheses = []

                        for methode in methodes_selectionnees:
                            type_m, param_m = options_methodes[methode]
                            cap_calc = trouver_capacite_ideale(df_res_t4, type_m, param_m)
                            caps_calculees.append(cap_calc)
                            details_hypotheses.append({
                                "Hypothèse validée": methode,
                                "Capacité minimale requise": f"{cap_calc} kWh"
                            })
                        
                        st.write("**Détail des exigences par hypothèse sélectionnée :**")
                        st.table(pd.DataFrame(details_hypotheses))

                        # La capacité finale est le MAX de toutes les capacités requises
                        cap_ideale_finale = max(caps_calculees)
                        ligne_ideale = df_res_t4[df_res_t4["Capacité (kWh)"] == cap_ideale_finale].iloc[0]

                        st.success(f"###  Capacité recommandée : {cap_ideale_finale:.0f} kWh")
                        st.markdown("*Pour satisfaire simultanément tous vos critères, le système retient la valeur la plus exigeante parmi vos choix.*")

                        col_res1, col_res2, col_res3, col_res4 = st.columns(4)
                        col_res1.metric("Capacité retenue", f"{cap_ideale_finale:.0f} kWh")
                        col_res2.metric("Gain net annuel", f"{ligne_ideale['Gain Énergétique (kWh)']:.0f} kWh")
                        col_res3.metric("TAP estimé", f"{ligne_ideale['TAP (%)']:.1f} %")
                        col_res4.metric("TAC estimé", f"{ligne_ideale['TAC (%)']:.1f} %")
                    else:
                        st.warning("Veuillez cocher au moins une hypothèse technique ci-dessus.")
  
        # ----------------------------------------------------
        # ONGLET 5 : Analyse Budgétaire
        # ----------------------------------------------------
                        
        with tab5:
            st.header("Analyse Économique ")
            st.markdown("""
            Cet onglet valorise financièrement le gain énergétique calculé dans l'onglet 4 (Analyse Annuelle),
            en utilisant les tarifs réels du BPU Octopus Energy 2026, pour déterminer la capacité de batterie
            économiquement optimale — pas seulement techniquement optimale.

             **Le CAPEX, l'OPEX, le taux d'actualisation et le prix de vente au réseau restent fictifs**,
            en attendant les données réelles. Les prix d'achat évités, eux, sont désormais calculés à partir
            du vrai BPU, pondérés par votre profil de consommation réel.
            """)

            if "df_resultats_t4" not in st.session_state:
                st.warning("Merci de d'abord lancer l'analyse annuelle dans l'onglet 4 : cet onglet réutilise "
                           "directement son résultat (gain énergétique par capacité testée).")
            else:
                df_res_t4 = st.session_state["df_resultats_t4"]

                # ==========================================
                # 1. TARIFICATION RÉELLE (BPU) — calculée en premier, réutilisée partout ensuite
                # ==========================================
                st.subheader("Tarification réelle (BPU Octopus Energy 2026)")

                col_t1, col_t2 = st.columns(2)
                with col_t1:
                    st.markdown("**Siège**")
                    segment_siege = st.selectbox("Segment tarifaire — Siège", list(TARIFS_BPU.keys()), key="segment_siege")
                    cadran_siege = st.selectbox("Structure de comptage — Siège",
                        TARIFS_BPU[segment_siege]["cadrans_disponibles"], key="cadran_siege")
                with col_t2:
                    st.markdown("**Bornes de recharge**")
                    segment_bornes = st.selectbox("Segment tarifaire — Bornes", list(TARIFS_BPU.keys()), key="segment_bornes")
                    cadran_bornes = st.selectbox("Structure de comptage — Bornes",
                        TARIFS_BPU[segment_bornes]["cadrans_disponibles"], key="cadran_bornes")

                col_t3, col_t4, col_t5 = st.columns(3)
                inclure_go = col_t3.checkbox("Inclure les Garanties d'Origine (GO)", value=True)
                accise_eur_mwh = col_t4.number_input("Accise électricité (€/MWh)", min_value=0.0, value=25.0, step=0.5,
                    help="Valeur fictive, en attente du taux réel applicable au TE13.")
                taux_tva = col_t5.number_input("TVA (%)", min_value=0.0, max_value=25.0, value=20.0, step=0.1) / 100.0

                dt_actuel = (df.index[1] - df.index[0]).total_seconds() / 3600.0
                conso_siege_seule = df["conso_kW"] - df["conso_bornes_kW"] if "conso_bornes_kW" in df.columns else df["conso_kW"]

                prix_ttc_siege, _ = prix_moyen_pondere_ttc(conso_siege_seule, dt_actuel, segment_siege, cadran_siege,
                    inclure_go, accise_eur_mwh, taux_tva)
                if "conso_bornes_kW" in df.columns and df["conso_bornes_kW"].sum() > 0:
                    prix_ttc_bornes, _ = prix_moyen_pondere_ttc(df["conso_bornes_kW"], dt_actuel, segment_bornes,
                        cadran_bornes, inclure_go, accise_eur_mwh, taux_tva)
                else:
                    prix_ttc_bornes = prix_ttc_siege

                volume_siege = conso_siege_seule.sum() * dt_actuel
                volume_bornes = df["conso_bornes_kW"].sum() * dt_actuel if "conso_bornes_kW" in df.columns else 0
                volume_total = volume_siege + volume_bornes
                prix_ttc_moyen = ((prix_ttc_siege * volume_siege + prix_ttc_bornes * volume_bornes) / volume_total
                                   if volume_total > 0 else prix_ttc_siege)

                col_p1, col_p2, col_p3 = st.columns(3)
                col_p1.metric("Prix moyen TTC — Siège", f"{prix_ttc_siege:.4f} €/kWh")
                col_p2.metric("Prix moyen TTC — Bornes", f"{prix_ttc_bornes:.4f} €/kWh")
                col_p3.metric("Prix moyen pondéré global", f"{prix_ttc_moyen:.4f} €/kWh")

                # ==========================================
                # 2. HYPOTHÈSES ÉCONOMIQUES (CAPEX / OPEX / actualisation) — toujours fictives
                # ==========================================
                st.markdown("---")
                st.subheader("Hypothèses économiques (valeurs fictives à ajuster)")

                col_e1, col_e2, col_e3 = st.columns(3)
                capex_unitaire = col_e1.number_input("Coût unitaire de la batterie (€/kWh)",
                    min_value=0.0, value=400.0, step=10.0,
                    help="Coût fictif par défaut. À remplacer par le devis réel du fournisseur.")
                capex_fixe = col_e2.number_input("Coûts fixes d'installation (€)",
                    min_value=0.0, value=15000.0, step=500.0,
                    help="Génie civil, raccordement, main d'œuvre... (valeur fictive).")
                duree_vie_ans = int(col_e3.number_input("Durée de vie / horizon d'étude (années)",
                    min_value=1, max_value=30, value=15, step=1))

                col_e4, col_e5, col_e6 = st.columns(3)
                opex_pct = col_e4.number_input("OPEX annuel (% du CAPEX)",
                    min_value=0.0, max_value=20.0, value=1.5, step=0.1,
                    help="Maintenance, assurance... (valeur fictive).") / 100.0
                taux_actualisation = col_e5.number_input("Taux d'actualisation (%)",
                    min_value=0.0, max_value=20.0, value=4.0, step=0.1) / 100.0
                degradation_pct = col_e6.number_input("Dégradation annuelle de la batterie (%)",
                    min_value=0.0, max_value=10.0, value=2.0, step=0.1,
                    help="Perte de capacité utile par an. Mettre 0 pour l'ignorer.") / 100.0

                prix_vente_reseau = st.number_input("Prix de vente au réseau du surplus (€/kWh)",
                    min_value=0.0, value=0.10, step=0.01, format="%.3f",
                    help="Valeur fictive. Tarif de rachat du surplus injecté au réseau — pas encore fourni par le BPU.")

                if prix_vente_reseau >= prix_ttc_moyen:
                    st.warning("Le prix de vente au réseau est supérieur ou égal au prix d'achat évité : "
                               "dans ce cas, stocker l'énergie n'a aucun intérêt économique par rapport à la "
                               "revendre directement.")

               # ==========================================
                # 3. VAN / TRI / LCOS PAR CAPACITÉ TESTÉE (utilise le prix BPU réel)
                # ==========================================
                resultats_eco = []
                for _, row in df_res_t4.iterrows():
                    cap = row["Capacité (kWh)"]
                    autoconso_totale_kwh = row["Autoconso Totale (kWh)"]
                    capex = capex_unitaire * cap + capex_fixe
                    opex_annuel = capex * opex_pct
                    indic = calculer_flux_et_indicateurs(
                        autoconso_totale_kwh, capex, opex_annuel, prix_ttc_moyen, prix_vente_reseau,
                        taux_actualisation, duree_vie_ans, degradation_pct
                    )
               
                    resultats_eco.append({
                        "Capacité (kWh)": cap, "CAPEX (€)": capex,
                        "VAN (€)": indic["van"], "TRI (%)": indic["tri"], "LCOS (€/kWh)": indic["lcos"],
                        "Payback (années)": indic["payback"], "Ratio B/C": indic["ratio_bc"],
                    })
                df_eco = pd.DataFrame(resultats_eco)
                idx_optimal = df_eco["VAN (€)"].idxmax()
                cap_optimale = df_eco.loc[idx_optimal, "Capacité (kWh)"]
                van_optimale = df_eco.loc[idx_optimal, "VAN (€)"]
                if van_optimale > 0:
                    st.success(f"### Capacité économiquement optimale : {cap_optimale:.0f} kWh "
                               f"(VAN maximale : {van_optimale:,.0f} €)")
                else:
                    st.error(f"### Aucune capacité testée n'est rentable avec ces hypothèses "
                             f"(la VAN la moins mauvaise est de {van_optimale:,.0f} € à {cap_optimale:.0f} kWh)")
                fig_eco = make_subplots(specs=[[{"secondary_y": True}]])
                fig_eco.add_trace(go.Scatter(x=df_eco["Capacité (kWh)"], y=df_eco["VAN (€)"], mode="lines+markers",
                    name="VAN (€)", fill="tozeroy", line=dict(color="green", width=3)), secondary_y=False)
                fig_eco.add_trace(go.Scatter(x=df_eco["Capacité (kWh)"], y=df_eco["TRI (%)"], mode="lines",
                    name="TRI (%)", line=dict(color="blue", width=2, dash="dash")), secondary_y=True)
                fig_eco.add_hline(y=0, line_dash="dot", line_color="red", secondary_y=False)
                fig_eco.update_layout(title="VAN et TRI en fonction de la capacité de la batterie",
                    xaxis_title="Taille de la batterie simulée (kWh)", hovermode="x unified")
                fig_eco.update_yaxes(title_text="VAN (€)", secondary_y=False)
                fig_eco.update_yaxes(title_text="TRI (%)", secondary_y=True)
                st.plotly_chart(fig_eco, use_container_width=True)

                st.subheader("Tableau récapitulatif par capacité testée")

                def fmt_eur0(x):
                    return "" if pd.isna(x) else f"{x:,.0f}"

                def fmt_num1(x):
                    return "" if pd.isna(x) else f"{x:.1f}"

                def fmt_num2(x):
                    return "" if pd.isna(x) else f"{x:.2f}"

                def fmt_num3(x):
                    return "" if pd.isna(x) else f"{x:.3f}"

                st.dataframe(df_eco.style.format({
                    "CAPEX (€)": fmt_eur0, "VAN (€)": fmt_eur0, "TRI (%)": fmt_num1,
                    "LCOS (€/kWh)": fmt_num3, "Payback (années)": fmt_num1, "Ratio B/C": fmt_num2,
                }))
                # ==========================================
                # 4. TABLEAU DÉTAILLÉ (format étude Enolab) POUR UNE CAPACITÉ DONNÉE
                # ==========================================
                st.markdown("---")
                st.subheader("Tableau de flux détaillé (format étude Enolab)")

                capacite_etude = st.number_input("Capacité de la batterie étudiée (kWh)",
                    min_value=0.0, max_value=300.0, value=250.0, step=5.0)
                ligne_capacite = df_res_t4.iloc[(df_res_t4["Capacité (kWh)"] - capacite_etude).abs().argsort()[:1]].iloc[0]
                autoconso_totale_kwh_reel = ligne_capacite["Autoconso Totale (kWh)"]

                col_v1, col_v2, col_v3 = st.columns(3)
                opex_an1_v2 = col_v1.number_input("OPEX année 1 (€ HT)", min_value=0.0, value=4600.0, step=100.0)
                taux_inflation_opex = col_v2.number_input("Inflation OPEX (%/an)", min_value=0.0, max_value=10.0,
                    value=1.5, step=0.1) / 100.0
                taux_inflation_energie = col_v3.number_input("Inflation prix électricité (%/an)", min_value=0.0,
                    max_value=10.0, value=3.0, step=0.1) / 100.0

                col_v4, col_v5 = st.columns(2)
                revenu_producteur_an1 = col_v4.number_input("Coût producteur année 1 (€)",
                    min_value=0.0, value=0.0, step=10.0,
                    help="Nature encore à définir (TURPE injection, GO/CEE sur le surplus...). "
                         "Laissé à 0 par défaut tant que ce n'est pas clarifié.")
                capex_v2 = col_v5.number_input("CAPEX total (€ HT)", min_value=0.0,
                    value=capacite_etude * 1000.0, step=1000.0,
                    help="Valeur fictive par défaut (1 000 €/kWh). À remplacer par le devis réel.")
                

                df_enolab = calculer_tableau_enolab(
                    capex=capex_v2, opex_an1=opex_an1_v2, taux_inflation_opex=taux_inflation_opex,
                    gain_net_kwh_an1=autoconso_totale_kwh_reel, prix_moyen_ttc_an1=prix_ttc_moyen,
                    taux_inflation_energie=taux_inflation_energie,
                    revenu_producteur_an1=revenu_producteur_an1, taux_inflation_revenu_producteur=taux_inflation_opex,
                    duree_vie_ans=20, degradation_pct_an=degradation_pct)

                def fmt_eur(x):
                    return "" if pd.isna(x) else f"{x:,.0f}"

                st.dataframe(df_enolab.style.format({
                    "CAPEX (€ HT)": fmt_eur, "OPEX (€ HT)": fmt_eur,
                    "Economie ACI (€ TTC)": fmt_eur, "Revenu producteur (€)": fmt_eur,
                    "Economie nette (€)": fmt_eur, "Flux cumulés (€)": fmt_eur,
                }))
else:
    st.info("Bienvenue ! Veuillez importer vos fichiers CSV ou EXCEL dans le panneau latéral pour commencer l'analyse.")