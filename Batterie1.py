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
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

    if "TAP (%)" in df_res.columns:
        taps = df_res["TAP (%)"].values.astype(float)
    elif "TAP global (%)" in df_res.columns:
        taps = df_res["TAP global (%)"].values.astype(float)
    else:
        taps = np.zeros_like(caps)

    if "TAC (%)" in df_res.columns:
        tacs = df_res["TAC (%)"].values.astype(float)
    elif "TAC global (%)" in df_res.columns:
        tacs = df_res["TAC global (%)"].values.astype(float)
    else:
        tacs = np.zeros_like(caps)

    if type_methode == "kneedle":
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
        seuil = gains.max() * param
        idx = int(np.argmax(gains >= seuil))
        return caps[idx]

    elif type_methode == "marginal_abs":
        for i in range(1, len(caps)):
            if (gains[i] - gains[i - 1]) < param:
                return caps[i - 1]
        return caps[-1]

    elif type_methode == "plateau_tap":
        seuil = taps.max() * param
        idx = int(np.argmax(taps >= seuil))
        return caps[idx]

    elif type_methode == "tac_cible":
        seuil = param * 100
        idx = int(np.argmax(tacs >= seuil))
        return caps[idx]

def classer_cadran(timestamp, structure_cadran, mois_saison_haute=None, hc_debut_haute=22, hc_fin_haute=6,
                    hc_debut_basse=22, hc_fin_basse=6):
    """
    Calendrier HP/HC et saisons — personnalisable via l'interface (sous-onglet Tarification).
    Par défaut (si non précisé) : Heures Creuses 22h-6h les deux saisons, Saison Haute novembre à mars.
    "Pointe" (C2) NON implémentée pour l'instant : les heures qui seraient en Pointe sont
    comptées comme HPSh/HPSb à la place.
    """
    if mois_saison_haute is None:
        mois_saison_haute = (11, 12, 1, 2, 3)

    heure = timestamp.hour
    mois = timestamp.month
    est_saison_haute = mois in mois_saison_haute

    if est_saison_haute:
        hc_debut, hc_fin = hc_debut_haute, hc_fin_haute
    else:
        hc_debut, hc_fin = hc_debut_basse, hc_fin_basse

    if hc_debut <= hc_fin:
        est_hc = hc_debut <= heure < hc_fin
    else:
        est_hc = (heure >= hc_debut) or (heure < hc_fin)

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

def calculer_ranges_alignes(serie1, serie2, marge=0.1):
    """
    Calcule des bornes [min, max] pour deux séries affichées sur deux axes Y
    différents d'un même graphique, de façon à ce que leur valeur zéro soit
    alignée à la même position verticale sur les deux axes.
    """
    def bornes_brutes(s):
        s = np.asarray(s, dtype=float)
        s = s[~np.isnan(s)]
        if len(s) == 0:
            return 0.0, 1.0
        mn, mx = float(np.min(s)), float(np.max(s))
        mn = min(mn, 0.0)
        mx = max(mx, 0.0)
        if mx == mn:
            mx = mn + 1.0
        return mn, mx

    mn1, mx1 = bornes_brutes(serie1)
    mn2, mx2 = bornes_brutes(serie2)

    amp1 = mx1 - mn1
    amp2 = mx2 - mn2
    mx1 += amp1 * marge
    mx2 += amp2 * marge
    if mn1 < 0:
        mn1 -= amp1 * marge
    if mn2 < 0:
        mn2 -= amp2 * marge

    f1 = (-mn1 / (mx1 - mn1)) if mn1 < 0 else 0.0
    f2 = (-mn2 / (mx2 - mn2)) if mn2 < 0 else 0.0
    f = max(f1, f2)

    if f > 0:
        mn1 = -f / (1 - f) * mx1
        mn2 = -f / (1 - f) * mx2

    return [mn1, mx1], [mn2, mx2]

def prix_moyen_pondere_ttc(series_puissance_kw, dt_heures, segment, structure_cadran,
                             inclure_go=True, accise_eur_mwh=25.0, taux_tva=0.20, turpe_dict=None,
                             mois_saison_haute=None, hc_debut_haute=22, hc_fin_haute=6,
                             hc_debut_basse=22, hc_fin_basse=6):
    """
    Prix moyen pondéré (€/kWh TTC) d'un profil de charge, pondéré par l'énergie réelle par cadran,
    avec TURPE dynamique et calendrier HP/HC personnalisable.
    """
    if turpe_dict is None:
        turpe_dict = {"Base": 0.0, "HP": 0.0, "HC": 0.0, "HPSh": 0.0, "HCSh": 0.0, "HPSb": 0.0, "HCSb": 0.0, "Pte": 0.0}

    tarif_fourniture = TARIFS_BPU[segment]["fourniture"][structure_cadran]
    tarif_capacite = TARIFS_BPU[segment]["capacite"][structure_cadran]
    cadrans = series_puissance_kw.index.map(lambda t: classer_cadran(
        t, structure_cadran, mois_saison_haute, hc_debut_haute, hc_fin_haute, hc_debut_basse, hc_fin_basse
    ))

    energie_kwh = series_puissance_kw.values * dt_heures
    df_tmp = pd.DataFrame({"cadran": cadrans, "energie_kwh": energie_kwh})
    energie_par_cadran = df_tmp.groupby("cadran")["energie_kwh"].sum()

    cout_total_htt = 0.0
    for cadran, energie in energie_par_cadran.items():
        prix_fourniture = tarif_fourniture.get(cadran, 0) / 1000.0
        prix_capacite = tarif_capacite.get(cadran, 0) / 1000.0
        prix_turpe = turpe_dict.get(cadran, 0.0)
        prix_go = (PRIX_GO / 1000.0) if inclure_go else 0.0
        prix_cee = PRIX_CEE / 1000.0
        prix_accise = accise_eur_mwh / 1000.0
        prix_htt = prix_fourniture + prix_capacite + prix_turpe + prix_go + prix_cee + prix_accise
        cout_total_htt += energie * prix_htt

    energie_totale = energie_par_cadran.sum()
    prix_moyen_htt = cout_total_htt / energie_totale if energie_totale > 0 else 0.0

    return prix_moyen_htt * (1 + taux_tva), energie_par_cadran

def prix_moyen_pondere_decharge_ttc(df_simu, dt_heures, segment_siege, cadran_siege, segment_bornes, cadran_bornes,
                                      volume_siege, volume_bornes, accise_eur_mwh, taux_tva, turpe_dict,
                                      mois_saison_haute=None, hc_debut_haute=22, hc_fin_haute=6,
                                      hc_debut_basse=22, hc_fin_basse=6):
    decharge_series = df_simu["Autoconso_Totale_kW"] - np.minimum(df_simu["conso_kW"], df_simu["prod_kW"])

    prix_decharge_siege, _ = prix_moyen_pondere_ttc(decharge_series, dt_heures, segment_siege, cadran_siege,
        True, accise_eur_mwh, taux_tva, turpe_dict, mois_saison_haute, hc_debut_haute, hc_fin_haute,
        hc_debut_basse, hc_fin_basse)
    prix_decharge_bornes, _ = prix_moyen_pondere_ttc(decharge_series, dt_heures, segment_bornes, cadran_bornes,
        True, accise_eur_mwh, taux_tva, turpe_dict, mois_saison_haute, hc_debut_haute, hc_fin_haute,
        hc_debut_basse, hc_fin_basse)

    volume_total = volume_siege + volume_bornes
    if volume_total > 0:
        return (prix_decharge_siege * volume_siege + prix_decharge_bornes * volume_bornes) / volume_total
    return prix_decharge_siege
def calculer_flux_et_indicateurs(gain_net_kwh_an1, capex, opex_annuel_an1, prix_achat_evite_an1, prix_vente_reseau,
                                 taux_actualisation, duree_vie_ans, degradation_pct_an=0.0,
                                 taux_inflation_energie=0.0, taux_inflation_opex=0.0):
    """Calcule VAN, TRI, LCOS, TRB et ratio B/C pour une capacité donnée en intégrant l'inflation."""
    annees = np.arange(0, duree_vie_ans + 1)
    recettes = np.zeros(duree_vie_ans + 1)
    opex = np.zeros(duree_vie_ans + 1)
    energies = np.zeros(duree_vie_ans + 1)
    
    for annee in range(1, duree_vie_ans + 1):
        facteur_degradation = (1 - degradation_pct_an) ** (annee - 1)
        energies[annee] = gain_net_kwh_an1 * facteur_degradation
        
        # Le prix de l'électricité économisée augmente avec l'inflation
        prix_achat_actuel = prix_achat_evite_an1 * ((1 + taux_inflation_energie) ** (annee - 1))
        marge_arbitrage = prix_achat_actuel - prix_vente_reseau
        
        recettes[annee] = energies[annee] * marge_arbitrage
        
        # L'OPEX augmente aussi avec l'inflation
        opex[annee] = opex_annuel_an1 * ((1 + taux_inflation_opex) ** (annee - 1))
        
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
                              duree_vie_ans, degradation_pct_an=0.0, prix_vente_reseau=0.0):
    lignes = [{
        "Année": "A0", "CAPEX (€ HT)": -capex, "OPEX (€ HT)": np.nan,
        "Énergie autoconsommée (kWh)": np.nan,
        "Economie ACI (€ TTC)": np.nan, "Revenu producteur (€)": np.nan,
        "Economie nette (€)": -capex, "Flux cumulés (€)": -capex,
    }]
    flux_cumule = -capex
    for annee in range(1, duree_vie_ans + 1):
        facteur_degradation = (1 - degradation_pct_an) ** (annee - 1)
        opex = -opex_an1 * (1 + taux_inflation_opex) ** (annee - 1)
        gain_kwh = gain_net_kwh_an1 * facteur_degradation
        prix_ttc = prix_moyen_ttc_an1 * (1 + taux_inflation_energie) ** (annee - 1)
        marge = prix_ttc - prix_vente_reseau
        economie_aci = gain_kwh * marge
        revenu_producteur = -revenu_producteur_an1 * (1 + taux_inflation_revenu_producteur) ** (annee - 1)
        economie_nette = economie_aci + revenu_producteur + opex
        flux_cumule += economie_nette
        lignes.append({
            "Année": f"A{annee}", "CAPEX (€ HT)": np.nan, "OPEX (€ HT)": opex,
            "Énergie autoconsommée (kWh)": gain_kwh,
            "Economie ACI (€ TTC)": economie_aci, "Revenu producteur (€)": revenu_producteur,
            "Economie nette (€)": economie_nette, "Flux cumulés (€)": flux_cumule,
        })
    return pd.DataFrame(lignes)

def carte_indicateur(titre, valeur, couleur_fond, couleur_accent, taille_titre=12, taille_valeur=20, aide=None):
    titre_html = titre
    if aide:
        aide_echap = aide.replace('"', "'")
        titre_html = f'{titre} <span title="{aide_echap}" style="cursor: help; color: #999;">&#9432;</span>'
    return f"""
    <div style="background-color:{couleur_fond}; border-left: 6px solid {couleur_accent};
                border-radius: 8px; padding: 16px 20px; margin-bottom: 8px;">
        <div style="font-size: {taille_titre}px; color: #444; font-weight: 500;">{titre_html}</div>
        <div style="font-size: {taille_valeur}px; color: {couleur_accent}; font-weight: 700; margin-top: 4px;">{valeur}</div>
    </div>
    """
    
def style_indicateur(x, favorable=False):
    if favorable:
        return "background-color: #C6EFCE; color: #006100; font-weight: 600"
    return ""

def choisir_segment_et_cadran(nom_site, key_prefix):
    st.markdown(f"**{nom_site}**")
    segment = st.selectbox(f"Segment tarifaire — {nom_site}", SEGMENTS_DISPONIBLES, key=f"segment_{key_prefix}")
    options_cadran = TARIFS_BPU[segment]["cadrans_disponibles"]
    if len(options_cadran) > 1:
        cadran = st.selectbox(f"Structure de comptage — {nom_site}", options_cadran, key=f"cadran_{key_prefix}")
    else:
        cadran = options_cadran[0]
        st.caption(f"Structure de comptage : {cadran} (seule option pour {segment}).")
    return segment, cadran

def style_van(x):
    return style_indicateur(x, favorable=(not pd.isna(x) and x > 0))

def style_tri(x):
    return style_indicateur(x, favorable=(not pd.isna(x) and x > 4))

def style_ratio_bc(x):
    return style_indicateur(x, favorable=(not pd.isna(x) and x > 1))

def style_lcos(x):
    return style_indicateur(x, favorable=False)

def style_payback(x):
    return style_indicateur(x, favorable=False)

def generer_pdf_enolab(df_enolab, capacite_etude, capex, tri_texte, lcos_texte, payback_texte, valorisation_texte):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=landscape(A4), topMargin=1.2*cm, bottomMargin=1.2*cm,
                             leftMargin=1.2*cm, rightMargin=1.2*cm,
                             title=f"Bilan financier — Batterie {capacite_etude:.0f} kWh")
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph(f"Bilan financier — Batterie {capacite_etude:.0f} kWh", styles["Title"]))
    elements.append(Spacer(1, 0.3*cm))
    elements.append(Paragraph(f"CAPEX : {capex:,.0f} € HT &nbsp;&nbsp;|&nbsp;&nbsp; TRI : {tri_texte} "
                               f"&nbsp;&nbsp;|&nbsp;&nbsp; LCOE : {lcos_texte} &nbsp;&nbsp;|&nbsp;&nbsp; "
                               f"TRB : {payback_texte} &nbsp;&nbsp;|&nbsp;&nbsp; Valorisation interne : {valorisation_texte}",
                               styles["Normal"]))
    elements.append(Spacer(1, 0.5*cm))

    colonnes = ["Année", "CAPEX (€ HT)", "OPEX (€ HT)", "Énergie autoconsommée (kWh)",
                "Economie ACI (€ TTC)", "Revenu producteur (€)", "Economie nette (€)", "Flux cumulés (€)"]
    data = [colonnes]
    for _, row in df_enolab.iterrows():
        ligne = [row["Année"]]
        for col in colonnes[1:]:
            val = row[col]
            ligne.append("" if pd.isna(val) else f"{val:,.0f}")
        data.append(ligne)

    table = Table(data, repeatRows=1)
    style_cmds = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F3864")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#DDDDDD")),
        ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7F9FC")]),
    ]
    idx_flux = colonnes.index("Flux cumulés (€)")
    for i, (_, row) in enumerate(df_enolab.iterrows(), start=1):
        val = row["Flux cumulés (€)"]
        if pd.notna(val):
            couleur = colors.HexColor("#C6EFCE") if val >= 0 else colors.HexColor("#FFC7CE")
            style_cmds.append(("BACKGROUND", (idx_flux, i), (idx_flux, i), couleur))
    table.setStyle(TableStyle(style_cmds))
    elements.append(table)

    doc.build(elements)
    buffer.seek(0)
    return buffer

def generer_png_enolab(df_enolab, capacite_etude, capex, tri_texte, lcos_texte, payback_texte, valorisation_texte):
    colonnes = ["Année", "CAPEX (€ HT)", "OPEX (€ HT)", "Énergie autoconsommée (kWh)",
                "Economie ACI (€ TTC)", "Revenu producteur (€)", "Economie nette (€)", "Flux cumulés (€)"]

    data = []
    for _, row in df_enolab.iterrows():
        ligne = [row["Année"]]
        for col in colonnes[1:]:
            val = row[col]
            ligne.append("" if pd.isna(val) else f"{val:,.0f}")
        data.append(ligne)

    n_rows = len(data) + 1
    n_cols = len(colonnes)
    fig_height = 0.32 * n_rows + 1.4

    fig, ax = plt.subplots(figsize=(15, fig_height))
    ax.axis("off")
    ax.set_title(f"Bilan financier — Batterie {capacite_etude:.0f} kWh", fontsize=15, fontweight="bold", pad=28)
    fig.text(0.5, 1 - 1.0 / fig_height,
              f"CAPEX : {capex:,.0f} € HT   |   TRI : {tri_texte}   |   LCOE : {lcos_texte}   |   "
              f"TRB : {payback_texte}   |   Valorisation interne : {valorisation_texte}",
              ha="center", fontsize=9.5, color="#444444")

    table = ax.table(cellText=data, colLabels=colonnes, loc="center", cellLoc="right")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.5)

    idx_flux = colonnes.index("Flux cumulés (€)")
    for j in range(n_cols):
        table[0, j].set_facecolor("#1F3864")
        table[0, j].set_text_props(color="white", fontweight="bold")

    for i, (_, row) in enumerate(df_enolab.iterrows(), start=1):
        for j in range(n_cols):
            if j != idx_flux:
                table[i, j].set_facecolor("#F7F9FC" if i % 2 == 0 else "white")
        val = row["Flux cumulés (€)"]
        if pd.notna(val):
            couleur = "#C6EFCE" if val >= 0 else "#FFC7CE"
            table[i, idx_flux].set_facecolor(couleur)

    buffer = io.BytesIO()
    plt.savefig(buffer, format="png", dpi=200, bbox_inches="tight",
                metadata={"Title": f"Bilan financier — Batterie {capacite_etude:.0f} kWh"})
    plt.close(fig)
    buffer.seek(0)
    return buffer
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

# ==========================================
# TURPE (Acheminement) — valeurs FICTIVES par défaut, à remplacer par la grille CRE réelle du TE13
# ==========================================
TARIFS_TURPE = {
    "HPSh": 0.02130,  # €/kWh — Heures Pleines Saison Haute
    "HCSh": 0.01520,  # €/kWh — Heures Creuses Saison Haute
    "HPSb": 0.06910,  # €/kWh — Heures Pleines Saison Basse
    "HCSb": 0.04210,  # €/kWh — Heures Creuses Saison Basse
}

DUREE_VIE_MAX_ANS = 25  # limite réaliste (durée de vie calendaire), même si le cycle life théorique calculé est plus long

ACCISE_EUR_KWH = 0.03085  # €/kWh 
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
    # --- Sélection de la période ---
        date_min = df_complet.index.min().date()
        date_max = df_complet.index.max().date()

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
            st.subheader("Période d'analyse")
            col_date1, col_date2 = st.columns(2)
            date_debut = col_date1.date_input("Date de début", value=date_min, min_value=date_min, max_value=date_max, format="DD/MM/YYYY", key="date_debut_t1")
            date_fin = col_date2.date_input("Date de fin (incluse)", value=date_min, min_value=date_min, max_value=date_max, format="DD/MM/YYYY", key="date_fin_t1")
            mask = (df_complet.index.date >= date_debut) & (df_complet.index.date <= date_fin)
            df = df_complet.loc[mask]
            if df.empty:
                st.error("Aucune donnée trouvée pour les dates sélectionnées.")

            col_bat1, col_bat2 = st.columns(2)
            capacite_batterie = col_bat1.slider("Capacité (kWh)", min_value=0.0, max_value=500.0, value=50.0, step=1.0, help="Volume total d'énergie stockable.")
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

            with col_kpi1:
                st.markdown(carte_indicateur("Taux d'Autoconso. (TAC)", f"{nouveau_tac:.1f} %",
                    "#E3F2FD", "#1565C0"), unsafe_allow_html=True)
            with col_kpi2:
                st.markdown(carte_indicateur("Taux d'Autoprod. (TAP)", f"{nouveau_tap:.1f} %",
                    "#FFF3E0", "#E65100"), unsafe_allow_html=True)
            with col_kpi3:
                st.markdown(carte_indicateur("Énergie totale économisée (gain net)", f"{gain_net:.1f} kWh",
                    "#E8F5E9", "#2E7D32",
                    aide=f"Gain net apporté par la batterie par rapport à une installation sans stockage, "
                         f"qui autoconsommerait naturellement {autoconso_sans_bat:.1f} kWh sur cette période."
                    ), unsafe_allow_html=True)
            
            st.markdown("---")
            st.subheader("Comparaison des courbes de charge")
            col_cb1, col_cb2, col_cb3 = st.columns(3)
            afficher_bornes_seules = col_cb1.checkbox("Charge Bornes", value=False, key="cb_bornes_t1")
            afficher_siege_seul = col_cb2.checkbox("Charge Siège", value=False, key="cb_siege_t1")
            afficher_total = col_cb3.checkbox("Siège + Bornes", value=True, key="cb_total_t1")

            if not (afficher_bornes_seules or afficher_siege_seul or afficher_total):
                st.info("Cochez au moins une case ci-dessus pour afficher une courbe.")
            else:
                fig_comp = go.Figure()
                if afficher_bornes_seules and "conso_bornes_kW" in df_simu.columns:
                    fig_comp.add_trace(go.Scatter(x=df.index, y=df_simu["conso_bornes_kW"], mode="lines",
                        name="Charge Bornes (kW)", line=dict(color="#E63946", width=2)))
                if afficher_siege_seul:
                    conso_siege_seule_t1 = (df_simu["conso_kW"] - df_simu["conso_bornes_kW"]
                                              if "conso_bornes_kW" in df_simu.columns else df_simu["conso_kW"])
                    fig_comp.add_trace(go.Scatter(x=df.index, y=conso_siege_seule_t1, mode="lines",
                        name="Charge Siège (kW)", line=dict(color="royalblue", width=2)))
                if afficher_total:
                    fig_comp.add_trace(go.Scatter(x=df.index, y=df_simu["conso_kW"], mode="lines",
                        name="Siège + Bornes (kW)", line=dict(color="#2A9D8F", width=2)))

                fig_comp.update_layout(
                    title="Comparaison des courbes de charge",
                    xaxis_title="Période temporelle", yaxis_title="Puissance (kW)",
                    hovermode="x unified",
                    legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5),
                )
                fig_comp.update_xaxes(type="date", tickformat="%H:%M\n%d/%m", hoverformat="%d/%m/%Y %H:%M")
                st.plotly_chart(fig_comp, use_container_width=True)
            
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
            
            st.subheader("Période d'analyse")
            col_date1_t2, col_date2_t2 = st.columns(2)
            date_debut = col_date1_t2.date_input("Date de début", value=date_min, min_value=date_min, max_value=date_max, format="DD/MM/YYYY", key="date_debut_t2")
            date_fin = col_date2_t2.date_input("Date de fin (incluse)", value=date_min, min_value=date_min, max_value=date_max, format="DD/MM/YYYY", key="date_fin_t2")
            mask = (df_complet.index.date >= date_debut) & (df_complet.index.date <= date_fin)
            df = df_complet.loc[mask]
            if df.empty:
                st.error("Aucune donnée trouvée pour les dates sélectionnées.")

            if (date_fin - date_debut).days < 30:
                st.warning("La période sélectionnée fait moins d'un mois. Pour une vision détaillée, "
                           "privilégiez plutôt l'onglet « Simulation Temporelle ».")

            col_bat1_ld, col_bat2_ld = st.columns(2)
            capacite_batterie_ld = col_bat1_ld.slider(
                "Capacité (kWh)", min_value=0.0, max_value=500.0, value=50.0, step=1.0,
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
            col_kpi1_ld, col_kpi2_ld, col_kpi3_ld, col_kpi4_ld = st.columns(4)

            conso_totale_ld = df_simu_ld["conso_kW"].sum() * dt_ld
            prod_totale_ld = df_simu_ld["prod_kW"].sum() * dt_ld
            export_totale_ld = df_simu_ld["Export_Reseau_kW"].sum() * dt_ld
            autoconso_totale_ld = df_simu_ld["Autoconso_Totale_kW"].sum() * dt_ld

            autoconso_sans_bat_ld = np.minimum(df_simu_ld["conso_kW"], df_simu_ld["prod_kW"]).sum() * dt_ld
            gain_net_ld = max(0, autoconso_totale_ld - autoconso_sans_bat_ld)

            energie_pv_valorisee_ld = prod_totale_ld - export_totale_ld
            tac_ld = (energie_pv_valorisee_ld / prod_totale_ld * 100) if prod_totale_ld > 0 else 0
            tap_ld = (autoconso_totale_ld / conso_totale_ld * 100) if conso_totale_ld > 0 else 0

            cycles_periode_ld = (gain_net_ld / capacite_batterie_ld) if capacite_batterie_ld > 0 else 0.0

            with col_kpi1_ld:
                st.markdown(carte_indicateur("Taux d'Autoconso. (TAC)", f"{tac_ld:.1f} %",
                    "#E3F2FD", "#1565C0"), unsafe_allow_html=True)
            with col_kpi2_ld:
                st.markdown(carte_indicateur("Taux d'Autoprod. (TAP)", f"{tap_ld:.1f} %",
                    "#FFF3E0", "#E65100"), unsafe_allow_html=True)
            with col_kpi3_ld:
                st.markdown(carte_indicateur("Énergie totale économisée (gain net)", f"{gain_net_ld:.1f} kWh",
                    "#E8F5E9", "#2E7D32",
                    aide=f"Gain net apporté par la batterie par rapport à une installation sans stockage, "
                         f"qui autoconsommerait naturellement {autoconso_sans_bat_ld:.1f} kWh sur cette période."
                    ), unsafe_allow_html=True)
            with col_kpi4_ld:
                st.markdown(carte_indicateur("Cycles sur la période", f"{cycles_periode_ld:.1f}",
                    "#F3E5F5", "#6A1B9A",
                    aide="Cycles équivalents pleine charge réalisés sur la période sélectionnée : "
                         "énergie déchargée par la batterie ÷ capacité."
                    ), unsafe_allow_html=True)
        
                
            st.markdown("---")
            st.subheader("Comparaison des courbes de charge (pics quotidiens)")
            col_cb1_ld, col_cb2_ld, col_cb3_ld = st.columns(3)
            afficher_bornes_seules_ld = col_cb1_ld.checkbox("Charge Bornes", value=False, key="cb_bornes_t2")
            afficher_siege_seul_ld = col_cb2_ld.checkbox("Charge Siège", value=False, key="cb_siege_t2")
            afficher_total_ld = col_cb3_ld.checkbox("Siège + Bornes", value=True, key="cb_total_t2")

            if not (afficher_bornes_seules_ld or afficher_siege_seul_ld or afficher_total_ld):
                st.info("Cochez au moins une case ci-dessus pour afficher une courbe.")
            else:
                fig_comp_ld = go.Figure()
                if afficher_bornes_seules_ld and "conso_bornes_kW" in df_simu_ld.columns:
                    bornes_pic_j = df_simu_ld.groupby(jours)["conso_bornes_kW"].max()
                    bornes_pic_j.index = pd.to_datetime(bornes_pic_j.index)
                    fig_comp_ld.add_trace(go.Scatter(x=bornes_pic_j.index, y=bornes_pic_j.values, mode="lines",
                        name="Pic Charge Bornes (kW/j)", line=dict(color="#E63946", width=2)))
                if afficher_siege_seul_ld:
                    conso_siege_seule_ld = (df_simu_ld["conso_kW"] - df_simu_ld["conso_bornes_kW"]
                                              if "conso_bornes_kW" in df_simu_ld.columns else df_simu_ld["conso_kW"])
                    siege_pic_j = conso_siege_seule_ld.groupby(jours).max()
                    siege_pic_j.index = pd.to_datetime(siege_pic_j.index)
                    fig_comp_ld.add_trace(go.Scatter(x=siege_pic_j.index, y=siege_pic_j.values, mode="lines",
                        name="Pic Charge Siège (kW/j)", line=dict(color="royalblue", width=2)))
                if afficher_total_ld:
                    fig_comp_ld.add_trace(go.Scatter(x=conso_pic_j.index, y=conso_pic_j.values, mode="lines",
                        name="Pic Siège + Bornes (kW/j)", line=dict(color="#2A9D8F", width=2)))

                fig_comp_ld.update_layout(
                    title="Comparaison des pics quotidiens de charge",
                    xaxis_title="Jour", yaxis_title="Puissance (kW)",
                    hovermode="x unified",
                    legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5),
                )
                fig_comp_ld.update_xaxes(type="date")
                st.plotly_chart(fig_comp_ld, use_container_width=True)
                
                st.markdown("---")
            st.subheader("État de charge à une heure fixe, jour par jour sur l'année")

            heure_selectionnee = st.time_input(
                "Heure de la journée à observer",
                value=pd.Timestamp("12:00").time(),
                key="heure_soc_annuelle",
                help="L'état de charge de la batterie sera extrait à cette heure précise, chaque jour de "
                     "la période sélectionnée. Si l'heure choisie ne tombe pas exactement sur un pas de "
                     "30 min des données, l'horodatage le plus proche est utilisé automatiquement."
            )

            minutes_cible = heure_selectionnee.hour * 60 + heure_selectionnee.minute
            minutes_donnees = df_simu_ld.index.hour * 60 + df_simu_ld.index.minute
            ecart_brut = np.abs(minutes_donnees - minutes_cible)
            ecart_minutes = np.minimum(ecart_brut, 1440 - ecart_brut)  # gère le passage minuit

            df_temp_heure = pd.DataFrame({
                "date": df_simu_ld.index,
                "jour": df_simu_ld.index.date,
                "SoC_pourcent": df_simu_ld["SoC_pourcent"].values,
                "ecart_minutes": ecart_minutes,
            })
            idx_plus_proche = df_temp_heure.groupby("jour")["ecart_minutes"].idxmin()
            soc_heure_fixe = df_temp_heure.loc[idx_plus_proche].set_index("date")["SoC_pourcent"].sort_index()

            fig_soc_heure = go.Figure()
            fig_soc_heure.add_trace(go.Scatter(
                x=soc_heure_fixe.index, y=soc_heure_fixe.values, mode="lines",
                name=f"État de charge à {heure_selectionnee.strftime('%H:%M')}",
                line=dict(color="purple", width=2),
                fill="tozeroy", fillcolor="rgba(128, 0, 128, 0.1)"
            ))
            fig_soc_heure.update_layout(
                title=f"État de charge de la batterie à {heure_selectionnee.strftime('%H:%M')}, jour par jour",
                xaxis_title="Jour", yaxis_title="État de charge (%)",
                hovermode="x unified"
            )
            fig_soc_heure.update_yaxes(range=[0, 105])
            fig_soc_heure.update_xaxes(type="date")
            st.plotly_chart(fig_soc_heure, use_container_width=True)
        # ----------------------------------------------------
        # ONGLET 3 :Etude du Gain de la batterie 
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
            
            st.subheader("Période d'analyse")
            col_date1_t3, col_date2_t3 = st.columns(2)
            date_debut = col_date1_t3.date_input("Date de début", value=date_min, min_value=date_min, max_value=date_max, format="DD/MM/YYYY", key="date_debut_t3")
            date_fin = col_date2_t3.date_input("Date de fin (incluse)", value=date_min, min_value=date_min, max_value=date_max, format="DD/MM/YYYY", key="date_fin_t3")
            mask = (df_complet.index.date >= date_debut) & (df_complet.index.date <= date_fin)
            df = df_complet.loc[mask]
            if df.empty:
                st.error("Aucune donnée trouvée pour les dates sélectionnées.")
            
            max_cap_test = 500
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

                max_cap_test_t4 = 500
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
                            cycles_par_an_t4 = (gain_batterie_t4 / cap) if cap > 0 else 0.0

                            resultats_t4.append({
                                "Capacité (kWh)": cap,
                                "Autoconso Totale (kWh)": autoconso_tot_t4,
                                "Gain Énergétique (kWh)": gain_batterie_t4,
                                "Cycles par an": cycles_par_an_t4,
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
                        "Atteindre un TAC de 99 %": ("tac_cible", 0.99),
                        "Atteindre un TAC de 95 %": ("tac_cible", 0.95),
                        "Atteindre un TAC de 90 %": ("tac_cible", 0.90),
                        "Atteindre un TAC de 80 %": ("tac_cible", 0.80),
                        "Atteindre le plateau à 90 % du TAP maximal possible": ("plateau_tap", 0.90),
                        "Atteindre le plateau à 95 % du TAP maximal possible": ("plateau_tap", 0.95),
                    }
                    
                    

# Texte affiché au survol du petit "?" à côté de chaque case
                    aide_methodes = {
                        "Coude géométrique de rentabilité":
                            "Détecte automatiquement le point d'inflexion (le « coude ») de la courbe de gain : "
                            "la capacité où l'écart entre la courbe et la droite reliant le premier et le dernier "
                            "point testés (0 et 500 kWh) est maximal. Méthode géométrique (Kneedle), sans seuil à choisir.",
                        "Atteindre le plateau à 90 % du gain net maximal":
                            "Retient la plus petite capacité testée qui atteint déjà 90 % du gain énergétique "
                            "maximal observé sur toute la plage testée (0 à 500 kWh).",
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
                            "Dès qu'un palier rapporte moins que ce seuil, l'algorithme s'arrête et garde la capacité juste avant."
                            "Conduit généralement à une capacité recommandée plus élevée que les deux options précédentes. ",
                        "Atteindre un TAC de 99 %":
                            "Retient la plus petite capacité testée pour laquelle le Taux d'Autoconsommation "
                            "(TAC) atteint réellement 99 % — une valeur cible absolue, pas relative à un maximum "
                            "observé. Le seuil le plus strict des quatre, conduit à la capacité recommandée la "
                            "plus élevée de ce groupe. Si aucune capacité testée n'atteint 99 %, retient la plus "
                            "petite capacité (0 kWh) par défaut.",
                        "Atteindre un TAC de 95 %":
                            "Même principe, cible un TAC de 95 % exactement.",
                        "Atteindre un TAC de 90 %":
                            "Même principe, cible un TAC de 90 % exactement.",
                        "Atteindre un TAC de 80 %":
                            "Même principe, cible un TAC de 80 % exactement — le plus souple des quatre, "
                            "conduit généralement à la capacité recommandée la plus faible de ce groupe.",
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
                        with col_res1:
                            st.markdown(carte_indicateur("Capacité retenue", f"{cap_ideale_finale:.0f} kWh",
                                "#E3F2FD", "#0D47A1"), unsafe_allow_html=True)
                        with col_res2:
                            st.markdown(carte_indicateur("Gain net annuel", f"{ligne_ideale['Gain Énergétique (kWh)']:.0f} kWh",
                                "#E8F5E9", "#2E7D32"), unsafe_allow_html=True)
                        with col_res3:
                            st.markdown(carte_indicateur("TAP estimé", f"{ligne_ideale['TAP (%)']:.1f} %",
                                "#E3F2FD", "#1565C0"), unsafe_allow_html=True)
                        with col_res4:
                            st.markdown(carte_indicateur("TAC estimé", f"{ligne_ideale['TAC (%)']:.1f} %",
                                "#FFF3E0", "#E65100"), unsafe_allow_html=True)
                    else:
                        st.warning("Veuillez cocher au moins une hypothèse technique ci-dessus.")
  
        # ----------------------------------------------------
        # ONGLET 5 : Analyse Budgétaire
        # ----------------------------------------------------
                        
        with tab5:
            st.header("Analyse Économique ", help=(
                "**Comment lire cet onglet :** il se déroule en 4 étapes, dans les sous-onglets ci-dessous.\n\n"
                "1. **Tarification** — calcule le prix réel de l'électricité évitée (€/kWh), à partir du BPU Octopus, du TURPE et des taxes.\n\n"
                "2. **Hypothèses** — les paramètres d'investissement (CAPEX, OPEX, taux d'actualisation, dégradation...), à ajuster.\n\n"
                "3. **Etude de la capacité** — VAN, TRI et LCOE pour chaque capacité testée, afin d'identifier la capacité économiquement optimale.\n\n"
                "4. **Bilan économique** — un plan de trésorerie année par année, pour une capacité choisie individuellement."
            ))
            st.markdown("""
            Cet onglet valorise financièrement le gain énergétique de la batterie (calculé dans l'onglet 4)
            pour déterminer si — et à quelle capacité — l'investissement est rentable.
            """)

            if "df_resultats_t4" not in st.session_state:
                st.warning("Merci de d'abord lancer l'analyse annuelle dans l'onglet 4 : cet onglet réutilise "
                           "directement son résultat (gain énergétique par capacité testée).")
            else:
                df_res_t4 = st.session_state["df_resultats_t4"]
                df = df_complet
                st.markdown("---")


                # ==========================================================
                # SOUS-ONGLETS
                # ==========================================================
                sous_tab1, sous_tab2, sous_tab3, sous_tab4 = st.tabs([
                    "1. Tarification", "2. Hypothèses", "3. Comparaison des capacités", "4. Bilan Financier"
                ])

                # ----------------------------------------------------------------
                # SOUS-ONGLET 1 : TARIFICATION
                # ----------------------------------------------------------------
                with sous_tab1:
                    st.caption("Le prix payé se décompose en 3 familles de coûts, additionnées pour obtenir "
                               "le prix complet évité : la fourniture (BPU), l'acheminement (TURPE, "
                               "Enedis) et les taxes.")

                    st.markdown("##### Fourniture — BPU Octopus Energy 2026")
                    SEGMENTS_DISPONIBLES = ["C5 - Bâtiments et équipements", "C4", "C2"]

                    col_t1, col_t2 = st.columns(2)
                    with col_t1:
                        segment_siege, cadran_siege = choisir_segment_et_cadran("Siège", "siege")
                    with col_t2:
                        segment_bornes, cadran_bornes = choisir_segment_et_cadran("Bornes de recharge", "bornes")

                    aide_cadrans_generique = {
                        "Base": "Tarif unique, valable à toute heure et toute saison.",
                        "HP": "Heures Pleines (6h-22h), toutes saisons confondues.",
                        "HC": "Heures Creuses (22h-6h), toutes saisons confondues.",
                        "HPSh": "Heures Pleines, Saison Haute (6h-22h, novembre à mars).",
                        "HCSh": "Heures Creuses, Saison Haute (22h-6h, novembre à mars).",
                        "HPSb": "Heures Pleines, Saison Basse (6h-22h, avril à octobre).",
                        "HCSb": "Heures Creuses, Saison Basse (22h-6h, avril à octobre).",
                        "Pte": "Pointe fixe (période de forte tension sur le réseau).  Non appliquée "
                               "dans le calcul actuel — la plage horaire de Pointe n'est pas encore "
                               "définie, ces instants sont comptés comme Heures Pleines à la place.",
                    }

                    with st.expander("Détail Fourniture (BPU) par segment tarifaire"):
                        st.caption("Prix de l'électricité elle-même (hors acheminement et taxes), facturé "
                                   "par Octopus Energy selon le BPU.")

                        combinaisons_fourniture = {}
                        for nom_site, segment, cadran in [("Siège", segment_siege, cadran_siege),
                                                            ("Bornes", segment_bornes, cadran_bornes)]:
                            cle = (segment, cadran)
                            combinaisons_fourniture.setdefault(cle, []).append(nom_site)

                        for (segment, cadran), sites in combinaisons_fourniture.items():
                            st.markdown(f"**{' & '.join(sites)}** — {segment}, {cadran}")
                            tarif_fourniture = TARIFS_BPU[segment]["fourniture"][cadran]
                            cadrans_du_segment = list(tarif_fourniture.keys())
                            col_fourniture = st.columns(len(cadrans_du_segment))
                            for i, c in enumerate(cadrans_du_segment):
                                with col_fourniture[i]:
                                    st.markdown(carte_indicateur(c, f"{tarif_fourniture[c]:.2f} €/MWh",
                                        "#F5F5F5", "#616161", taille_titre=11, taille_valeur=13,
                                        aide=aide_cadrans_generique.get(c, "")), unsafe_allow_html=True)
                            if segment == "C2":
                                st.caption(" Le tarif Pointe ci-dessus est affiché à titre informatif, mais "
                                           "n'est pas encore pris en compte dans le calcul du prix évité "
                                           "(voir l'infobulle du cadran « Pte »).")

                    with st.expander("Détail Acheminement (TURPE) et composantes fixes du BPU"):
                        st.markdown("**TURPE (€/kWh)**")
                        st.caption("Tarif d'Utilisation des Réseaux Publics d'Électricité : rémunère Enedis "
                                   "pour l'usage du réseau de distribution. Fixé par la CRE, pas par le "
                                   "fournisseur — s'ajoute au prix de fourniture, quel que soit le fournisseur.")
                        col_turpe = st.columns(4)
                        for i, cadran in enumerate(["HPSh", "HCSh", "HPSb", "HCSb"]):
                            with col_turpe[i]:
                                st.markdown(carte_indicateur(cadran, f"{TARIFS_TURPE[cadran]:.5f}",
                                    "#F5F5F5", "#616161", taille_titre=11, taille_valeur=14,
                                    aide=aide_cadrans_generique[cadran]), unsafe_allow_html=True)

                        st.markdown("**Composantes fixes du BPU**")
                        col_fix1, col_fix2 = st.columns(2)
                        with col_fix1:
                            st.markdown(carte_indicateur("Garanties d'Origine (GO)", f"{PRIX_GO:.2f} €/MWh",
                                "#F5F5F5", "#616161", taille_titre=11, taille_valeur=14,
                                aide="Certificat attestant que l'électricité fournie provient de sources "
                                     "renouvelables. Coût additionnel fixe (indépendant du cadran horaire), "
                                     "toujours inclus dans le calcul du prix évité."
                                ), unsafe_allow_html=True)
                        with col_fix2:
                            st.markdown(carte_indicateur("Obligations CEE", f"{PRIX_CEE:.2f} €/MWh",
                                "#F5F5F5", "#616161", taille_titre=11, taille_valeur=14,
                                aide="Certificats d'Économies d'Énergie : obligation réglementaire imposée "
                                     "aux fournisseurs de financer des actions de réduction de consommation, "
                                     "répercutée sur le prix de vente. Coût additionnel fixe, toujours inclus."
                                ), unsafe_allow_html=True)

                    turpe_dict = TARIFS_TURPE
                    
                    with st.expander("Calendrier Heures Pleines / Heures Creuses"):
                        st.caption("Détermine comment chaque instant de vos données est classé dans son cadran "
                                   "tarifaire — à ajuster selon le contrat Enedis.")

                        mois_labels = ["Janvier","Février","Mars","Avril","Mai","Juin","Juillet","Août",
                                       "Septembre","Octobre","Novembre","Décembre"]
                        mois_numeros = {label: i + 1 for i, label in enumerate(mois_labels)}
                        mois_defaut_haute = ["Novembre","Décembre","Janvier","Février","Mars"]

                        col_cal1, col_cal2 = st.columns(2)
                        with col_cal1:
                            mois_saison_haute_labels = st.multiselect("Mois en Saison Haute (hiver)", mois_labels,
                                default=mois_defaut_haute, key="mois_saison_haute")
                        with col_cal2:
                            mois_saison_basse_affichage = [m for m in mois_labels if m not in mois_saison_haute_labels]
                            st.caption("Mois en Saison Basse (été), déduits automatiquement :")
                            st.write(", ".join(mois_saison_basse_affichage) if mois_saison_basse_affichage else "Aucun")

                        mois_saison_haute_num = tuple(mois_numeros[m] for m in mois_saison_haute_labels)

                        col_hc1, col_hc2 = st.columns(2)
                        with col_hc1:
                            st.markdown("**Heures Creuses — Saison Haute (hiver)**")
                            heure_debut_hc_haute = st.number_input("Début (h)", min_value=0, max_value=23, value=22,
                                key="hc_debut_haute")
                            heure_fin_hc_haute = st.number_input("Fin (h)", min_value=0, max_value=23, value=6,
                                key="hc_fin_haute")
                        with col_hc2:
                            st.markdown("**Heures Creuses — Saison Basse (été)**")
                            heure_debut_hc_basse = st.number_input("Début (h)", min_value=0, max_value=23, value=22,
                                key="hc_debut_basse")
                            heure_fin_hc_basse = st.number_input("Fin (h)", min_value=0, max_value=23, value=6,
                                key="hc_fin_basse")
                    
                    st.markdown("##### Taxes")
                    col_t4, col_t5 = st.columns(2)
                    with col_t4:
                        st.markdown(carte_indicateur("Accise électricité", f"{ACCISE_EUR_KWH:.5f} €/kWh",
                            "#F5F5F5", "#616161", taille_titre=11, taille_valeur=14,
                            aide="taxe qui s’ajoute sur les factures d’électricité des usagers en France. "
                            "Elle est notamment destinée à dédommager les opérateurs des divers surcoûts qu’ils supportent et à financer "
                            "les politiques de soutien aux énergies renouvelables."
                            ), unsafe_allow_html=True)
                    taux_tva = col_t5.number_input("TVA (%)", min_value=0.0, max_value=25.0, value=20.0, step=0.1,
                        key="taux_tva_input") / 100.0

                    accise_eur_mwh = ACCISE_EUR_KWH * 1000.0

                    dt_actuel = (df.index[1] - df.index[0]).total_seconds() / 3600.0
                    conso_siege_seule = df["conso_kW"] - df["conso_bornes_kW"] if "conso_bornes_kW" in df.columns else df["conso_kW"]

                    prix_ttc_siege, _ = prix_moyen_pondere_ttc(conso_siege_seule, dt_actuel, segment_siege, cadran_siege,
                        True, accise_eur_mwh, taux_tva, turpe_dict,
                        mois_saison_haute_num, heure_debut_hc_haute, heure_fin_hc_haute,
                        heure_debut_hc_basse, heure_fin_hc_basse)
                    if "conso_bornes_kW" in df.columns and df["conso_bornes_kW"].sum() > 0:
                        prix_ttc_bornes, _ = prix_moyen_pondere_ttc(df["conso_bornes_kW"], dt_actuel, segment_bornes,
                            cadran_bornes, True, accise_eur_mwh, taux_tva, turpe_dict,
                            mois_saison_haute_num, heure_debut_hc_haute, heure_fin_hc_haute,
                            heure_debut_hc_basse, heure_fin_hc_basse)
                    else:
                        prix_ttc_bornes = prix_ttc_siege

                    volume_siege = conso_siege_seule.sum() * dt_actuel
                    volume_bornes = df["conso_bornes_kW"].sum() * dt_actuel if "conso_bornes_kW" in df.columns else 0
                    volume_total = volume_siege + volume_bornes
                    prix_ttc_moyen = ((prix_ttc_siege * volume_siege + prix_ttc_bornes * volume_bornes) / volume_total
                                       if volume_total > 0 else prix_ttc_siege)

                    st.markdown("##### Résultat : prix complet évité")
                    col_p1, col_p2 = st.columns(2)
                    with col_p1:
                        st.markdown(carte_indicateur("Siège", f"{prix_ttc_siege:.4f} €/kWh",
                            "#E3F2FD", "#1565C0"), unsafe_allow_html=True)
                    with col_p2:
                        st.markdown(carte_indicateur("Bornes", f"{prix_ttc_bornes:.4f} €/kWh",
                            "#FFF3E0", "#E65100"), unsafe_allow_html=True)

                    st.markdown(carte_indicateur("Prix moyen pondéré global (évité)", f"{prix_ttc_moyen:.4f} €/kWh",
                        "#E8F5E9", "#2E7D32", taille_titre=14, taille_valeur=32), unsafe_allow_html=True)

                # ----------------------------------------------------------------
                # SOUS-ONGLET 2 : HYPOTHÈSES ÉCONOMIQUES
                # ----------------------------------------------------------------
                with sous_tab2:

                    with st.container(border=True):
                        st.markdown("##### Investissement")
                        col_e1, col_e2, col_e3 = st.columns(3)
                        
                        capex_unitaire = col_e1.number_input("Coût unitaire batterie (€/kWh)", min_value=0.0,
                            value=400.0, step=10.0, key="capex_unitaire_input")
                        capex_fixe = col_e2.number_input("Coûts fixes d'installation (€)", min_value=0.0,
                            value=15000.0, step=500.0, key="capex_fixe_input")
                        nombre_cycles_nominal = col_e3.number_input("Nombre de cycles nominal (garantie fabricant)",
                            min_value=100, max_value=20000, value=6000, step=100, key="nombre_cycles_input",
                            help= "**Comment :** cycles/an = énergie déchargée sur l'année (kWh) ÷ capacité de la "
                                "batterie (kWh). C'est la méthode des « cycles équivalents pleine charge » : "
                                "chaque petite décharge compte comme une fraction de cycle, qui s'additionne aux "
                                "autres au fil de l'année — inutile que la batterie aille jusqu'à 0 % puis 100 % "
                                "pour qu'un cycle « complet » soit compté.\n\n"
                                "**Pourquoi cette méthode :** une décharge de 20 % de la capacité compte pour "
                                "0,2 cycle ; dix décharges de 20 % équivalent à 2 cycles complets, peu importe "
                                "l'ordre ou la taille de chaque décharge — seul le total compte. C'est exactement "
                                "la convention qu'utilisent les fabricants pour établir leur propre garantie "
                                "(« 6 000 cycles » signifie 6 000 cycles équivalents pleine charge, pas 6 000 "
                                "vidages complets), donc comparer notre calcul à leur nombre de cycles nominal "
                                "est cohérent.\n\n"
                                "**Limite à connaître :** ceci suppose une usure uniforme par kWh traversé, quelle "
                                "que soit la profondeur de chaque décharge — une simplification en l'absence de "
                                "courbe de dégradation détaillée du fabricant, mais c'est aussi celle qu'ils "
                                "utilisent eux-mêmes pour convertir un usage réel en équivalent cycles.\n\n"
                                f"La durée de vie en années est recalculée automatiquement pour chaque capacité à "
                                f"partir de son propre nombre de cycles réalisés par an, plafonnée à "
                                f"{DUREE_VIE_MAX_ANS} ans (durée de vie calendaire réaliste, même si le cycle "
                                "life théorique calculé est plus long).")
                      

                    with st.container(border=True):
                        st.markdown("##### Exploitation")
                        col_e4, col_e5, col_e6 = st.columns(3)
                        opex_pct = col_e4.number_input("OPEX annuel (% du CAPEX)", min_value=0.0, max_value=20.0,
                            value=1.5, step=0.1, key="opex_pct_input") / 100.0
                        taux_inflation_opex = col_e5.number_input("Inflation OPEX (%/an)", min_value=0.0,
                            max_value=10.0, value=1.5, step=0.1, key="taux_inflation_opex_input") / 100.0
                        degradation_pct = col_e6.number_input("Dégradation batterie (%/an)", min_value=0.0,
                            max_value=10.0, value=2.0, step=0.1, key="degradation_pct_input") / 100.0

                    with st.container(border=True):
                        st.markdown("##### Marché")
                        col_e7, col_e8, col_e9 = st.columns(3)
                        taux_actualisation = col_e7.number_input("Taux d'actualisation (%)", min_value=0.0,
                            max_value=20.0, value=4.0, step=0.1, key="taux_actualisation_input") / 100.0
                        taux_inflation_energie = col_e8.number_input("Inflation prix électricité (%/an)",
                            min_value=0.0, max_value=10.0, value=3.0, step=0.1, key="taux_inflation_energie_input") / 100.0
                        prix_vente_reseau = col_e9.number_input("Prix de vente au réseau (€/kWh)", min_value=0.0,
                            value=0.0, step=0.01, format="%.3f", key="prix_vente_reseau_input")

                    if prix_vente_reseau >= prix_ttc_moyen:
                        st.warning("Le prix de vente au réseau est supérieur ou égal au prix d'achat évité : "
                                   "stocker n'a pas de sens économique dans ce cas.")

                # ----------------------------------------------------------------
                # SOUS-ONGLET 3 : COMPARAISON DES CAPACITÉS
                # ----------------------------------------------------------------
                with sous_tab3:
                    st.caption(" Le prix utilisé ici pour valoriser le gain net est pondéré par la "
                               "consommation totale du site (identique au sous-onglet « Tarification »), "
                               "pas par les instants précis où la batterie décharge — recalculer ce prix "
                               "pour chacune des ~100 capacités testées serait trop coûteux ici. Pour une "
                               "valorisation plus précise (pondérée par la décharge réelle), voir le "
                               "sous-onglet « 4. Bilan Financier », qui l'applique à la capacité choisie "
                               "individuellement.")
                    
                    
                    resultats_eco = []
                    for _, row in df_res_t4.iterrows():
                        cap = row["Capacité (kWh)"]
                        gain_net_kwh = row["Gain Énergétique (kWh)"]
                        cycles_par_an = row["Cycles par an"]
                        duree_vie_capacite = int(round(min(nombre_cycles_nominal / cycles_par_an, DUREE_VIE_MAX_ANS)
                                                          if cycles_par_an > 0 else DUREE_VIE_MAX_ANS))
                        capex = capex_unitaire * cap + capex_fixe
                        opex_annuel_an1 = capex * opex_pct
                        indic = calculer_flux_et_indicateurs(
                            gain_net_kwh, capex, opex_annuel_an1, prix_ttc_moyen, prix_vente_reseau,
                            taux_actualisation, duree_vie_capacite, degradation_pct,
                            taux_inflation_energie, taux_inflation_opex
                        )
                        resultats_eco.append({
                            "Capacité (kWh)": cap, "CAPEX (€)": capex,
                            "Durée de vie (années)": duree_vie_capacite,
                            "VAN (€)": indic["van"], "TRI (%)": indic["tri"], "LCOE (€/kWh)": indic["lcos"],
                            "TRB (années)": indic["payback"], "Ratio B/C": indic["ratio_bc"],
                        })
                  
                    df_eco = pd.DataFrame(resultats_eco)

                    range_van, range_tri = calculer_ranges_alignes(df_eco["VAN (€)"].values, df_eco["TRI (%)"].values)

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
                    fig_eco.update_yaxes(title_text="VAN (€)", range=range_van, secondary_y=False)
                    fig_eco.update_yaxes(title_text="TRI (%)", range=range_tri, secondary_y=True)
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
                        "LCOE (€/kWh)": fmt_num3, "TRB (années)": fmt_num1, "Ratio B/C": fmt_num2,
                    }).map(
                        style_van, subset=["VAN (€)"]
                    ).map(
                        style_tri, subset=["TRI (%)"]
                    ).map(
                        style_lcos, subset=["LCOE (€/kWh)"]
                    ).map(
                        style_payback, subset=["TRB (années)"]
                    ).map(
                        style_ratio_bc, subset=["Ratio B/C"]
                    ), column_config={
                        "Capacité (kWh)": st.column_config.Column(
                            help="Capacité de batterie testée, de 0 à 500 kWh par pas de 5 kWh. Pour chaque "
                                 "valeur, le gain énergétique net associé (calculé dans l'onglet 4) est "
                                 "valorisé avec les hypothèses économiques de l'onglet « Hypothèses »."
                        ),
                        "CAPEX (€)": st.column_config.Column(
                            help="Investissement initial (année 0) pour cette capacité : (coût unitaire "
                                 "€/kWh × capacité) + coûts fixes d'installation, tous deux définis dans "
                                 "l'onglet « Hypothèses ». Sert de base au calcul de la VAN, du LCOS et du "
                                 "ratio B/C."
                        ),
                        "VAN (€)": st.column_config.Column(
                            help="Valeur Actuelle Nette : somme de tous les flux de trésorerie annuels "
                                 "(recettes − OPEX), actualisés au taux d'actualisation retenu, moins le "
                                 "CAPEX initial. VAN = −CAPEX + Σ [flux net de l'année t ÷ (1+taux)^t]. "
                                 "Positive (vert) = le projet crée de la valeur à ce taux ; négative (gris) "
                                 "= il en détruit. C'est cet indicateur qui désigne la « capacité "
                                 "économiquement optimale » au-dessus du graphique."
                        ),
                        "TRI (%)": st.column_config.Column(
                            help="Taux de Rentabilité Interne : le taux d'actualisation pour lequel la VAN "
                                 "serait exactement nulle — le rendement annuel moyen implicite de "
                                 "l'investissement. Vert si supérieur à 0 %. Non calculable (N/A) si les "
                                 "flux ne changent jamais de signe sur la durée de vie retenue (jamais "
                                 "rentable, quel que soit le taux)."
                        ),
                        "LCOE (€/kWh)": st.column_config.Column(
                            help="Levelized Cost Of Storage : coût actualisé moyen de chaque kWh délivré "
                                 "par la batterie sur sa durée de vie = (CAPEX + OPEX actualisés) ÷ (énergie "
                                 "délivrée actualisée). Se compare directement au prix d'achat évité calculé "
                                 "dans l'onglet « Tarification » : plus il est bas, moins cher revient le "
                                 "kWh stocké par rapport à l'acheter."
                        ),
                        "TRB (années)": st.column_config.Column(
                            help="Temps de retour brut (non actualisé) : nombre d'années pour que la somme "
                                 "cumulée des flux de trésorerie annuels redevienne positive après "
                                 "l'investissement initial, calculé par interpolation entre les deux années "
                                 "encadrant le passage à zéro. Vide (N/A) si jamais atteint sur la durée de "
                                 "vie retenue."
                        ),
                        "Ratio B/C": st.column_config.Column(
                            help="Ratio Bénéfice/Coût : recettes actualisées ÷ coûts actualisés (CAPEX + "
                                 "OPEX). Vert si supérieur à 1 (les bénéfices actualisés dépassent les "
                                 "coûts) ; sous 1, l'investissement ne se rembourse pas sur la durée de vie "
                                 "retenue, même en tenant compte de l'actualisation."
                        ),
                    })

                # ----------------------------------------------------------------
                # SOUS-ONGLET 4 : DÉTAIL (FORMAT ENOLAB)
                # ----------------------------------------------------------------
                with sous_tab4:
                    capacite_etude = st.number_input("Capacité de la batterie étudiée (kWh)",
                        min_value=0.0, max_value=500.0, value = 0.0, step=5.0, key="capacite_etude_input")
                    ligne_capacite = df_res_t4.iloc[(df_res_t4["Capacité (kWh)"] - capacite_etude).abs().argsort()[:1]].iloc[0]
                    gain_net_kwh_reel = ligne_capacite["Gain Énergétique (kWh)"]
                    
                    duree_etude_v2 = st.number_input("Durée d'étude du bilan financier (années)",
                        min_value=1, max_value=30, value=20, step=1, key="duree_etude_v2_input",
                        help="Indépendante de la « Durée de vie » du sous-onglet Hypothèses (utilisée pour la "
                             "VAN/TRI du sous-onglet 3). Ici, c'est l'horizon du plan de trésorerie détaillé.")

                    opex_an1_v2 = st.number_input("OPEX année 1 (€ HT)", min_value=0.0, value=0.0, step=100.0,
                        key="opex_an1_v2_input")
                    st.caption(f"Inflation OPEX et inflation électricité réutilisées depuis l'onglet "
                               f"« 2. Hypothèses » : {taux_inflation_opex*100:.1f} %/an et "
                               f"{taux_inflation_energie*100:.1f} %/an.")

                    col_v4, col_v5 = st.columns(2)
                    revenu_producteur_an1 = col_v4.number_input("Coût producteur année 1 (€)",
                        min_value=0.0, value=0.0, step=10.0, key="revenu_producteur_an1_input",
                        help="Nature encore à définir. Laissé à 0 par défaut.")
                    capex_v2 = col_v5.number_input("CAPEX total (€ HT)", min_value=0.0,
                        value= 0, step=1000.0, key="capex_v2_input",
                        help="Valeur fictive par défaut (1 000 €/kWh).")
                    
                    
                    puissance_onduleur_etude = capacite_etude / 2.0
                    df_simu_etude, dt_etude = simuler_systeme_avec_batterie(df, capacite_etude, puissance_onduleur_etude, 0)

                    prix_ttc_moyen_decharge = prix_moyen_pondere_decharge_ttc(
                        df_simu_etude, dt_etude, segment_siege, cadran_siege, segment_bornes, cadran_bornes,
                        volume_siege, volume_bornes, accise_eur_mwh, taux_tva, turpe_dict,
                        mois_saison_haute_num, heure_debut_hc_haute, heure_fin_hc_haute,
                        heure_debut_hc_basse, heure_fin_hc_basse
                    )
                    st.caption(f"Prix évité pondéré par le moment réel de décharge de la batterie : "
                               f"{prix_ttc_moyen_decharge:.4f} €/kWh (contre {prix_ttc_moyen:.4f} €/kWh "
                               f"si pondéré par la conso totale du site).")

                    df_enolab = calculer_tableau_enolab(
                        capex=capex_v2, opex_an1=opex_an1_v2, taux_inflation_opex=taux_inflation_opex,
                        gain_net_kwh_an1=gain_net_kwh_reel, prix_moyen_ttc_an1=prix_ttc_moyen_decharge,
                        taux_inflation_energie=taux_inflation_energie,
                        revenu_producteur_an1=revenu_producteur_an1, taux_inflation_revenu_producteur=taux_inflation_opex,
                        duree_vie_ans=duree_etude_v2, degradation_pct_an=degradation_pct,
                        prix_vente_reseau=prix_vente_reseau
                    )


                    def fmt_eur(x):
                        return "" if pd.isna(x) else f"{x:,.0f}"

                    def couleur_flux(x):
                        if pd.isna(x):
                            return ""
                        return "background-color: #C6EFCE; color: #006100" if x >= 0 else "background-color: #FFC7CE; color: #9C0006"

                    st.dataframe(df_enolab.style.format({
                        "CAPEX (€ HT)": fmt_eur, "OPEX (€ HT)": fmt_eur,
                        "Énergie autoconsommée (kWh)": fmt_eur,
                        "Economie ACI (€ TTC)": fmt_eur, "Revenu producteur (€)": fmt_eur,
                        "Economie nette (€)": fmt_eur, "Flux cumulés (€)": fmt_eur,
                    }).map(couleur_flux, subset=["Flux cumulés (€)"]))

                    flux_annuels = df_enolab["Economie nette (€)"].values.astype(float)
                    cumul_v2 = df_enolab["Flux cumulés (€)"].values.astype(float)
                    annees_v2 = np.arange(0, len(flux_annuels))

                    def van_pour_taux_v2(r):
                        return np.sum(flux_annuels / (1 + r) ** annees_v2)

                    tri_v2 = None
                    if van_pour_taux_v2(-0.99) > 0 and van_pour_taux_v2(10.0) < 0:
                        lo, hi = -0.99, 10.0
                        for _ in range(200):
                            mid = (lo + hi) / 2
                            if van_pour_taux_v2(mid) > 0:
                                lo = mid
                            else:
                                hi = mid
                        tri_v2 = (lo + hi) / 2

                    opex_v2 = -df_enolab["OPEX (€ HT)"].fillna(0).values.astype(float)
                    energie_v2 = df_enolab["Énergie autoconsommée (kWh)"].fillna(0).values.astype(float)
                    facteurs_v2 = 1 / (1 + taux_actualisation) ** annees_v2
                    couts_actualises_v2 = capex_v2 + float(np.sum(opex_v2[1:] * facteurs_v2[1:]))
                    energie_actualisee_v2 = float(np.sum(energie_v2[1:] * facteurs_v2[1:]))
                    lcos_v2 = couts_actualises_v2 / energie_actualisee_v2 if energie_actualisee_v2 > 0 else float("nan")

                    payback_v2 = None
                    idx_positif_v2 = np.where(cumul_v2 >= 0)[0]
                    if len(idx_positif_v2) > 0 and idx_positif_v2[0] > 0:
                        i = idx_positif_v2[0]
                        payback_v2 = float((i - 1) + (-cumul_v2[i - 1] / flux_annuels[i])) if flux_annuels[i] != 0 else float(i)

                    st.markdown("##### Indicateurs de synthèse pour cette capacité")
                    
                    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
                    with col_s1:
                        st.markdown(carte_indicateur("TRI", f"{tri_v2*100:.1f} %" if tri_v2 is not None else "N/A",
                            "#E3F2FD", "#1565C0"), unsafe_allow_html=True)
                    with col_s2:
                        st.markdown(carte_indicateur("LCOE (LCOS)", f"{lcos_v2*100:.2f} c€/kWh" if not np.isnan(lcos_v2) else "N/A",
                            "#FFF3E0", "#E65100"), unsafe_allow_html=True)
                    with col_s3:
                        st.markdown(carte_indicateur("TRB (temps de retour Brut)", f"{payback_v2:.1f} ans" if payback_v2 is not None else "N/A",
                            "#F3E5F5", "#6A1B9A"), unsafe_allow_html=True)
                    with col_s4:
                        st.markdown(carte_indicateur("Valorisation interne", f"{prix_ttc_moyen*100:.2f} c€/kWh",
                            "#E8F5E9", "#2E7D32"), unsafe_allow_html=True)
                    pdf_buffer = generer_pdf_enolab(
                        df_enolab, capacite_etude, capex_v2,
                        f"{tri_v2*100:.1f} %" if tri_v2 is not None else "N/A",
                        f"{lcos_v2*100:.2f} c€/kWh" if not np.isnan(lcos_v2) else "N/A",
                        f"{payback_v2:.1f} ans" if payback_v2 is not None else "N/A",
                        f"{prix_ttc_moyen*100:.2f} c€/kWh"
                    )
                    st.download_button(
                        label=" Télécharger le bilan financier en PDF",
                        data=pdf_buffer,
                        file_name=f"bilan_financier_{capacite_etude:.0f}kWh.pdf",
                        mime="application/pdf"
                    )
                    png_buffer = generer_png_enolab(
                        df_enolab, capacite_etude, capex_v2,
                        f"{tri_v2*100:.1f} %" if tri_v2 is not None else "N/A",
                        f"{lcos_v2*100:.2f} c€/kWh" if not np.isnan(lcos_v2) else "N/A",
                        f"{payback_v2:.1f} ans" if payback_v2 is not None else "N/A",
                        f"{prix_ttc_moyen*100:.2f} c€/kWh"
                    )
                    st.download_button(
                        label=" Télécharger le bilan financier en PNG",
                        data=png_buffer,
                        file_name=f"bilan_financier_{capacite_etude:.0f}kWh.png",
                        mime="image/png"
                    )
else:
    st.info("Bienvenue ! Veuillez importer vos fichiers CSV ou EXCEL dans le panneau latéral pour commencer l'analyse.")