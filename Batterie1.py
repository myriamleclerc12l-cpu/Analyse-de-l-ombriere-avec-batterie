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
def charger_donnees_reelles(file_conso, file_prod):
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
# 2. INTERFACE STREAMLIT
# ==========================================

st.title("Simulation d'Autoconsommation et Stockage")

# --- Panneau latéral : Données ---
st.sidebar.header("Données réelles")

fichier_conso = st.sidebar.file_uploader("Courbe de charge (Siège TE13)", type=["csv", "xlsx", "xls"])
fichier_prod = st.sidebar.file_uploader("Production PV (Ombrière)", type=["csv", "xlsx", "xls"])

if fichier_conso is not None and fichier_prod is not None:
    st.sidebar.success("Fichiers réels chargés.")
    df_complet = charger_donnees_reelles(fichier_conso, fichier_prod)
    
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
        # CREATION DES TROIS ONGLETS
        tab1, tab2, tab3 = st.tabs(["Simulation Temporelle", "Isoler le Gain de la Batterie", "Analyse Annuelle"])
        
        # ----------------------------------------------------
        # ONGLET 1 : Simulation Temporelle
        # ----------------------------------------------------
        with tab1:
            st.markdown("---")
            st.header("Dimensionnement du Stockage")

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

            energie_pv_valorisee = prod_totale - export_totale
            nouveau_tac = (energie_pv_valorisee / prod_totale * 100) if prod_totale > 0 else 0
            nouveau_tap = (autoconso_totale / conso_totale * 100) if conso_totale > 0 else 0

            col_kpi1.metric("Taux d'Autoconso. (TAC)", f"{nouveau_tac:.1f} %")
            col_kpi2.metric("Taux d'Autoprod. (TAP)", f"{nouveau_tap:.1f} %")
            col_kpi3.metric("Énergie totale économisée", f"{autoconso_totale:.1f} kWh")

        # ----------------------------------------------------
        # ONGLET 2 : Isoler le Gain de la Batterie
        # ----------------------------------------------------
        with tab2:
            st.header("Gain de la Batterie")
            st.info(f"Période d'analyse : Du {date_debut.strftime('%d/%m/%Y')} au {date_fin.strftime('%d/%m/%Y')} ({(date_fin - date_debut).days + 1} jours)")
            st.markdown("""
            L'objectif ici est de visualiser uniquement l'apport de la batterie par rapport à une installation solaire simple sans stockage.
            **Gain net** = Énergie autoconsommée AVEC batterie − Énergie autoconsommée SANS batterie.
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
        # ONGLET 3 : Analyse Annuelle
        # ----------------------------------------------------
        with tab3:
            st.header("Analyse Annuelle : Gain, Autoproduction et Autoconsommation")
            st.markdown("""
            Cette analyse simule une **année complète glissante démarrant le 1er janvier** (indépendamment de la période
            sélectionnée plus haut), et calcule pour chaque capacité de batterie testée le gain, le TAP et le TAC.
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
                    st.subheader("💡 Validation des Hypothèses Techniques")
                    st.markdown("Cochez les critères que vous souhaitez imposer à votre batterie. L'outil calculera la capacité nécessaire pour valider **simultanément** toutes vos hypothèses.")

                    options_methodes = {
                        "Coude géométrique de rentabilité (Kneedle)": ("kneedle", None),
                        "Atteindre le plateau à 90 % du gain net maximal": ("plateau_gain", 0.90),
                        "Atteindre le plateau à 95 % du gain net maximal": ("plateau_gain", 0.95),
                        "Gain marginal < 200 kWh (pour chaque ajout de 5 kWh)": ("marginal_abs", 200),
                        "Gain marginal < 100 kWh (pour chaque ajout de 5 kWh)": ("marginal_abs", 100),
                        "Gain marginal < 50 kWh (pour chaque ajout de 5 kWh)": ("marginal_abs", 50),
                        "Atteindre le plateau à 90 % du TAP maximal possible": ("plateau_tap", 0.90),
                        "Atteindre le plateau à 95 % du TAP maximal possible": ("plateau_tap", 0.95),
                    }

                    col_c1, col_c2 = st.columns(2)
                    methodes_selectionnees = []
                    
                    for i, (label, params) in enumerate(options_methodes.items()):
                        # Cocher par défaut certaines options pour l'exemple
                        default_check = (i == 1 or i == 4) 
                        with (col_c1 if i % 2 == 0 else col_c2):
                            if st.checkbox(label, value=default_check, key=f"chk_hyp_{i}"):
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

                        st.success(f"### 🔋 Capacité recommandée : {cap_ideale_finale:.0f} kWh")
                        st.markdown("*Pour satisfaire simultanément tous vos critères, le système retient la valeur la plus exigeante parmi vos choix.*")

                        col_res1, col_res2, col_res3, col_res4 = st.columns(4)
                        col_res1.metric("Capacité retenue", f"{cap_ideale_finale:.0f} kWh")
                        col_res2.metric("Gain net annuel", f"{ligne_ideale['Gain Énergétique (kWh)']:.0f} kWh")
                        col_res3.metric("TAP estimé", f"{ligne_ideale['TAP (%)']:.1f} %")
                        col_res4.metric("TAC estimé", f"{ligne_ideale['TAC (%)']:.1f} %")
                    else:
                        st.warning("Veuillez cocher au moins une hypothèse technique ci-dessus.")

else:
    st.info("Bienvenue ! Veuillez importer vos fichiers CSV ou EXCEL dans le panneau latéral pour commencer l'analyse.")