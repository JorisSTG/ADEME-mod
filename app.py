import streamlit as st
import pandas as pd
import xarray as xr
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# ---- STYLE sombre pour se fondre avec le thème Streamlit ----
plt.style.use("dark_background")
plt.rcParams.update({
    "figure.facecolor": "none",
    "axes.facecolor": "none",
    "savefig.facecolor": "none",
    "axes.edgecolor": "#FFFFFF",
    "axes.labelcolor": "#FFFFFF",
    "xtick.color": "#DDDDDD",
    "ytick.color": "#DDDDDD",
    "text.color": "#FFFFFF",
})


st.title("Comparaison : Modèle / TRACC")

st.markdown(
    """

    L’objectif de cette application est d’évaluer la précision de données météorologiques en les comparant à des données de référence, afin de juger de leur pertinence pour les projections climatiques futures en France. Ces données de référence correspondent aux jeux TRACC, issus de différentes méthodes de génération d’années types.
    Les comparaisons sont réalisées entre les projections climatiques TRACC — constituées d’années types dans un climat à +X °C, avec ou sans vague de chaleur — et les données issues du « modèle », fournies sous forme d’un fichier CSV contenant uniquement la température, soit une série de 1 × 8760 valeurs.
    Cet outil est principalement utilisé dans le domaine du bâtiment, notamment pour l’évaluation thermique à travers des modèles de simulation dynamique (STD).

    https://data.ademe.fr/datasets/?q=tracc&topics=fJZXrdcRGP
    
    **Note sur les couleurs :**  
    - Les couleurs visent à caractériser le **MODÈLE** (données issues du fichier `.csv`).  
    - Rouge → Modèle plus chaud que TRACC  
    - Bleu → Modèle plus froid que TRACC  
    - Pour les indicateurs de précision : vert → bon résultat, rouge → moins bon résultat
    """,
    unsafe_allow_html=True
)

# -------- Paramètres --------
scenarios = ["2", "2_VC", "2-7", "2-7_VC", "4", "4_VC"]
villes = ["AGEN", "CARPENTRAS", "MACON", "MARIGNANE", "NANCY", "RENNES", "TOURS", "TRAPPES"]
heures_par_mois = [744, 672, 744, 720, 744, 720, 744, 744, 720, 744, 720, 744]
percentiles_list = [10, 25, 50, 75, 90]

couleur_modele = "goldenrod"
couleur_TRACC = "lightgray"
vmaxT=5
vminT=-5

vmaxP=100
vminP=50

vmaxH=100
vminH=-100

vmaxDJU=150
vminDJU=-150

# -------- Noms des mois --------
mois_noms = {
    1: "01 - Janvier",   2: "02 - Février",  3: "03 - Mars",
    4: "04 - Avril",     5: "05 - Mai",      6: "06 - Juin",
    7: "07 - Juillet",   8: "08 - Août",     9: "09 - Septembre",
    10: "10 - Octobre", 11: "11 - Novembre", 12: "12 - Décembre"
}

# -------- Choix scénario et ville --------
scenario_sel = st.selectbox("Choisir le scénario :", scenarios)
ville_sel = st.selectbox("Choisir la ville :", villes)
base_folder = "ADEME"

# -------- Upload CSV modèle --------
uploaded = st.file_uploader("Déposer le fichier CSV du modèle (colonne unique T°C) :", type=["csv"])

if uploaded:

    st.markdown("")
    
    # Lecture CSV modèle
    model_values = pd.read_csv(uploaded, header=0).iloc[:, 0].values

    # Lecture NetCDF
    nc_file_sel = os.path.join(base_folder, scenario_sel, f"{ville_sel}.nc")
    ds_obs = xr.open_dataset(nc_file_sel, decode_times=True)
    obs_series = ds_obs["T2m"].to_series()
    df_obs = obs_series.reset_index()
    df_obs["year"] = df_obs["time"].dt.year
    df_obs["month_num"] = df_obs["time"].dt.month
    df_obs["month"] = df_obs["month_num"].map(mois_noms)
    df_obs["day"] = df_obs["time"].dt.day

    # -------- RMSE --------
    def rmse(a, b):
        min_len = min(len(a), len(b))
        a_sorted = np.sort(a[:min_len])
        b_sorted = np.sort(b[:min_len])
        return np.sqrt(np.nanmean((a_sorted - b_sorted) ** 2))
    
    # -------- Nouvelle fonction : indice de recouvrement --------
    def precision_overlap(a, b, bin_width=1.0):
        """
        Calcule l'indice de recouvrement (%) entre deux séries de données.
        bin_width : largeur des tranches pour l'histogramme (en °C)
        """
        if len(a) == 0 or len(b) == 0:
            return np.nan
    
        # Définir les bornes de l'histogramme
        min_val = min(np.min(a), np.min(b))
        max_val = max(np.max(a), np.max(b))
        bins = np.arange(min_val, max_val + bin_width, bin_width)
    
        # Calcul des histogrammes normalisés
        hist_a, _ = np.histogram(a, bins=bins, density=True)
        hist_b, _ = np.histogram(b, bins=bins, density=True)
    
        # Indice de recouvrement
        overlap = np.sum(np.minimum(hist_a, hist_b) * bin_width)
        indice_percent = overlap * 100
        return round(indice_percent, 2)


    # -------- Boucle sur les mois --------
    results_rmse = []
    obs_mois_all = []
    start_idx_model = 0

    for mois_num, nb_heures in enumerate(heures_par_mois, start=1):
        mois = mois_noms[mois_num]
        mod_mois = model_values[start_idx_model:start_idx_model + nb_heures]
        obs_mois_vals = df_obs[df_obs["month_num"] == mois_num]["T2m"].values
        obs_mois_all.append(obs_mois_vals)

        val_rmse = rmse(mod_mois, obs_mois_vals)
        pct_precision = precision_overlap(mod_mois, obs_mois_vals)

        results_rmse.append({
            "Mois": mois,
            "RMSE (°C)": round(val_rmse, 2),
            "Précision percentile (%)": pct_precision
        })

        start_idx_model += nb_heures

    # -------- DataFrame final --------
    df_rmse = pd.DataFrame(results_rmse)
    df_rmse_styled = (
        df_rmse.style
        .background_gradient(subset=["Précision percentile (%)"], cmap="RdYlGn", vmin=vminP, vmax=vmaxP, axis=None)
        .format({"Précision percentile (%)": "{:.2f}", "RMSE (°C)": "{:.2f}"})
    )
    st.subheader("")
    st.subheader("Précision du modèle : RMSE et précision via écarts des percentiles")
    
    st.markdown(
        """
        La précision est calculée à partir de la moyenne des différences absolues entre les percentiles du modèle et ceux de la TRACC (c’est-à-dire le RMSE), ainsi que de l’écart-type du mois de référence issu des données TRACC.
        """,
        unsafe_allow_html=True
    )
    st.dataframe(df_rmse_styled, hide_index=True)

    # -------- Précision globale annuelle --------
    model_annee = model_values[:sum(heures_par_mois)]        # toutes les heures de l'année
    obs_annee = np.concatenate(obs_mois_all)                # toutes les heures TRACC concaténées
    
    precision_annuelle = precision_overlap(model_annee, obs_annee)
    st.subheader(f"Précision globale annuelle : {precision_annuelle} %")
    st.subheader("")

    # -------- Seuils --------
    t_sup_thresholds = st.text_input("Seuils Tmax supérieur (°C, séparés par des virgules)", "25,30,35")
    t_inf_thresholds = st.text_input("Seuils Tmin inférieur (°C, séparés par des virgules)", "-5,0,5")
    t_sup_thresholds_list = [int(float(x.strip())) for x in t_sup_thresholds.split(",")]
    t_inf_thresholds_list = [int(float(x.strip())) for x in t_inf_thresholds.split(",")]
    
    stats_sup = []
    stats_inf = []
    
    for mois_num, nb_heures in enumerate(heures_par_mois, start=1):
        mois = mois_noms[mois_num]
        idx0 = sum(heures_par_mois[:mois_num-1])
        idx1 = sum(heures_par_mois[:mois_num])
        mod_mois = model_values[idx0:idx1]
        obs_mois = obs_mois_all[mois_num-1]
    
        # Seuils supérieurs
        for seuil in t_sup_thresholds_list:
            heures_obs = np.sum(obs_mois > seuil)
            nb_heures_mod = np.sum(mod_mois > seuil)
            ecart = nb_heures_mod - heures_obs  # Modèle - TRACC
            stats_sup.append({
                "Mois": mois,
                "Seuil (°C)": f"{seuil}",
                "Heures Modèle": nb_heures_mod,
                "Heures TRACC": heures_obs,
                "Ecart (Modèle - TRACC)": ecart
            })
        
        # Seuils inférieurs
        for seuil in t_inf_thresholds_list:
            heures_obs = np.sum(obs_mois < seuil)
            nb_heures_mod = np.sum(mod_mois < seuil)
            ecart = nb_heures_mod - heures_obs  # Modèle - TRACC
            stats_inf.append({
                "Mois": mois,
                "Seuil (°C)": f"{seuil}",
                "Heures Modèle": nb_heures_mod,
                "Heures TRACC": heures_obs,
                "Ecart (Modèle - TRACC)": ecart
            })
    
    # Création des DataFrames
    df_sup = pd.DataFrame(stats_sup)
    df_inf = pd.DataFrame(stats_inf)
    
    # Conversion en int
    for df in [df_sup, df_inf]:
        df["Heures Modèle"] = df["Heures Modèle"].astype(int)
        df["Heures TRACC"] = df["Heures TRACC"].astype(int)
        df["Ecart (Modèle - TRACC)"] = df["Ecart (Modèle - TRACC)"].astype(int)
    
    # Style : seuils supérieurs → rouge = plus chaud
    df_sup_styled = (
        df_sup.style
        .background_gradient(subset=["Ecart (Modèle - TRACC)"], cmap="bwr", vmin=vminH, vmax=vmaxH, axis=None)
    )
    st.subheader("Nombre d'heures supérieur au(x) seuil(s)")
    st.dataframe(df_sup_styled, hide_index=True)
    
    # Style : seuils inférieurs → rouge = plus froid
    # Pour inverser les couleurs, on peut juste inverser le cmap
    df_inf_styled = (
        df_inf.style
        .background_gradient(subset=["Ecart (Modèle - TRACC)"], cmap="bwr_r", vmin=vminH, vmax=vmaxH, axis=None)
    )
    st.subheader("Nombre d'heures inférieur au(x) seuil(s)")
    st.dataframe(df_inf_styled, hide_index=True)

    # =====================================
    # ======= SOMMES ANNUELLES =============
    # =====================================
    
    obs_all = np.concatenate(obs_mois_all)
    mod_all = np.array(model_values)
    
    annual_sup = []
    annual_inf = []
    
    # ----- Supérieurs -----
    for seuil in t_sup_thresholds_list:
        heures_obs = np.sum(obs_all > seuil)
        heures_mod = np.sum(mod_all > seuil)
        ecart = heures_mod - heures_obs
    
        annual_sup.append({
            "Période": "Année",
            "Seuil (°C)": f"{seuil}",
            "Heures source 1": int(heures_mod),
            "Heures source 2": int(heures_obs),
            "Ecart (soure 1 - source 2)": int(ecart)
        })
    
    # ----- Inférieurs -----
    for seuil in t_inf_thresholds_list:
        heures_obs = np.sum(obs_all < seuil)
        heures_mod = np.sum(mod_all < seuil)
        ecart = heures_mod - heures_obs
    
        annual_inf.append({
            "Période": "Année",
            "Seuil (°C)": f"{seuil}",
            "Heures source 1": int(heures_mod),
            "Heures source 2": int(heures_obs),
            "Ecart (soure 1 - source 2)": int(ecart)
        })
    
    df_sup_year = pd.DataFrame(annual_sup)
    df_inf_year = pd.DataFrame(annual_inf)
    
    # =====================================
    # ======= AFFICHAGE ANNUEL =============
    # =====================================
    
    st.subheader("Somme annuelle — Nombre d'heures supérieur au(x) seuil(s)")
    df_sup_year_styled = (
        df_sup_year.style
        .background_gradient(subset=["Ecart (soure 1 - source 2)"], cmap="bwr", vmin=vminH*12, vmax=vmaxH*12, axis=None)
    )
    st.dataframe(df_sup_year_styled, hide_index=True)
    
    st.subheader("Somme annuelle — Nombre d'heures inférieur au(x) seuil(s)")
    df_inf_year_styled = (
        df_inf_year.style
        .background_gradient(subset=["Ecart (soure 1 - source 2)"], cmap="bwr_r", vmin=vminH*12, vmax=vmaxH*12, axis=None)
    )
    st.dataframe(df_inf_year_styled, hide_index=True)


    # -------- Histogrammes par plage de température --------
    st.subheader(f"Histogrammes horaire : Modèle et TRACC +{scenario_sel}/{ville_sel}")
    st.markdown(
        """
        La valeur de chaque barre est égal au total d'heure compris entre [ X°C , X+1°C [
        """,
        unsafe_allow_html=True
    )
    # Bins correspondant à [X, X+1[ pour chaque température entière
    bin_edges = bins = np.arange(-5, 46, 1)  # bornes des bins
    bin_labels = bin_edges[:-1].astype(int)  # labels = début de l'intervalle
    
    def count_hours_in_bins(temp_hourly, bins):
        counts, _ = np.histogram(temp_hourly, bins=bins)
        return counts
    
    for mois_num in range(1, 13):
        mois = mois_noms[mois_num]
        
        # Observations
        obs_hourly = obs_mois_all[mois_num-1]
        obs_counts = count_hours_in_bins(obs_hourly, bin_edges)
        
        # Modèle
        idx0 = sum(heures_par_mois[:mois_num-1])
        idx1 = sum(heures_par_mois[:mois_num])
        mod_hourly = model_values[idx0:idx1]
        mod_counts = count_hours_in_bins(mod_hourly, bin_edges)
        
        # Préparer le DataFrame pour le plot
        df_plot = pd.DataFrame({
            "Temp_Num": bin_labels,
            "Température": bin_labels.astype(str),
            "TRACC": obs_counts,
            "Modèle": mod_counts
        }).sort_values("Temp_Num")
        
        # Création du plot
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.bar(df_plot["Temp_Num"] - 0.25, df_plot["TRACC"], width=0.5, label=f"TRACC +{scenario_sel}/{ville_sel}", color=couleur_TRACC)
        ax.bar(df_plot["Temp_Num"] + 0.25, df_plot["Modèle"], width=0.5, label="Modèle", color=couleur_modele)
        ax.set_title(f"{mois} - Durée en heure par seuil de température")
        ax.set_xlabel("Température (°C)")
        ax.set_ylabel("Durée en heure")
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)

    # -------- Histogramme annuel par plage de température --------
    st.subheader(f"Histogramme annuel : Modèle et TRACC +{scenario_sel}/{ville_sel}")
    st.markdown(
        """
        La valeur de chaque barre est égale au total d'heures compris entre [ X°C , X+1°C [
        sur l'année entière.
        """,
        unsafe_allow_html=True
    )
    
    # Bins correspondant à [X, X+1[
    bin_edges = np.arange(-5, 46, 1)
    bin_labels = bin_edges[:-1].astype(int)
    
    def count_hours_in_bins(temp_hourly, bins):
        counts, _ = np.histogram(temp_hourly, bins=bins)
        return counts
    
    # -------- Regroupement ANNUEL --------
    # Observations : concaténer tous les mois
    obs_hourly_annual = np.concatenate(obs_mois_all)
    
    # Modèle : toutes les valeurs de l'année
    mod_hourly_annual = model_values  # déjà une série horaire complète
    
    # Comptages annuels
    obs_counts_annual = count_hours_in_bins(obs_hourly_annual, bin_edges)
    mod_counts_annual = count_hours_in_bins(mod_hourly_annual, bin_edges)

    diff_counts_annual_TRACC = np.maximum(0, obs_counts_annual - mod_counts_annual)
    diff_counts_annual_modele = np.maximum(0, mod_counts_annual - obs_counts_annual)

    # Préparer DataFrame pour le plot
    df_plot_year = pd.DataFrame({
        "Temp_Num": bin_labels,
        "Température": bin_labels.astype(str),
        "TRACC": obs_counts_annual,
        "Modèle": mod_counts_annual
    }).sort_values("Temp_Num")
    
    # Plot
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.bar(df_plot_year["Temp_Num"] - 0.25, df_plot_year["TRACC"], width=0.5,
           label=f"TRACC +{scenario_sel}/{ville_sel}", color=couleur_TRACC)
    ax.bar(df_plot_year["Temp_Num"] + 0.25, df_plot_year["Modèle"], width=0.5,
           label="Modèle", color=couleur_modele)

    fig_hist_year = fig
    ax.set_title("Année entière - Durée en heures par seuil de température")
    ax.set_xlabel("Température (°C)")
    ax.set_ylabel("Durée en heure")
    ax.legend()
    
    st.pyplot(fig)
    plt.close(fig)

    # Préparer DataFrame pour le plot
    df_plot_year = pd.DataFrame({
        "Temp_Num": bin_labels,
        "Température": bin_labels.astype(str),
        "Différence absolue modele": diff_counts_annual_modele,
        "Différence absolue TRACC": diff_counts_annual_TRACC
    }).sort_values("Temp_Num")
    
    # Plot
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.bar(df_plot_year["Temp_Num"], df_plot_year["Différence absolue modele"], width=0.8,
           label="Différence : Modèle > TRACC", color=couleur_modele)
    
    ax.bar(df_plot_year["Temp_Num"], df_plot_year["Différence absolue TRACC"], width=0.8,
           label="Différence : Modèle < TRACC", color=couleur_TRACC)
    
    ax.set_title("Année entière - Différence en heures par seuil de température")
    ax.set_xlabel("Température (°C)")
    ax.set_ylabel("Durée en heure")
    ax.legend()
    fig_hist_diff = fig
    st.pyplot(fig)
    plt.close(fig)

    st.markdown(
        """
        La couleur de la différence est définie ainsi :

        Barres jaunes : le modèle compte davantage d’heures que la TRACC dans cette plage de température.

        Barres blanches : la TRACC compte davantage d’heures que le modèle dans cette plage de température.

        La conclusion dépend donc de l’endroit où se situe cette différence. Une analyse doit être réalisée manuellement : par exemple, si la TRACC présente plus d’heures dans les plages « froides », cela signifie qu’elle est globalement plus froide que le modèle.
        Comme les deux séries possèdent le même nombre total d’heures, un excès d’heures froides dans la TRACC implique mécaniquement un excès d’heures chaudes dans le modèle (et inversement).
        """,
        unsafe_allow_html=True
    )

    # =============================
    # Comparaison annuelle histogrammes horaires
    # =============================
    
    # Comparaison pour les températures élevées
    tx_seuil_chaud = 25
    heures_TRACC_chaud = np.sum(obs_hourly_annual > tx_seuil_chaud)
    heures_modele_chaud = np.sum(mod_hourly_annual > tx_seuil_chaud)
    
    if heures_TRACC_chaud > heures_modele_chaud:
        phrase_tx_chaud = f"TRACC a plus d'heures avec une T>{tx_seuil_chaud}°C ({heures_TRACC_chaud}) que le modèle ({heures_modele_chaud})."
    else:
        phrase_tx_chaud = f"Le modèle a plus d'heures avec une T>{tx_seuil_chaud}°C ({heures_modele_chaud}) que TRACC ({heures_TRACC_chaud})."

    tn_seuil_froid = 5
    heures_TRACC_froid = np.sum(obs_hourly_annual < tn_seuil_froid)
    heures_modele_froid = np.sum(mod_hourly_annual < tn_seuil_froid)
    
    if heures_TRACC_froid > heures_modele_chaud:
        phrase_tn_froid = f"Le modèle a plus d'heures avec une T<{tn_seuil_froid}°C ({heures_modele_froid}) que TRACC ({heures_TRACC_froid})."
    else:
        phrase_tn_froid = f"TRACC a plus d'heures avec une T<{tn_seuil_froid}°C ({heures_TRACC_froid}) que le modèle ({heures_modele_froid})."

    # Stocker dans st.session_state pour la page Résumé
    st.session_state["resume_hist"] = [phrase_tx_chaud, phrase_tn_froid]
    
    # Optionnel : affichage sur la page actuelle
    st.subheader("Résumé comparatif histogrammes horaires/annuels")
    for p in st.session_state["resume_hist"]:
        st.write("- " + p)


    # -------- Précision par créneau horaire --------
    results_temp = []
    def rmse_hours(obs_counts, mod_counts):
        min_len = min(len(obs_counts), len(mod_counts))
        return np.sqrt(np.nanmean((np.array(obs_counts[:min_len]) - np.array(mod_counts[:min_len]))**2))

    for mois_num in range(1, 13):
        mois = mois_noms[mois_num]
        obs_hourly = obs_mois_all[mois_num-1]
        idx0 = sum(heures_par_mois[:mois_num-1])
        idx1 = sum(heures_par_mois[:mois_num])
        mod_hourly = model_values[idx0:idx1]
        obs_counts = count_hours_in_bins(obs_hourly, bins)
        mod_counts = count_hours_in_bins(mod_hourly, bins)
        total_hours = 2*heures_par_mois[mois_num-1]
        hours_error = sum(abs(np.array(obs_counts) - np.array(mod_counts)))
        pct_precision = round(100 * (1 - hours_error / total_hours), 2)
        val_rmse = rmse_hours(obs_counts, mod_counts)
        results_temp.append({
            "Mois": mois,
            "RMSE (heure)": round(val_rmse,2),
            "Précision (%)": pct_precision
        })

    df_temp_precision = pd.DataFrame(results_temp)
    df_temp_precision_styled = df_temp_precision.style \
        .background_gradient(subset=["Précision (%)"], cmap="RdYlGn", vmin=vminP, vmax=vmaxP, axis=None) \
        .format({"Précision (%)": "{:.2f}", "RMSE (heure)": "{:.2f}"})

    st.subheader(f"Précision du modèle sur la répartition des durées des plages de température (TRACC +{scenario_sel}/{ville_sel})")
    st.markdown(
        """
        Le RMSE correspond à la moyenne de l’écart absolu entre les valeurs du modèle et celles de la TRACC pour chaque intervalle de température.
        La précision est calculée à partir de la différence totale d’heures dans chaque intervalle 
        """,
        unsafe_allow_html=True
    )
    st.dataframe(df_temp_precision_styled, hide_index=True)

    # ============================
    #   COURBES Tn / Tmoy / Tx
    # ============================
    st.subheader("Évolution mensuelle : Tn_mois / Tmoy_mois / Tx_mois (Modèle vs TRACC)")
    st.markdown(
        """  
        - Les valeurs tracées représentent les températures minimales et maximales **absolues** du mois (c’est-à-dire P0 et P100)
        - De ce fait, les températures du mois ne dépassent jamais les bornes définies par Tn_mois et Tx_mois.
        - La température moyenne (Tmoy_mois) correspond à la moyenne mensuelle calculée sur l’ensemble des pas de temps
        """,
        unsafe_allow_html=True
    )
    # Calcul des Tn/Tmoy/Tx pour 12 mois
    results_tstats = []
    for mois_num in range(1, 12+1):
        mois = mois_noms[mois_num]
    
        # Observations
        obs_vals = obs_mois_all[mois_num-1]
        obs_tn = np.min(obs_vals)
        obs_tm = np.mean(obs_vals)
        obs_tx = np.max(obs_vals)
    
        # Modèle
        idx0 = sum(heures_par_mois[:mois_num-1])
        idx1 = sum(heures_par_mois[:mois_num])
        mod_vals = model_values[idx0:idx1]
        mod_tn = np.min(mod_vals)
        mod_tm = np.mean(mod_vals)
        mod_tx = np.max(mod_vals)
    
        results_tstats.append({
            "Mois": mois,
            "TRACC_Tn": obs_tn, "Modèle_Tn": mod_tn, "TRACC_Tm": obs_tm, "Modèle_Tm": mod_tm, "TRACC_Tx": obs_tx, "Modèle_Tx": mod_tx
        })
    
    df_tstats = pd.DataFrame(results_tstats)
    
    # ---- Plot ----
    fig, ax = plt.subplots(figsize=(14,4))

    ax.plot(df_tstats["Mois"], df_tstats["Modèle_Tx"], color="red", label="Modèle Tx", linestyle="-")
    ax.plot(df_tstats["Mois"], df_tstats["Modèle_Tm"], color="white", label="Modèle Tmoy", linestyle="-")
    ax.plot(df_tstats["Mois"], df_tstats["Modèle_Tn"], color="cyan", label="Modèle Tn", linestyle="-")

    ax.plot(df_tstats["Mois"], df_tstats["TRACC_Tx"], color="red", label="TRACC Tx", linestyle="--")
    ax.plot(df_tstats["Mois"], df_tstats["TRACC_Tm"], color="white", label="TRACC Tmoy", linestyle="--")
    ax.plot(df_tstats["Mois"], df_tstats["TRACC_Tn"], color="cyan", label="TRACC Tn", linestyle="--")

    ax.set_title(f"Tn_mois / Tmoy_mois / Tx_mois – Modèle vs TRACC +{scenario_sel}/{ville_sel}")
    ax.set_ylabel("Température (°C)")
    ax.tick_params(axis='x', rotation=45)
    ax.legend(facecolor="black")

    fig_tn_tx_mois = fig
    
    st.pyplot(fig)
    plt.close(fig)
    
    # ---- Tableau correspondant ----
    st.write("Tableau Tn_mois / Tmoy_mois / Tx_mois")
    st.dataframe(df_tstats.round(2), hide_index=True)

    # ---- Tableau des différences (Modèle - TRACC) ----
    df_diff = pd.DataFrame({
        "Mois": df_tstats["Mois"],
        "Diff_Tn_mois": df_tstats["Modèle_Tn"] - df_tstats["TRACC_Tn"],
        "Diff_Tmoy_mois": df_tstats["Modèle_Tm"] - df_tstats["TRACC_Tm"],
        "Diff_Tx_mois": df_tstats["Modèle_Tx"] - df_tstats["TRACC_Tx"],
    })
    
    df_diff_round = df_diff.copy()
    df_diff_round[["Diff_Tn_mois","Diff_Tmoy_mois","Diff_Tx_mois"]] = df_diff_round[["Diff_Tn_mois","Diff_Tmoy_mois","Diff_Tx_mois"]].round(2)
    
    st.write("Différences Modèle - TRACC (Tn_mois / Tmoy_mois / Tx_mois)")
        
    # ---- Coloration avec background_gradient ----
    st.dataframe(
        df_diff_round.style
            .background_gradient(cmap="bwr", vmin=vminT, vmax=vmaxT)
            .format("{:.2f}", subset=["Diff_Tn_mois","Diff_Tmoy_mois","Diff_Tx_mois"]),
        hide_index=True,
        use_container_width=True
    )

    # =============================
    # Comparaison moyenne annuelle
    # =============================
    
    # Moyenne annuelle sur 12 mois pour TRACC et Modèle
    mean_TRACC_Tx = df_tstats["TRACC_Tx"].mean()
    mean_Model_Tx = df_tstats["Modèle_Tx"].mean()
    
    mean_TRACC_Tm = df_tstats["TRACC_Tm"].mean()
    mean_Model_Tm = df_tstats["Modèle_Tm"].mean()
    
    mean_TRACC_Tn = df_tstats["TRACC_Tn"].mean()
    mean_Model_Tn = df_tstats["Modèle_Tn"].mean()
    
    # Générer les phrases
    if mean_TRACC_Tx > mean_Model_Tx:
        phrase_Tx = "En moyenne, la TRACC est plus chaude que le modèle pour les températures maximales (Tx)."
    else:
        phrase_Tx = "En moyenne, le modèle est plus chaud que TRACC pour les températures maximales (Tx)."
    
    if mean_TRACC_Tm > mean_Model_Tm:
        phrase_Tm = "En moyenne, la TRACC est plus chaude que le modèle pour les températures moyennes (Tmoy)."
    else:
        phrase_Tm = "En moyenne, le modèle est plus chaud que TRACC pour les températures moyennes (Tmoy)."
    
    if mean_TRACC_Tn > mean_Model_Tn:
        phrase_Tn = "En moyenne, la TRACC est plus chaude que le modèle pour les températures minimales (Tn)."
    else:
        phrase_Tn = "En moyenne, le modèle est plus chaud que TRACC pour les températures minimales (Tn)."
    
    # Stocker dans st.session_state pour pouvoir les réutiliser dans la page Résumé
    st.session_state["resume_temp"] = [phrase_Tx, phrase_Tm, phrase_Tn]
    
    # Optionnel : afficher directement les phrases sur cette page
    st.subheader("Résumé comparatif annuel des températures")
    for p in st.session_state["resume_temp"]:
        st.write("- " + p)


    # ============================
    #  SECTION: Tn / Tmoy / Tx journaliers
    # ============================
    st.subheader("Tn_jour / Tmoy_jour /  — CDF par mois et tableaux de percentiles")
    
    def daily_stats_from_hourly(hourly):
        """
        Retourne trois tableaux journaliers (min, mean, max).
        Tronque si nécessaire pour avoir des jours complets (24h).
        """
        if len(hourly) < 24:
            return np.array([]), np.array([]), np.array([])
        n_full_days = len(hourly) // 24
        arr = np.array(hourly[: n_full_days * 24]).reshape((n_full_days, 24))
        daily_min = arr.min(axis=1)
        daily_max = arr.max(axis=1)
        daily_mean = (daily_max+daily_min)/2
        return daily_min, daily_mean, daily_max
    
    # percentiles pour les petits tableaux
    pct_table = percentiles_list  # utilise la liste déjà définie en haut (ex: [10,25,50,75,90])
    pct_for_cdf = np.linspace(0, 100, 100)  # pour tracer les CDF
    
    Tx_jour_all = []
    Tn_jour_all = []
    Tm_jour_all = []

    Tx_jour_mod_all = []
    Tn_jour_mod_all = []
    Tm_jour_mod_all = []
    
    # boucle mois par mois
    for mois_num in range(1, 13):
        mois = mois_noms[mois_num]
    
        # ---- extraire hourly pour le mois: TRACC (obs) et modèle (csv) ----
        obs_hourly = obs_mois_all[mois_num - 1] if len(obs_mois_all) >= mois_num else np.array([])
        idx0 = sum(heures_par_mois[:mois_num - 1])
        idx1 = sum(heures_par_mois[:mois_num])
        model_hourly = model_values[idx0:idx1]
    
        # ---- calculer stats journalières ----
        obs_tn, obs_tm, obs_tx = daily_stats_from_hourly(obs_hourly)
        mod_tn, mod_tm, mod_tx = daily_stats_from_hourly(model_hourly)
        
        # Stocker les séries journalières OBS uniquement
        Tn_jour_all.append(obs_tn)
        Tm_jour_all.append(obs_tm)
        Tx_jour_all.append(obs_tx)

        # Stocker les séries journalières Modèle
        Tn_jour_mod_all.append(mod_tn)
        Tm_jour_mod_all.append(mod_tm)
        Tx_jour_mod_all.append(mod_tx)
        
        # ---- préparer CDFs (percentiles des séries journalières) ----
        obs_tn_cdf = np.percentile(obs_tn, pct_for_cdf)
        mod_tn_cdf = np.percentile(mod_tn, pct_for_cdf)
        obs_tm_cdf = np.percentile(obs_tm, pct_for_cdf)
        mod_tm_cdf = np.percentile(mod_tm, pct_for_cdf)
        obs_tx_cdf = np.percentile(obs_tx, pct_for_cdf)
        mod_tx_cdf = np.percentile(mod_tx, pct_for_cdf)
    
        # ---- tracé : un seul graphique regroupant Tn / Tmoy / Tx ----
        fig, ax = plt.subplots(figsize=(12, 4))
    
        # Couleurs cohérentes pour chaque variable
        colors = {
            "Tn": "cyan",
            "Tm": "white",
            "Tx": "red"
        }
    
        # Tracer Modèle
        ax.plot(pct_for_cdf, mod_tx_cdf, linestyle="-", linewidth=2, label="Modèle Tx", color=colors["Tx"])
        ax.plot(pct_for_cdf, mod_tm_cdf, linestyle="-", linewidth=2, label="Modèle Tmoy", color=colors["Tm"])
        ax.plot(pct_for_cdf, mod_tn_cdf, linestyle="-", linewidth=2, label="Modèle Tn", color=colors["Tn"])
    
        # Tracer TRACC
        ax.plot(pct_for_cdf, obs_tx_cdf, linestyle="--", linewidth=1.7, label="TRACC Tx", color=colors["Tx"])
        ax.plot(pct_for_cdf, obs_tm_cdf, linestyle="--", linewidth=1.7, label="TRACC Tmoy", color=colors["Tm"])
        ax.plot(pct_for_cdf, obs_tn_cdf, linestyle="--", linewidth=1.7, label="TRACC Tn", color=colors["Tn"])
    
        # Mise en forme
        ax.set_title(f"{mois} — CDF Tn_jour / Tmoy_jour / Tx_jour (Modèle vs TRACC +{scenario_sel}/{ville_sel})", color="white")
        ax.set_xlabel("Percentile", color="white")
        ax.set_ylabel("Température (°C)", color="white")
        ax.tick_params(colors="white")
        ax.legend(facecolor="black")
        ax.set_facecolor("none")
    
        st.pyplot(fig)
        plt.close(fig)
    
        def pct_table_values(arr, pct_list):
            return [np.percentile(arr, p) for p in pct_list]
    
        # ---- Tableau des percentiles ----
        tab = pd.DataFrame({
            "Percentile": [f"P{p}" for p in pct_table],
            "TRACC_Tn": np.round(pct_table_values(obs_tn, pct_table), 2),
            "Mod_Tn": np.round(pct_table_values(mod_tn, pct_table), 2),
            "TRACC_Tm": np.round(pct_table_values(obs_tm, pct_table), 2),
            "Mod_Tm": np.round(pct_table_values(mod_tm, pct_table), 2),
            "TRACC_Tx": np.round(pct_table_values(obs_tx, pct_table), 2),
            "Mod_Tx": np.round(pct_table_values(mod_tx, pct_table), 2),
        })
    
        st.write(f"{mois} — Table des percentiles journaliers (Tn_jour / Tmoy_jour / Tx_jour)")
    
        num_cols = tab.select_dtypes(include=[np.number]).columns
        tab[num_cols] = tab[num_cols].apply(pd.to_numeric, errors="coerce")
        styler = tab.style.format({col: "{:.2f}" for col in num_cols})
        st.dataframe(styler, hide_index=True)
    
        # ---- Tableau des différences (Modèle - TRACC) ----
        df_diff = pd.DataFrame({
            "Percentile": tab["Percentile"],
            "Diff_Tn_jour": tab["Mod_Tn"] - tab["TRACC_Tn"],
            "Diff_Tm_jour": tab["Mod_Tm"] - tab["TRACC_Tm"],
            "Diff_Tx_jour": tab["Mod_Tx"] - tab["TRACC_Tx"],
        })
        
        # Redéfinir num_cols_diff avant l'utilisation
        num_cols_diff = ["Diff_Tn_jour", "Diff_Tm_jour", "Diff_Tx_jour"]
        
        # Convertir en float + arrondir
        df_diff[num_cols_diff] = df_diff[num_cols_diff].apply(pd.to_numeric, errors="coerce").round(2)

    
        st.write(f"{mois} — Différences Modèle - TRACC (Tn_jour / Tmoy_jour / Tx_jour)")
    
        df_diff_styled = (
            df_diff.style
            .background_gradient(cmap="bwr", vmin=vminT, vmax=vmaxT, subset=["Diff_Tn_jour","Diff_Tm_jour","Diff_Tx_jour"])
            .format({col: "{:.2f}" for col in ["Diff_Tn_jour","Diff_Tm_jour","Diff_Tx_jour"]})
        )
        st.dataframe(df_diff_styled, hide_index=True)

    # ============================
    #  SECTION: Répartition annuelle des Tn / Tmoy / Tx journaliers
    # ============================
    st.subheader("Répartition annuelle des Tn_jour / Tmoy_jour / Tx_jour — CDF annuelle")
    
    # Concaténation annuelle des données journalières
    Tn_obs_year = np.concatenate(Tn_jour_all) if len(Tn_jour_all) > 0 else np.array([])
    Tm_obs_year = np.concatenate(Tm_jour_all) if len(Tm_jour_all) > 0 else np.array([])
    Tx_obs_year = np.concatenate(Tx_jour_all) if len(Tx_jour_all) > 0 else np.array[]
    
    Tn_mod_year = np.concatenate(Tn_jour_mod_all) if len(Tn_jour_mod_all) > 0 else np.array([])
    Tm_mod_year = np.concatenate(Tm_jour_mod_all) if len(Tm_jour_mod_all) > 0 else np.array([])
    Tx_mod_year = np.concatenate(Tx_jour_mod_all) if len(Tx_jour_mod_all) > 0 else np.array[]
    
    # Calcul des CDF annuelles (0-100%)
    pct_for_cdf = np.linspace(0, 100, 100)
    obs_tn_cdf_year = np.percentile(Tn_obs_year, pct_for_cdf) if Tn_obs_year.size > 0 else np.array([])
    mod_tn_cdf_year = np.percentile(Tn_mod_year, pct_for_cdf) if Tn_mod_year.size > 0 else np.array([])
    obs_tm_cdf_year = np.percentile(Tm_obs_year, pct_for_cdf) if Tm_obs_year.size > 0 else np.array([])
    mod_tm_cdf_year = np.percentile(Tm_mod_year, pct_for_cdf) if Tm_mod_year.size > 0 else np.array([])
    obs_tx_cdf_year = np.percentile(Tx_obs_year, pct_for_cdf) if Tx_obs_year.size > 0 else np.array([])
    mod_tx_cdf_year = np.percentile(Tx_mod_year, pct_for_cdf) if Tx_mod_year.size > 0 else np.array([])
    
    # Tracé des CDF annuelles sur un même graphique
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Couleurs cohérentes pour chaque variable
    colors = {
        "Tn": "cyan",
        "Tm": "white",
        "Tx": "red"
    }
    
    # Tracer Modèle (lignes pleines)
    ax.plot(pct_for_cdf, mod_tx_cdf_year, linestyle="-", linewidth=2, label=f"Modèle Tx", color=colors["Tx"])
    ax.plot(pct_for_cdf, mod_tm_cdf_year, linestyle="-", linewidth=2, label=f"Modèle Tmoy", color=colors["Tm"])
    ax.plot(pct_for_cdf, mod_tn_cdf_year, linestyle="-", linewidth=2, label=f"Modèle Tn", color=colors["Tn"])
    
    # Tracer TRACC (lignes pointillées)
    ax.plot(pct_for_cdf, obs_tx_cdf_year, linestyle="--", linewidth=1.7, label=f"TRACC Tx", color=colors["Tx"])
    ax.plot(pct_for_cdf, obs_tm_cdf_year, linestyle="--", linewidth=1.7, label=f"TRACC Tmoy", color=colors["Tm"])
    ax.plot(pct_for_cdf, obs_tn_cdf_year, linestyle="--", linewidth=1.7, label=f"TRACC Tn", color=colors["Tn"])
    
    # Mise en forme
    ax.set_title(f"Année complète — CDF Tn_jour / Tmoy_jour / Tx_jour (Modèle vs TRACC)", color="white")
    ax.set_xlabel("Percentile", color="white")
    ax.set_ylabel("Température (°C)", color="white")
    ax.tick_params(colors="white")
    ax.legend(facecolor="black", ncol=2)
    ax.set_facecolor("none")
    
    st.pyplot(fig)
    plt.close(fig)
    
    # Tableau des percentiles annuels
    st.write("Tableau des percentiles annuels (Tn_jour / Tmoy_jour / Tx_jour)")
    
    # Calcul des percentiles pour les valeurs définies
    pct_table = percentiles_list  # [10, 25, 50, 75, 90]
    
    percentiles_values = {
        "TRACC_Tn": np.round(np.percentile(Tn_obs_year, pct_table), 2) if Tn_obs_year.size > 0 else np.array([np.nan]*len(pct_table)),
        "Mod_Tn": np.round(np.percentile(Tn_mod_year, pct_table), 2) if Tn_mod_year.size > 0 else np.array([np.nan]*len(pct_table)),
        "TRACC_Tm": np.round(np.percentile(Tm_obs_year, pct_table), 2) if Tm_obs_year.size > 0 else np.array([np.nan]*len(pct_table)),
        "Mod_Tm": np.round(np.percentile(Tm_mod_year, pct_table), 2) if Tm_mod_year.size > 0 else np.array([np.nan]*len(pct_table)),
        "TRACC_Tx": np.round(np.percentile(Tx_obs_year, pct_table), 2) if Tx_obs_year.size > 0 else np.array([np.nan]*len(pct_table)),
        "Mod_Tx": np.round(np.percentile(Tx_mod_year, pct_table), 2) if Tx_mod_year.size > 0 else np.array([np.nan]*len(pct_table)),
    }
    
    tab_annuel = pd.DataFrame({
        "Percentile": [f"P{p}" for p in pct_table],
        "TRACC_Tn": percentiles_values["TRACC_Tn"],
        "Mod_Tn": percentiles_values["Mod_Tn"],
        "TRACC_Tm": percentiles_values["TRACC_Tm"],
        "Mod_Tm": percentiles_values["Mod_Tm"],
        "TRACC_Tx": percentiles_values["TRACC_Tx"],
        "Mod_Tx": percentiles_values["Mod_Tx"],
    })
    
    num_cols = tab_annuel.select_dtypes(include=[np.number]).columns
    tab_annuel[num_cols] = tab_annuel[num_cols].apply(pd.to_numeric, errors="coerce")
    styler = tab_annuel.style.format({col: "{:.2f}" for col in num_cols})
    st.dataframe(styler, hide_index=True)
    
    # Tableau des différences annuelles
    st.write("Différences annuelles (Modèle - TRACC)")
    
    df_diff_annuel = pd.DataFrame({
        "Percentile": [f"P{p}" for p in pct_table],
        "Diff_Tn_jour": percentiles_values["Mod_Tn"] - percentiles_values["TRACC_Tn"],
        "Diff_Tm_jour": percentiles_values["Mod_Tm"] - percentiles_values["TRACC_Tm"],
        "Diff_Tx_jour": percentiles_values["Mod_Tx"] - percentiles_values["TRACC_Tx"],
    })
    
    num_cols_diff = ["Diff_Tn_jour", "Diff_Tm_jour", "Diff_Tx_jour"]
    df_diff_annuel[num_cols_diff] = df_diff_annuel[num_cols_diff].apply(pd.to_numeric, errors="coerce").round(2)
    
    df_diff_annuel_styled = (
        df_diff_annuel.style
        .background_gradient(cmap="bwr", vmin=vminT, vmax=vmaxT, subset=num_cols_diff)
        .format({col: "{:.2f}" for col in num_cols_diff})
    )
    st.dataframe(df_diff_annuel_styled, hide_index=True)


    # ============================
    # GRAPHIQUES : Jours chauds et nuits tropicales par mois
    # ============================

    st.subheader("Graphiques : jours chauds et nuits tropicales par mois")
    
    # Choix seuil pour Tx
    tx_seuil = st.number_input("Seuil Tx_jour (°C) pour jours chauds :", min_value=-50.0, max_value=60.0, value=30.0, step=1.0)
    tn_seuil = st.number_input("Seuil Tn_jour (°C) pour nuits tropicales :", min_value=-50.0, max_value=60.0, value=20.0, step=1.0) 
    
    # Préparer listes pour stocker les valeurs par mois
    jours_chauds_tracc = []
    jours_chauds_modele = []
    nuits_tropicales_tracc = []
    nuits_tropicales_modele = []
    
    jours_chauds_total_tracc = 0
    jours_chauds_total_modele = 0
    nuits_tropicales_total_tracc = 0
    nuits_tropicales_total_modele = 0
    
    for mois_num in range(1, 13):
        # TRACC
        obs_tx_jour = Tx_jour_all[mois_num - 1]
        obs_tn_jour = Tn_jour_all[mois_num - 1]
        jours_tx = np.sum(obs_tx_jour > tx_seuil)
        nuits_trop = np.sum(obs_tn_jour > tn_seuil)
        jours_chauds_tracc.append(jours_tx)
        nuits_tropicales_tracc.append(nuits_trop)
        jours_chauds_total_tracc += jours_tx
        nuits_tropicales_total_tracc += nuits_trop
    
        # Modèle
        mod_tx_jour = Tx_jour_mod_all[mois_num - 1]
        mod_tn_jour = Tn_jour_mod_all[mois_num - 1]
        jours_tx_mod = np.sum(mod_tx_jour > tx_seuil)
        nuits_trop_mod = np.sum(mod_tn_jour > tn_seuil)
        jours_chauds_modele.append(jours_tx_mod)
        nuits_tropicales_modele.append(nuits_trop_mod)
        jours_chauds_total_modele += jours_tx_mod
        nuits_tropicales_total_modele += nuits_trop_mod
    
    # Labels pour les mois
    mois_labels = [mois_noms[m] for m in range(1, 13)]
    x = np.arange(len(mois_labels))
    
    # ---- Diagramme jours chauds ----
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.bar(x - 0.25, jours_chauds_tracc, width=0.5, color=couleur_TRACC, label="TRACC")
    ax.bar(x + 0.25, jours_chauds_modele, width=0.5, color=couleur_modele, label="Modèle")
    ax.set_xticks(x)
    ax.set_xticklabels(mois_labels, rotation=45)
    ax.set_ylabel(f"Nombre de jours Tx_jour > {tx_seuil}°C")
    ax.set_title("Jours chauds par mois")
    ax.legend()
    fig_jourschaud=fig
    st.pyplot(fig)
    plt.close(fig)
    
    # ---- Diagramme nuits tropicales ----
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.bar(x - 0.25, nuits_tropicales_tracc, width=0.5, color=couleur_TRACC, label="TRACC")
    ax.bar(x + 0.25, nuits_tropicales_modele, width=0.5, color=couleur_modele, label="Modèle")
    ax.set_xticks(x)
    ax.set_xticklabels(mois_labels, rotation=45)
    ax.set_ylabel(f"Nombre de nuits Tn_jour > {tn_seuil}°C")
    ax.set_title("Nuits tropicales par mois")
    ax.legend()
    fig_nuittrop=fig
    st.pyplot(fig)
    plt.close(fig)
    
    # ---- Affichage des totaux ----
    st.markdown(f"**Total jours chauds TRACC :** {jours_chauds_total_tracc}, **Modèle :** {jours_chauds_total_modele}")
    st.markdown(f"**Total nuits tropicales TRACC :** {nuits_tropicales_total_tracc}, **Modèle :** {nuits_tropicales_total_modele}")

    # =============================
    # Comparaison annuelle jours chauds / nuits tropicales
    # =============================
    
    # Jours chauds
    if jours_chauds_total_tracc > jours_chauds_total_modele:
        phrase_jours = f"TRACC enregistre plus de jours chauds (Tx>{tx_seuil}°C) sur l'année ({jours_chauds_total_tracc}) que le modèle ({jours_chauds_total_modele})."
    else:
        phrase_jours = f"Le modèle enregistre plus de jours chauds (Tx>{tx_seuil}°C) sur l'année ({jours_chauds_total_modele}) que TRACC ({jours_chauds_total_tracc})."
    
    # Nuits tropicales
    if nuits_tropicales_total_tracc > nuits_tropicales_total_modele:
        phrase_nuits = f"TRACC enregistre plus de nuits tropicales (Tn>{tn_seuil}°C) sur l'année ({nuits_tropicales_total_tracc}) que le modèle ({nuits_tropicales_total_modele})."
    else:
        phrase_nuits = f"Le modèle enregistre plus de nuits tropicales (Tn>{tn_seuil}°C) sur l'année ({nuits_tropicales_total_modele}) que TRACC ({nuits_tropicales_total_tracc})."
    
    # Stocker dans st.session_state pour la page Résumé
    st.session_state["resume_chaud_nuit"] = [phrase_jours, phrase_nuits]
    
    # Optionnel : affichage sur la page actuelle
    st.subheader("Résumé comparatif jours chauds / nuits tropicales")
    for p in st.session_state["resume_chaud_nuit"]:
        st.write("- " + p)
   
    # ============================
    # Calcul DJC (chauffage) et DJF (froid)
    # ============================
    
    st.subheader("DJC (chauffage) et DJF (froid) journaliers — TRACC vs Modèle")
    
    T_base_chauffage = float(st.text_input("Base DJC (°C) — chauffage", "19"))
    T_base_froid = float(st.text_input("Base DJF (°C) — refroidissement", "23"))
    
    results_djc = []
    results_djf = []
    mois_noms_sans_num = {
    1: "Janvier",   2: "Février",  3: "Mars",
    4: "Avril",     5: "Mai",      6: "Juin",
    7: "Juillet",   8: "Août",     9: "Septembre",
    10: "Octobre", 11: "Novembre", 12: "Décembre"
    }

    for mois_num in range(1, 13):
        mois = mois_noms_sans_num[mois_num]
    
        # Séries journalières déjà calculées
        Tx_tracc = Tx_jour_all[mois_num-1]
        Tn_tracc = Tn_jour_all[mois_num-1]
    
        idx0 = sum(heures_par_mois[:mois_num-1])
        idx1 = sum(heures_par_mois[:mois_num])
        model_hourly = model_values[idx0:idx1]
        Tx_mod, Tm_mod, Tn_mod = daily_stats_from_hourly(model_hourly)
    
        DJC_tracc_jours, DJF_tracc_jours = [], []
        DJC_mod_jours, DJF_mod_jours = [], []
    
        n_jours = len(Tx_tracc)
        for j in range(n_jours):
            Tm_tracc = (Tx_tracc[j] + Tn_tracc[j]) / 2
            DJC_tracc_jours.append(max(0, T_base_chauffage - Tm_tracc))
            DJF_tracc_jours.append(max(0, Tm_tracc - T_base_froid))
    
            if j < len(Tx_mod):
                Tm_mod = (Tx_mod[j] + Tn_mod[j]) / 2
                DJC_mod_jours.append(max(0, T_base_chauffage - Tm_mod))
                DJF_mod_jours.append(max(0, Tm_mod - T_base_froid))
            else:
                DJC_mod_jours.append(0)
                DJF_mod_jours.append(0)
    
        DJC_tracc_sum = float(np.nansum(DJC_tracc_jours))
        DJC_mod_sum = float(np.nansum(DJC_mod_jours))
        DJF_tracc_sum = float(np.nansum(DJF_tracc_jours))
        DJF_mod_sum = float(np.nansum(DJF_mod_jours))
    
        results_djc.append({
            "Mois": mois,
            "TRACC": DJC_tracc_sum,
            "Modèle": DJC_mod_sum,
            "Différence": DJC_mod_sum - DJC_tracc_sum
        })
        results_djf.append({
            "Mois": mois,
            "TRACC": DJF_tracc_sum,
            "Modèle": DJF_mod_sum,
            "Différence": DJF_mod_sum - DJF_tracc_sum
        })
    
    df_DJC = pd.DataFrame(results_djc).fillna(0)
    df_DJF = pd.DataFrame(results_djf).fillna(0)
    
    # Convertir explicitement les colonnes numériques en float
    for df in [df_DJC, df_DJF]:
        for col in ["TRACC", "Modèle", "Différence"]:
            df[col] = df[col].astype(float)
    
    # --------------------------
    # Affichage tables Streamlit
    # --------------------------
    st.subheader("DJU / DJC – Chauffage (somme journalière par mois)")
    st.dataframe(df_DJC.round(2))  # Arrondi à 2 décimales
    
    st.subheader("DJF – Refroidissement (somme journalière par mois)")
    st.dataframe(df_DJF.round(2))  # Arrondi à 2 décimales

    
    # --------------------------
    # Diagrammes bâtons mensuels
    # --------------------------
    st.subheader("Diagrammes bâtons mensuels — DJC et DJF")

    # Convertir en DataFrames
    df_DJC = pd.DataFrame(results_djc)
    df_DJF = pd.DataFrame(results_djf)
    
    # -----------------------------
    # Diagrammes en bâtons par mois
    # -----------------------------
    figures = {}   # dictionnaire où on stocke les figures

    for df, titre in zip([df_DJC, df_DJF], ["DJC", "DJF"]):
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.bar(df.index - 0.25, df["TRACC"], width=0.5,
               color=couleur_TRACC, label="TRACC")
        ax.bar(df.index + 0.25, df["Modèle"], width=0.5,
               color=couleur_modele, label="Modèle")
    
        ax.set_xticks(df.index)
        ax.set_xticklabels(df["Mois"])
        ax.set_title(f"{titre} mensuel — Modèle vs TRACC")
        ax.set_ylabel(f"{titre} (°C·jour)")
        ax.set_xlabel("Mois")
        ax.legend()
    
        # 🔥 enregistrer la figure dans le dictionnaire
        figures[titre] = fig
    
        st.pyplot(fig)
        plt.close(fig)

    # --------------------------
    # Somme annuelle DJC et DJF
    # --------------------------
    total_DJC_TRACC = df_DJC["TRACC"].sum()
    total_DJC_modele = df_DJC["Modèle"].sum()
    
    total_DJF_TRACC = df_DJF["TRACC"].sum()
    total_DJF_modele = df_DJF["Modèle"].sum()
    
    st.subheader("Sommes annuelles")
    st.write(f"DJC annuel : TRACC = {total_DJC_TRACC:.0f}    /    Modèle = {total_DJC_modele:.0f}")
    st.write(f"DJF annuel : TRACC = {total_DJF_TRACC:.0f}    /    Modèle = {total_DJF_modele:.0f}")

    # =============================
    # Résumé automatique DJC / DJF
    # =============================
    
    # DJC (chauffage)
    if total_DJC_TRACC > total_DJC_modele:
        phrase_djc = f"TRACC a une demande de chauffage annuelle plus élevée ({total_DJC_TRACC:.0f} °C·jour) que le modèle ({total_DJC_modele:.0f} °C·jour)."
    elif total_DJC_modele > total_DJC_TRACC:
        phrase_djc = f"Le modèle a une demande de chauffage annuelle plus élevée ({total_DJC_modele:.0f} °C·jour) que TRACC ({total_DJC_TRACC:.0f} °C·jour)."
    else:
        phrase_djc = "TRACC et le modèle ont la même demande de chauffage annuelle."
    
    # DJF (refroidissement)
    if total_DJF_TRACC > total_DJF_modele:
        phrase_djf = f"TRACC a une demande de refroidissement annuelle plus élevée ({total_DJF_TRACC:.0f} °C·jour) que le modèle ({total_DJF_modele:.0f} °C·jour)."
    elif total_DJF_modele > total_DJF_TRACC:
        phrase_djf = f"Le modèle a une demande de refroidissement annuelle plus élevée ({total_DJF_modele:.0f} °C·jour) que TRACC ({total_DJF_TRACC:.0f} °C·jour)."
    else:
        phrase_djf = "TRACC et le modèle ont la même demande de refroidissement annuelle."
    
    # Stocker dans st.session_state pour la page Résumé
    st.session_state["resume_djc_djf"] = [phrase_djc, phrase_djf]
    
    # Optionnel : affichage sur la page actuelle
    st.subheader("Résumé comparatif DJC / DJF")
    for p in st.session_state["resume_djc_djf"]:
        st.write("- " + p)

    # ======================================
    #  COURBES DES PERCENTILES PAR MOIS
    # ======================================
    st.subheader("Évolution mensuelle des percentiles (Modèle vs TRACC)")

    df_percentiles_all = []
    percentiles_list2 = [10,50,90]
    
    for mois_num in range(1, 13):
        mois = mois_noms[mois_num]
    
        # Observations
        obs_vals = obs_mois_all[mois_num-1]
    
        # Modèle
        idx0 = sum(heures_par_mois[:mois_num-1])
        idx1 = sum(heures_par_mois[:mois_num])
        mod_vals = model_values[idx0:idx1]

        
        # Ajout des percentiles
        for p in percentiles_list2:
            df_percentiles_all.append({
                "Mois": mois,
                "Percentile": f"P{p}",
                "Obs": np.percentile(obs_vals, p),
                "Mod": np.percentile(mod_vals, p)
            })

    # Table ordonnée pour faciliter les tracés
    df_percentiles_ordered = (
        pd.DataFrame(df_percentiles_all)
        .assign(Pnum=lambda d: d["Percentile"].str.extract("(\d+)").astype(int))
        .sort_values(["Pnum", "Mois"])
    )
    
    # Construction du graphique par percentile
    fig, ax = plt.subplots(figsize=(14,5))
    colors_perc = ["darkcyan", "khaki", "firebrick"]
    i=0
    for p in percentiles_list2:
        dfp = df_percentiles_ordered[df_percentiles_ordered["Pnum"] == p]
        # TRACC : ligne pointillée
        ax.plot(
            dfp["Mois"], dfp["Obs"],
            linestyle="--", label=f"TRACC P{p}", color=colors_perc[i]
        )
        # Modèle : ligne pleinne
        ax.plot(
            dfp["Mois"], dfp["Mod"],
            linestyle="-", label=f"Modèle P{p}", color=colors_perc[i]
        )
        i+=1
    
    ax.set_title(f"Percentiles {percentiles_list} – Modèle vs TRACC +{scenario_sel}/{ville_sel}")
    ax.set_ylabel("Température (°C)")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(ncol=2, facecolor="black")
    st.pyplot(fig)
    plt.close(fig)

    # -------- Calcul des percentiles P1 à P100 --------
    percentiles = np.arange(1, 101)
    P_obs = np.percentile(obs_annee, percentiles)
    P_mod = np.percentile(model_annee, percentiles)
    
    # -------- Graphique PXX modèle vs TRACC avec croix et couleurs conditionnelles --------
    fig, ax = plt.subplots(figsize=(6,6))
    
    # Définir les couleurs selon qui est plus chaud
    colors = ['lightgray' if obs > mod else 'goldenrod' for obs, mod in zip(P_obs, P_mod)]
    
    # Tracer les croix
    ax.scatter(P_obs, P_mod, color=colors, marker='x', s=50, label='Percentiles')
    
    # Diagonale y=x
    min_val = min(min(P_obs), min(P_mod))
    max_val = max(max(P_obs), max(P_mod))
    ax.plot([min_val, max_val], [min_val, max_val], color='white', linestyle='--', label='y=x')
    
    # Carré : même échelle sur x et y
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.set_aspect('equal', 'box')
    
    ax.set_xlabel("PXX TRACC (°C)")
    ax.set_ylabel("PXX Modèle (°C)")
    ax.set_title("Comparaison des percentiles annuels")
    ax.grid(True, linestyle=':', color='gray', alpha=0.5)
    ax.legend()
    st.pyplot(fig)

    # -------- Graphiques CDF et percentiles --------
    st.subheader("Fonctions de répartition mensuelles (CDF)")
    df_percentiles_all = []
    
    for mois_num in range(1, 13):
        mois = mois_noms[mois_num]
        obs_mois = obs_mois_all[mois_num-1]
        mod_mois = model_values[sum(heures_par_mois[:mois_num-1]):sum(heures_par_mois[:mois_num])]
        obs_percentiles_100 = np.percentile(obs_mois, np.linspace(0, 100, 100))
        mod_percentiles_100 = np.percentile(mod_mois, np.linspace(0, 100, 100))

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(np.linspace(0, 100, 100), mod_percentiles_100, label="Modèle", color=couleur_modele)
        ax.plot(np.linspace(0, 100, 100), obs_percentiles_100, label=f"TRACC +{scenario_sel}/{ville_sel}", color=couleur_TRACC)
        ax.set_title(f"{mois} - Fonction de répartition", color="white")
        ax.set_xlabel("Percentile", color="white")
        ax.set_ylabel("Température (°C)", color="white")
        ax.tick_params(colors="white")
        ax.legend(facecolor="black")
        ax.set_facecolor("none")
        st.pyplot(fig)
        plt.close(fig)

        obs_p = np.percentile(obs_mois, percentiles_list)
        mod_p = np.percentile(mod_mois, percentiles_list)
        df_p = pd.DataFrame({
            "Percentile": [f"P{p}" for p in percentiles_list],
            f"TRACC +{scenario_sel}/{ville_sel}": obs_p,
            "Modèle": mod_p
        }).round(2)
        st.write(f"{mois} - Percentiles")
        st.dataframe(df_p, hide_index=True)

        for i, p in enumerate(percentiles_list):
            df_percentiles_all.append({
                "Mois": mois,
                "Percentile": f"P{p}",
                "Obs": obs_p[i],
                "Mod": mod_p[i]
            })

    # -------- Fonction de répartition ANNUELLE --------
    st.subheader("Fonction de répartition annuelle (CDF)")
    
    # Regroupement annuel
    obs_annual = np.concatenate(obs_mois_all)         # Observations TRACC - toutes les heures de l'année
    mod_annual = model_values                         # Modèle : déjà toutes les heures
    
    # Percentiles pour CDF (0–100)
    percentiles_cdf = np.linspace(0, 100, 100)
    obs_percentiles_annual = np.percentile(obs_annual, percentiles_cdf)
    mod_percentiles_annual = np.percentile(mod_annual, percentiles_cdf)
    
    # ----- Plot de la CDF annuelle -----
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(percentiles_cdf, mod_percentiles_annual, label="Modèle", color=couleur_modele)
    ax.plot(percentiles_cdf, obs_percentiles_annual, label=f"TRACC +{scenario_sel}/{ville_sel}", color=couleur_TRACC)
    
    ax.set_title("Année entière - Fonction de répartition", color="white")
    ax.set_xlabel("Percentile", color="white")
    ax.set_ylabel("Température (°C)", color="white")
    ax.tick_params(colors="white")
    ax.legend(facecolor="black")
    ax.set_facecolor("none")
    
    fig_cdf = fig
    
    st.pyplot(fig)
    plt.close(fig)
    
    # ------ Tableau des percentiles annuels ------
    obs_p_annual = np.percentile(obs_annual, percentiles_list)
    mod_p_annual = np.percentile(mod_annual, percentiles_list)
    
    df_p_annual = pd.DataFrame({
        "Percentile": [f"P{p}" for p in percentiles_list],
        f"TRACC +{scenario_sel}/{ville_sel}": obs_p_annual,
        "Modèle": mod_p_annual
    }).round(2)
    
    st.write("Année entière - Percentiles")
    st.dataframe(df_p_annual, hide_index=True)


    st.subheader(f"Bilan modèle vs TRACC +{scenario_sel}/{ville_sel} (Modèle - TRACC)") 
    # Création du DataFrame
    df_bilan = pd.DataFrame(df_percentiles_all).round(2)
    df_bilan["Ecart"] = df_bilan["Mod"] - df_bilan["Obs"]
    # Extraire le numéro du percentile (5, 25, ...) pour imposer l'ordre
    df_bilan["Percentile_num"] = df_bilan["Percentile"].str.extract("(\d+)").astype(int)
    # Imposer l'ordre des percentiles
    df_bilan["Percentile"] = pd.Categorical(df_bilan["Percentile"], 
                                            categories=[f"P{p}" for p in percentiles_list], 
                                            ordered=True)
    # Pivot pour affichage
    df_bilan_pivot = df_bilan.pivot(index="Percentile", columns="Mois", values="Ecart").round(2)
    # Affichage stylé avec couleurs selon l'écart
    st.dataframe(
        df_bilan_pivot.style
        .background_gradient(cmap="bwr", vmin=vminT, vmax=vmaxT)
        .format("{:.2f}")
    )
    # -------- Section multi-scénarios pour la ville --------
    st.subheader(f"Comparaison multi-scénarios pour {ville_sel}")

    
    df_percentiles_scenarios = []
    for scenario in scenarios:
        nc_file = os.path.join(base_folder, scenario, f"{ville_sel}.nc")
        ds = xr.open_dataset(nc_file, decode_times=True)
        temps = ds["T2m"].to_series().values
        start_idx = 0
        for mois_num, nb_heures in enumerate(heures_par_mois, start=1):
            mois = mois_noms[mois_num]
            obs_mois = temps[start_idx:start_idx + nb_heures]
            obs_p = np.percentile(obs_mois, percentiles_list)
            for i, p in enumerate(percentiles_list):
                df_percentiles_scenarios.append({
                    "Scénario": scenario,
                    "Mois": mois,
                    "Percentile": f"P{p}",
                    "Valeur": round(obs_p[i],2)
                })
            start_idx += nb_heures
    
    df_scenarios = pd.DataFrame(df_percentiles_scenarios)
    
    # -------- Graphique CDF comparatif par scénario avec matplotlib --------
    st.subheader("CDF comparatif des 6 scénarios")
    
    scenario_pairs = [("2", "2_VC"), ("2-7", "2-7_VC"), ("4", "4_VC")]
    colors = ["green", "orange", "magenta"]  # couleur par paire
    
    for mois_num in range(1, 13):
        mois = mois_noms[mois_num]
    
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.set_ylim(-5, 45)
    
        # ------------------------------
        # EXTRACTION MODELE CSV (en blanc)
        # ------------------------------
        idx0 = sum(heures_par_mois[:mois_num - 1])
        idx1 = sum(heures_par_mois[:mois_num])
        mod_mois_csv = model_values[idx0:idx1]
    
        cdf_model = np.percentile(mod_mois_csv, np.linspace(0, 100, 100))
    
        ax.plot(
            np.linspace(0, 100, 100),
            cdf_model,
            label="Modèle",
            color="white",
            linewidth=2,
            linestyle="-"
        )
    
        # ------------------------------
        # COURBES DES SCÉNARIOS
        # ------------------------------
        for i, (sc1, sc2) in enumerate(scenario_pairs):
            ax.set_ylim(-5, 45)
            color = colors[i]
    
            # ---- Scénario 1 (trait plein) ----
            nc_file = os.path.join(base_folder, sc1, f"{ville_sel}.nc")
            ds = xr.open_dataset(nc_file, decode_times=True)
            temp = ds["T2m"].to_series().values
            mod_mois = temp[idx0:idx1]
            cdf_values = np.percentile(mod_mois, np.linspace(0, 100, 100))
    
            ax.plot(
                np.linspace(0, 100, 100),
                cdf_values,
                label=f"{sc1}",
                color=color,
                linestyle="-"
            )
    
            # ---- Scénario 2 (pointillé) ----
            nc_file = os.path.join(base_folder, sc2, f"{ville_sel}.nc")
            ds = xr.open_dataset(nc_file, decode_times=True)
            temp = ds["T2m"].to_series().values
            mod_mois = temp[idx0:idx1]
            cdf_values = np.percentile(mod_mois, np.linspace(0, 100, 100))
    
            ax.plot(
                np.linspace(0, 100, 100),
                cdf_values,
                label=f"{sc2}",
                color=color,
                linestyle="--"
            )
    
        # ------------------------------
        # Mise en forme
        # ------------------------------
        ax.set_title(f"{mois} - CDF comparatif par scénario", color="white")
        ax.set_xlabel("Percentile", color="white")
        ax.set_ylabel("Température (°C)", color="white")
        ax.tick_params(colors="white")
        ax.legend(facecolor="black")
        ax.set_facecolor("none")
    
        st.pyplot(fig)
        plt.close(fig)

    # -------- Fonction de répartition ANNUELLE pour tous les scénarios --------
    st.subheader(f"Fonction de répartition annuelle (CDF) - {ville_sel}")
    
    # Percentiles pour CDF (0–100)
    percentiles_cdf = np.linspace(0, 100, 100)
    
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_ylim(-5, 45)
    
    # Définition des paires et des couleurs pour 6 scénarios
    scenario_pairs = [("2", "2_VC"), ("2-7", "2-7_VC"), ("4", "4_VC")]
    colors = ["green", "orange", "magenta"]
    
    # Boucle sur toutes les paires pour les 6 scénarios
    for i, (sc1, sc2) in enumerate(scenario_pairs):
        color = colors[i]
    
        # Scénario 1 (plein)
        nc_file = os.path.join(base_folder, sc1, f"{ville_sel}.nc")
        if os.path.exists(nc_file):
            ds = xr.open_dataset(nc_file, decode_times=True)
            temp_annee = ds["T2m"].to_series().values
            if len(temp_annee) > 0:
                cdf_values = np.percentile(temp_annee, percentiles_cdf)
                ax.plot(percentiles_cdf, cdf_values, label=sc1, color=color, linestyle="-")
    
        # Scénario 2 (pointillé)
        nc_file = os.path.join(base_folder, sc2, f"{ville_sel}.nc")
        if os.path.exists(nc_file):
            ds = xr.open_dataset(nc_file, decode_times=True)
            temp_annee = ds["T2m"].to_series().values
            if len(temp_annee) > 0:
                cdf_values = np.percentile(temp_annee, percentiles_cdf)
                ax.plot(percentiles_cdf, cdf_values, label=sc2, color=color, linestyle="--")
    
    # CDF du modèle
    if 'model_values' in locals() and len(model_values) > 0:
        mod_percentiles_annual = np.percentile(model_values, percentiles_cdf)
        ax.plot(percentiles_cdf, mod_percentiles_annual, label="Modèle", color=couleur_modele, linewidth=2, linestyle="-")
    
    ax.set_title("Année entière - Fonction de répartition (CDF) pour tous les scénarios", color="white")
    ax.set_xlabel("Percentile", color="white")
    ax.set_ylabel("Température (°C)", color="white")
    ax.tick_params(colors="white")
    ax.legend(facecolor="black")
    ax.set_facecolor("none")
    
    st.pyplot(fig)
    plt.close(fig)
    
    # -------- Heatmap des écarts des percentiles par mois et scénario --------
    st.subheader(f"Ecarts des percentiles (Modèle - Scénarios TRACC)")
    
    # Création du dictionnaire de référence Modèle
    ref_model = {}
    for mois_num in range(1, 13):
        mois = mois_noms[mois_num]
        obs_mois = obs_mois_all[mois_num-1]
        mod_mois = model_values[sum(heures_par_mois[:mois_num-1]):sum(heures_par_mois[:mois_num])]
        for i, p in enumerate(percentiles_list):
            ref_model[(mois, f"P{p}")] = np.percentile(mod_mois, p)
    
    for p in percentiles_list:
        df_ecart = df_scenarios[df_scenarios["Percentile"] == f"P{p}"].copy()
        df_ecart["Ecart"] = -df_ecart.apply(lambda row: row["Valeur"] - ref_model[(row["Mois"], f"P{p}")], axis=1)
        df_ecart["Ecart"] = df_ecart["Ecart"].round(2).astype(float)
        df_pivot = df_ecart.pivot(index="Scénario", columns="Mois", values="Ecart").round(2)
        st.write(f"Percentile {p} : Modèle - TRACC/{ville_sel}")
        st.dataframe(df_pivot.style.background_gradient(cmap="bwr", vmin=vminT, vmax=vmaxT).format("{:.2f}"))


    # ---- Stockage des figures dans session_state ----
    st.session_state["fig_hist_year"] = fig_hist_year
    st.session_state["fig_hist_diff"] = fig_hist_diff
    st.session_state["df_rmse"] = df_rmse
    st.session_state["df_rmse_styled"] = df_rmse_styled
    st.session_state["fig_tn_tx_mois"] = fig_tn_tx_mois
    st.session_state["fig_jourschaud"] = fig_jourschaud
    st.session_state["fig_nuittrop"] = fig_nuittrop
    st.session_state["fig_cdf"] = fig_cdf
    st.session_state["fig_DJC"] = figures["DJC"]
    st.session_state["fig_DJF"] = figures["DJF"]






