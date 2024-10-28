import streamlit as st
import seaborn as sns
from transformers import pipeline
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import re
from nltk.corpus import stopwords
import nltk
from wordcloud import WordCloud
import matplotlib.patches as patches
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from wordcloud import WordCloud
import re

# Téléchargement des stop words en français
nltk.download("stopwords")
stop_words = set(stopwords.words("french"))
custom_exclusions = {
    "maroc",
    "tanger",
    "port",
    "med",
    "a",
    "être",
    "etre",
    "souvent",
    "gestion",
}

# Configuration de la page Streamlit
st.set_page_config(page_title="Tanger Med - Analyse des Commentaires", layout="wide")

# Style personnalisé
st.markdown(
    """
    <style>
        .reportview-container {background-color: #1E1E1E; color: #FFFFFF;}
        h1, h2, h3, h4 {color: #004080;}
        .positive-title {color: #66B2FF; font-size: 20px; font-weight: bold;}
        .negative-title {color: red; font-size: 20px; font-weight: bold;}
        .percentage {color: black; font-size: 18px;}
        .stButton>button {background-color: #004080; color: white;}
        .stButton>button:hover {background-color: #007FFF;}
        .graph-container {border: 2px solid #004080; border-radius: 10px; padding: 10px;}
        .wordcloud {border: 2px solid #004080; border-radius: 10px; padding: 10px;}
        .spacer {height: 30px;}
    </style>
""",
    unsafe_allow_html=True,
)

# Titre de l'application
st.markdown(
    """
    <h1 style='text-align: center; font-size: 40px; color: #004080;'>
        🚢 Tanger Med - Analyse des Commentaires
    </h1>
""",
    unsafe_allow_html=True,
)

# Chargement du modèle de classification
classifier = pipeline("text-classification", model="Ikram111/camembert_sentiment_model")


# Option 1 : Analyse en temps réel
st.markdown(
    "<span style='color: #D17F36; font-size: 31px;'>📝 Option 1 : Analyse en Temps Réel</span>",
    unsafe_allow_html=True,
)
user_input = st.text_area(
    "Entrez un ou plusieurs commentaires (séparés par un saut de ligne uniquement) :"
)

if st.button("Analyser"):
    commentaires = re.split(r"\n+", user_input)
    commentaires = [comment.strip() for comment in commentaires if comment.strip()]

    if commentaires:
        predictions = []
        for idx, commentaire in enumerate(commentaires, 1):
            prediction = classifier(commentaire)[0]["label"]
            sentiment = "Positif" if prediction == "LABEL_1" else "Négatif"
            predictions.append(f"Commentaire {idx} : Sentiment prédit -> {sentiment}")

        st.markdown("**Résultats des prédictions pour les commentaires**")
        for pred in predictions:
            st.markdown(pred)

st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown(
    "<span style='color: #D17F36; font-size: 31px;'>📁 Option 2 : Importer un Fichier CSV</span>",
    unsafe_allow_html=True,
)

# Option 2 : Importation du fichier CSV
uploaded_file = st.file_uploader("Choisissez un fichier CSV", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)

        if "comments" not in df.columns:
            st.error(
                "Le fichier ne contient pas la colonne 'comments'. Veuillez vérifier."
            )
        else:
            st.success("Fichier chargé avec succès !")
            st.write(df.head())

            # Prédire
            if st.button("Prédire"):
                df["prediction"] = df["comments"].apply(
                    lambda x: classifier(x)[0]["label"]
                )
                label_mapping = {"LABEL_0": "Négatif", "LABEL_1": "Positif"}
                df["prediction"] = df["prediction"].map(label_mapping)

                st.markdown("<br>", unsafe_allow_html=True)

                # Affichage des résultats de classification
                st.subheader("Résultats de Classification:")
                st.write(
                    df[["comments", "prediction"]]
                )  # Affichez les commentaires avec leurs prédictions

                # Calculer les pourcentages
                total_comments = len(df)
                positif_count = df["prediction"].value_counts().get("Positif", 0)
                negatif_count = df["prediction"].value_counts().get("Négatif", 0)

                if total_comments > 0:
                    positif_percentage = (positif_count / total_comments) * 100
                    negatif_percentage = (negatif_count / total_comments) * 100

                    # Utilisation de session_state pour mémoriser les données
                    if "class_counts" not in st.session_state:
                        st.session_state.class_counts = df["prediction"].value_counts()

                    # Affichage des résultats en utilisant les deux graphiques côte à côte
                    st.subheader("Graphique des prédictions:")
                    class_counts = st.session_state.class_counts

                    st.markdown("<br>", unsafe_allow_html=True)

                    # Initialisation des graphiques
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

                    # Graphique en barres
                    colors = [
                        "red" if label == "Négatif" else "blue"
                        for label in class_counts.index
                    ]
                    ax1.bar(class_counts.index, class_counts.values, color=colors)

                    ax1.spines["top"].set_visible(False)
                    ax1.spines["right"].set_visible(False)
                    ax1.spines["left"].set_visible(True)
                    ax1.spines["bottom"].set_visible(True)

                    ax1.set_xlabel("Classe")
                    ax1.set_ylabel("Nombre de commentaires")
                    ax1.annotate(
                        "Distribution des Commentaires par Classe",
                        xy=(0.5, -0.2),
                        xycoords="axes fraction",
                        ha="center",
                        fontsize=16,
                    )

                    # Graphique circulaire
                    colors = ["blue", "red"]
                    ax2.pie(
                        class_counts,
                        labels=class_counts.index,
                        autopct="%1.1f%%",
                        colors=colors,
                    )
                    ax2.annotate(
                        "Répartition des Sentiments",
                        xy=(0.5, -0.2),
                        xycoords="axes fraction",
                        ha="center",
                        fontsize=16,
                    )

                    # Afficher les graphiques
                    st.pyplot(fig)

                    # Affichage des pourcentages sous le graphique
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.subheader("Pourcentages des Sentiments:")
                    st.markdown(
                        "<span class='percentage'>• Pourcentage Positif : {:.2f}%</span>".format(
                            positif_percentage
                        ),
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        "<span class='percentage'>• Pourcentage Négatif : {:.2f}%</span>".format(
                            negatif_percentage
                        ),
                        unsafe_allow_html=True,
                    )

                    # Extraction des mots fréquents et génération de nuages de mots
                    def get_frequent_words(texts):
                        all_words = " ".join(texts).lower()
                        words = re.findall(r"\w+", all_words)
                        filtered_words = [
                            word
                            for word in words
                            if word not in stop_words and word not in custom_exclusions
                        ]
                        return Counter(filtered_words)

                    positif_texts = df[df["prediction"] == "Positif"][
                        "comments"
                    ].tolist()
                    negatif_texts = df[df["prediction"] == "Négatif"][
                        "comments"
                    ].tolist()

                    st.markdown("<br>", unsafe_allow_html=True)

                    st.subheader("Mots Fréquents dans les Commentaires Classifiés:")

                    st.markdown("<br>", unsafe_allow_html=True)

                    # Affichage des nuages de mots un sur l'autre
                    if positif_texts:
                        st.markdown(
                            """
                            <div style='text-align: center;'>
                                 <span style='font-size: 30px; color:#00BFFF;'> <strong>Mots Fréquents pour la Classe Positive</strong></span>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                        positif_words = get_frequent_words(positif_texts)
                        positif_wordcloud = WordCloud(
                            width=400,
                            height=240,
                            background_color="white",
                            colormap="Blues",
                        ).generate_from_frequencies(dict(positif_words))
                        plt.figure(figsize=(6, 4))
                        plt.imshow(positif_wordcloud, interpolation="bilinear")
                        plt.axis("off")
                        st.pyplot(plt.gcf())

                        st.markdown("<br>", unsafe_allow_html=True)

                    if negatif_texts:
                        st.markdown(
                            """
                            <div style='text-align: center;'>
                                 <span style='font-size: 30px; color:red;'><strong>Mots Fréquents pour la Classe Négative</strong></span>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                        negatif_words = get_frequent_words(negatif_texts)
                        negatif_wordcloud = WordCloud(
                            width=400,
                            height=200,
                            background_color="white",
                            colormap="Reds",
                        ).generate_from_frequencies(dict(negatif_words))
                        plt.figure(figsize=(6, 4))
                        plt.imshow(negatif_wordcloud, interpolation="bilinear")
                        plt.axis("off")

                        # Affichage du nuage de mots avec un cadre
                        st.pyplot(plt.gcf())
                        st.markdown("<br>", unsafe_allow_html=True)

                # Téléchargement des résultats
                st.subheader("Télécharger les résultats de classification:")
                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Télécharger le fichier CSV",
                    data=csv,
                    file_name="resultats_classification.csv",
                    mime="text/csv",
                )

    except Exception as e:
        st.error(f"Une erreur s'est produite : {e}")

# Instructions supplémentaires pour l'utilisateur, mieux organisées et stylisées
st.sidebar.title("📋 Instructions d'Utilisation")

st.sidebar.markdown(
    "<span style='color: #D17F36; font-size: 24px;'>📝 Option 1 : Analyse en Temps Réel</span>",
    unsafe_allow_html=True,
)
st.sidebar.write(
    """
- Entrez vos commentaires dans le champ ci-dessus, séparés par des sauts de ligne.
- Cliquez sur "Analyser" pour obtenir les prédictions de sentiment.
"""
)

st.sidebar.markdown(
    "<span style='color: #D17F36; font-size: 24px;'>📁Option 2 : Importer un Fichier CSV </span>",
    unsafe_allow_html=True,
)

st.sidebar.write(
    """
- Téléchargez votre fichier CSV contenant une colonne nommée 'comments'.
- Cliquez sur "Prédire" pour classer les sentiments de tous les commentaires.
- Vous pouvez également télécharger les résultats de la classification au format CSV.
"""
)
