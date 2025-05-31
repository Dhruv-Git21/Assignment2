# Replacing Sentence Transformers with TF-IDF
import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
from transformers import pipeline

import os
import asyncio
import sys

if sys.platform.startswith('win'):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


@st.cache_resource
def load_summarizer():
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    return summarizer

def load_data(uploaded_files):
    dfs = []
    for idx, file in enumerate(uploaded_files):
        df = pd.read_excel(file)
        df['Analyst'] = f'Person {idx+1}'
        dfs.append(df[['Notes', 'Analyst']])
    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df.dropna(subset=['Notes'], inplace=True)
    combined_df.reset_index(drop=True, inplace=True)
    return combined_df

# Use TF-IDF + Agglomerative Clustering
def cluster_notes(notes_list, threshold=0.75):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(notes_list)
    
    cosine_sim_matrix = cosine_similarity(tfidf_matrix)
    
    # Agglomerative Clustering
    clustering_model = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=1-threshold,
        metric='precomputed', 
        linkage='average'
    )
    clustering_model.fit(1 - cosine_sim_matrix)  # 1 - similarity = distance
    cluster_assignment = clustering_model.labels_

    clustered_notes = {}
    for idx, cluster_id in enumerate(cluster_assignment):
        clustered_notes.setdefault(cluster_id, []).append(notes_list[idx])

    return list(clustered_notes.values())

def summarize_notes(notes_cluster, summarizer, max_length=100):
    combined_text = " ".join(notes_cluster)
    try:
        summary = summarizer(combined_text, max_length=max_length, min_length=15, do_sample=False)[0]['summary_text']
    except:
        summary = "Summary could not be generated (too long)."
    return summary

def categorize(text):
    categories = {
        'Banking': ['Bank', 'Loan', 'Finance', 'Yes Bank', 'Union Bank'],
        'EV': ['Electric Vehicle', 'EV', 'Ather', 'Two Wheeler'],
        'Trade Deals': ['Free Trade Agreement', 'FTA', 'UK', 'Trade'],
        'Tech': ['Swiggy', 'Uber', 'Paytm', 'One97'],
        'Macro': ['Human Development Index', 'Services Sector', 'Inflation'],
        'Hospitality': ['Hilton', 'Hotel']
    }
    tags = []
    for cat, keywords in categories.items():
        for keyword in keywords:
            if keyword.lower() in text.lower():
                tags.append(cat)
                break
    if not tags:
        tags.append('Others')
    return tags

def generate_title(summary, max_words=12):
    words = summary.split()
    if len(words) <= max_words:
        return summary
    else:
        return ' '.join(words[:max_words]) + '...'

import datetime

def generate_email_template(summaries):
    today = datetime.date.today().strftime("%B %d, %Y")
    html_template = f"""
    <html>
    <body>
        <h2>🗞️ Daily Research Highlights — {today}</h2>
        <hr>
    """
    for idx, row in summaries.iterrows():
        title = generate_title(row['Summary'])
        html_template += f"""
        <h4><b>{idx+1}. {title}</b></h4>
        <p>{row['Summary']}</p>
        <p><b>Categories:</b> {', '.join(row['Tags'])}</p>
        <hr>
        """
    
    html_template += """
        <p>Best Regards,</p>
        <p><b>Research Team</b></p>
    </body>
    </html>
    """
    return html_template

def main():
    st.set_page_config(page_title="Daily Research Summaries", layout="wide")
    st.title("📊 Daily Research Summaries Assignment")

    summarizer = load_summarizer()

    st.sidebar.header("Upload Analyst Inputs")
    uploaded_files = st.sidebar.file_uploader("Upload Excel Files", type=["xlsx"], accept_multiple_files=True)

    if uploaded_files:
        st.success(f"{len(uploaded_files)} files uploaded successfully!")

        if st.sidebar.button("Summarize Notes"):
            df = load_data(uploaded_files)
            notes_list = df['Notes'].tolist()

            st.info("Clustering similar news topics using TF-IDF...")
            clusters = cluster_notes(notes_list, threshold=0.75)

            summarized_notes = []
            tags_list = []

            st.info("Summarizing the clustered notes...")

            for cluster in clusters:
                summary = summarize_notes(cluster, summarizer)
                tags = categorize(summary)
                summarized_notes.append(summary)
                tags_list.append(tags)

            final_df = pd.DataFrame({'Summary': summarized_notes, 'Tags': tags_list})

            all_tags = sorted(set(tag for sublist in tags_list for tag in sublist))
            selected_tags = st.sidebar.multiselect("Select Categories", all_tags, default=all_tags)

            filtered_df = final_df[final_df['Tags'].apply(lambda tags: any(tag in selected_tags for tag in tags))]

            st.subheader("Summarized Insights (After TF-IDF Clustering)")

            for idx, row in filtered_df.iterrows():
                st.markdown(f"### 📝 Insight {idx+1}")
                st.write(row['Summary'])
                st.caption(f"**Tags:** {', '.join(row['Tags'])}")
                st.markdown("---")

            st.download_button("Download Summarized Report", filtered_df.to_csv(index=False), "summarized_report.csv", "text/csv")

            # Newsletter
            email_template = generate_email_template(filtered_df)

            st.subheader("📩 Preview: Automated Newsletter Email Template")
            st.components.v1.html(email_template, height=600, scrolling=True)

            st.download_button("Download Email Template (HTML Format)", email_template, file_name="newsletter.html")
    else:
        st.warning("Please upload at least one Excel file to proceed.")

# Run the app
if __name__ == "__main__":
    main()
