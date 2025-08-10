import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from googleapiclient.discovery import build
from sklearn.linear_model import LinearRegression
from datetime import datetime
import base64

# ---- SETUP ----
st.set_page_config(page_title="YouTube Search Trends Analyzer", layout="wide")
st.title("📺 YouTube Search Trends Analyzer with ML Forecasting & Video Insights")

API_KEY = "AIzaSyDnWkgu1dVleOPAfw1urFfv1k7PLb7IKWc"  # Replace with your YouTube API Key
youtube = build("youtube", "v3", developerKey=API_KEY)

keywords_input = st.text_input("🔑 Enter keyword(s) separated by commas (e.g. AI, Python)", "AI")
keywords = [kw.strip() for kw in keywords_input.split(",") if kw.strip()]

# ---- FUNCTIONS ----
def get_video_data(keyword, max_results=50):
    all_videos = []
    next_page_token = None

    while len(all_videos) < max_results:
        request = youtube.search().list(
            q=keyword,
            part="id",
            type="video",
            maxResults=min(50, max_results - len(all_videos)),
            pageToken=next_page_token
        )
        response = request.execute()

        video_ids = [item["id"]["videoId"] for item in response["items"]]
        if not video_ids:
            break

        stats_request = youtube.videos().list(
            part="statistics,snippet",
            id=",".join(video_ids)
        )
        stats_response = stats_request.execute()

        for item in stats_response["items"]:
            snippet = item["snippet"]
            stats = item["statistics"]
            video = {
                "video_id": item["id"],
                "title": snippet["title"],
                "description": snippet.get("description", "")[:150],
                "thumbnail": snippet["thumbnails"]["high"]["url"],
                "published_at": pd.to_datetime(snippet["publishedAt"]).date(),
                "views": int(stats.get("viewCount", 0))
            }
            all_videos.append(video)

        next_page_token = response.get("nextPageToken")
        if not next_page_token:
            break

    df = pd.DataFrame(all_videos)
    return df

def classify_trend(series):
    if len(series) < 7:
        return "Not enough data"
    y = series[-7:].values.reshape(-1, 1)
    X = np.arange(len(y)).reshape(-1, 1)
    model = LinearRegression().fit(X, y)
    slope = model.coef_[0][0]
    if slope > 1000:
        return "📈 Rising"
    elif slope < -1000:
        return "📉 Declining"
    else:
        return "➖ Stable"

def forecast_views(df):
    df = df.rename(columns={"published_at": "ds", "views": "y"})
    df = df[df["y"] > 0]
    if len(df) < 10:
        return None
    m = Prophet(daily_seasonality=True)
    m.fit(df)
    future = m.make_future_dataframe(periods=7)
    forecast = m.predict(future)
    return m, forecast

def display_top_videos(df):
    st.subheader("🎬 Top 3 Recent Videos")
    top_videos = df.sort_values(by="views", ascending=False).head(3)
    for _, row in top_videos.iterrows():
        col1, col2 = st.columns([1, 4])
        with col1:
            st.image(row["thumbnail"], width=150)
        with col2:
            st.markdown(f"**[{row['title']}]"
                        f"(https://www.youtube.com/watch?v={row['video_id']})")
            st.caption(row["description"])
            st.write(f"🗓 Published: {row['published_at']} | 👁 Views: {row['views']:,}")

# ---- APP LOGIC ----
if keywords:
    for kw in keywords:
        st.markdown(f"---\n### 🔍 Results for: `{kw}`")

        with st.spinner("Fetching video data..."):
            video_df = get_video_data(kw, max_results=100)

        if video_df.empty:
            st.warning("No videos found for this keyword.")
            continue

        # Show top videos
        display_top_videos(video_df)

        # Aggregate for daily views
        daily_views = video_df.groupby("published_at")["views"].sum().reset_index()

        st.subheader("📊 Daily Video Views")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(daily_views["published_at"], daily_views["views"], marker='o')
        ax.set_title(f"Total Daily Views for Videos Matching '{kw}'")
        ax.set_xlabel("Date")
        ax.set_ylabel("Views")
        ax.grid(True)
        st.pyplot(fig)

        trend = classify_trend(daily_views["views"])
        st.header(f"**📈 Trend Classification:** `{trend}`")

        st.subheader("🔮 7-Day Forecast (Prophet Model)")
        with st.spinner("Forecasting with Prophet..."):
            result = forecast_views(daily_views)
            if result:
                model, forecast = result
                fig2 = model.plot(forecast)
                st.pyplot(fig2)
            else:
                st.info("Not enough data to make a forecast.")

else:
    st.info("Please enter keyword(s) to begin.")

