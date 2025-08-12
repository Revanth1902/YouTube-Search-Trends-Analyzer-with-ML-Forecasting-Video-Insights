import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from googleapiclient.discovery import build
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta
from textblob import TextBlob
from dotenv import load_dotenv

st.set_page_config(page_title="YouTube Analyzer", layout="wide")
st.title("📺 YouTube Trends Analyzer with Forecast & Sentiment")

load_dotenv()
API_KEY = os.getenv('API_KEY')

youtube = build("youtube", "v3", developerKey=API_KEY)

keywords_input = st.text_input("🔑 Enter keyword(s) separated by commas (e.g. AI, Python)", "AI")
keywords = [kw.strip() for kw in keywords_input.split(",") if kw.strip()]



def get_video_data(keyword, max_results=100):
    videos = []
    next_token = None
    while len(videos) < max_results:
        search = youtube.search().list(q=keyword, part="id", type="video",
                                       maxResults=min(50, max_results - len(videos)),
                                       pageToken=next_token).execute()
        ids = [item["id"]["videoId"] for item in search.get("items", [])]
        if not ids:
            break
        video_details = youtube.videos().list(part="statistics,snippet", id=",".join(ids)).execute()
        for item in video_details["items"]:
            stats = item["statistics"]
            snippet = item["snippet"]
            videos.append({
                "video_id": item["id"],
                "title": snippet["title"],
                "description": snippet.get("description", "")[:150],
                "published_at": pd.to_datetime(snippet["publishedAt"]).date(),
                "thumbnail": snippet["thumbnails"]["high"]["url"],
                "views": int(stats.get("viewCount", 0))
            })
        next_token = search.get("nextPageToken")
        if not next_token:
            break
    return pd.DataFrame(videos)

def classify_trend(series):
    if len(series) < 7:
        return "Not enough data"
    y = series[-7:].values.reshape(-1, 1)
    X = np.arange(len(y)).reshape(-1, 1)
    slope = LinearRegression().fit(X, y).coef_[0][0]
    return "📈 Rising" if slope > 1000 else "📉 Declining" if slope < -1000 else "➖ Stable"

def forecast_views(df):
    df = df.rename(columns={"published_at": "ds", "views": "y"})
    df = df[df["y"] > 0]
    if len(df) < 14:
        return None
    m = Prophet(daily_seasonality=True)
    m.fit(df)
    future = m.make_future_dataframe(periods=7)
    forecast = m.predict(future)
    return m, forecast

def format_views(n):
    return f"{n / 1_000_000:.1f}M" if n >= 1_000_000 else f"{n / 1_000:.1f}K" if n >= 1_000 else str(n)

def get_comments(video_id, max_comments=50):
    comments = []
    try:
        req = youtube.commentThreads().list(part="snippet", videoId=video_id,
                                            maxResults=100, textFormat="plainText").execute()
        for item in req.get("items", []):
            text = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            comments.append(text)
            if len(comments) >= max_comments:
                break
    except:
        pass
    return comments

def analyze_sentiment(comments):
    results = {"Positive": 0, "Neutral": 0, "Negative": 0}
    for c in comments:
        polarity = TextBlob(c).sentiment.polarity
        if polarity > 0.1:
            results["Positive"] += 1
        elif polarity < -0.1:
            results["Negative"] += 1
        else:
            results["Neutral"] += 1
    return results

def display_top_videos(df):
    
    st.subheader("🎬 Top 3 Recent Videos")
    top = df.sort_values("views", ascending=False).head(3)
    for _, row in top.iterrows():
        c1, c2, c3 = st.columns([1, 3, 2])
        with c1:
            st.image(row["thumbnail"], width=150)
        with c2:
            st.markdown(f"**[{row['title']}](https://www.youtube.com/watch?v={row['video_id']})**")
            st.caption(row["description"])
            st.write(f"🗓 Published: {row['published_at']} | 👁 Views: {format_views(row['views'])}")
        with c3:
            comments = get_comments(row["video_id"])
            if not comments:
                st.info("No comments")
            else:
                sentiments = analyze_sentiment(comments)
                fig, ax = plt.subplots(figsize=(2.5, 2.5))
                ax.pie(list(sentiments.values()), labels=list(sentiments.keys()),
                       colors=['green', 'gray', 'red'], autopct='%1.1f%%', startangle=140)
                ax.axis('equal')
                st.pyplot(fig)


# --- APP LOGIC ---
if keywords:
    for kw in keywords:
        st.markdown(f"---\n### 🔍 Results for: `{kw}`")
        
        with st.spinner(f"Fetching videos for '{kw}'..."):
            video_df = get_video_data(kw, max_results=100)

        if video_df.empty:
            st.warning("No videos found.")
            continue

        display_top_videos(video_df)

        # Daily view aggregation (only last 7 days)
        daily_views = video_df.groupby("published_at")["views"].sum().reset_index()
        daily_views = daily_views.sort_values("published_at", ascending=True)
        last_7_days = daily_views[daily_views["published_at"] >= (daily_views["published_at"].max() - pd.Timedelta(days=6))]

        st.subheader("📊 Daily View Trends (Last 7 Days)")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(last_7_days["published_at"], last_7_days["views"], marker='o', color='dodgerblue')
        ax.set_xlabel("Date")
        ax.set_ylabel("Views")
        ax.set_title("Total Daily Views (Past 7 Days)")
        ax.grid(True)
        st.pyplot(fig)

        trend = classify_trend(daily_views["views"])
        st.markdown(f"""<div style='
    background-color:#f0f8ff;
    padding: 10px;
    border-radius: 8px;
    font-size: 22px;
    font-weight: bold;
    text-align: center;
'>
📉 Trend Analysis: <code>{trend}</code>
</div>
""", unsafe_allow_html=True)





        st.subheader("🔮 View Forecast (Past 7 + Next 7 Days)")
        with st.spinner("Building forecast model..."):
            result = forecast_views(daily_views)

        if result:
            model, forecast = result

            today = datetime.today().date()
            past_week = forecast[forecast["ds"].dt.date.between(today - timedelta(days=7), today)]
            future_week = forecast[forecast["ds"].dt.date > today]

            fig2, ax2 = plt.subplots(figsize=(10, 5))
            ax2.plot(past_week["ds"], past_week["yhat"], label="📊 Past 7 Days (Estimated)", color="skyblue")
            ax2.plot(future_week["ds"], future_week["yhat"], label="🔮 Next 7 Days (Forecast)", color="orange")
            ax2.fill_between(future_week["ds"], future_week["yhat_lower"], future_week["yhat_upper"],
                             color="orange", alpha=0.2)
            ax2.legend()
            ax2.set_title("Prophet Forecast - Past 7 & Future 7 Days")
            ax2.set_xlabel("Date")
            ax2.set_ylabel("Views")
            st.pyplot(fig2)
        else:
            st.warning("Not enough data to build a reliable forecast.")
else:
    st.info("Please enter a keyword to begin.")
