import streamlit as st
import pandas as pd

st.title("Akshayapatra Food Feedback Dashboard")

df = pd.read_csv("data/predictions.csv")

# Show full data
st.subheader("Predicted Feedback Data")
st.dataframe(df)

# Overall sentiment distribution
st.subheader("Overall Sentiment Count")
st.bar_chart(df["predicted_sentiment"].value_counts())

# Negative item analysis
st.subheader("Top Negative Food Items")

negative_items = (
    df[df["predicted_sentiment"] == "negative"]
    .groupby("item_name")
    .size()
    .reset_index(name="negative_count")
    .sort_values(by="negative_count", ascending=False)
)

st.dataframe(negative_items)

# Positive item analysis
st.subheader("Top Positive Food Items")

positive_items = (
    df[df["predicted_sentiment"] == "positive"]
    .groupby("item_name")
    .size()
    .reset_index(name="positive_count")
    .sort_values(by="positive_count", ascending=False)
)

st.dataframe(positive_items)

# Search specific item
st.subheader("Search Food Item Feedback")

item = st.selectbox("Select Item", df["item_name"].unique())

filtered = df[df["item_name"] == item]

st.write(filtered[["feedback_text", "predicted_sentiment"]])