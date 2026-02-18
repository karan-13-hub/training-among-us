import pandas as pd
import json
import streamlit as st

RESULT_PATH: str = 'results/2025-01-25_phi_llama_100_games_strategy_skill.json'

with open(RESULT_PATH, "r") as f:
    data = [json.loads(line) for line in f]

# Convert JSON to a DataFrame
df = pd.json_normalize(data)

# Streamlit App

st.set_page_config(layout="wide")

st.title("Among Us Visualization (Many Games, Strategy Score)")

# Filter Sidebar
default_columns = ["game_index", "step", "player_name", "player_identity", "action", "strategy_score"]

# Sidebar for first column selection
st.sidebar.header("Filter by First Column")

# Select the first column for filtering
first_column = st.sidebar.selectbox("Choose a column to filter by", df.columns)

# Get unique values from the selected column
unique_vals = df[first_column].dropna().unique()

# Select a value from the first column
selected_first_value = st.sidebar.selectbox(f"Select a {first_column} value", options=[None] + list(unique_vals))

# Filter the DataFrame based on the first column value selection
if selected_first_value:
    filtered_df = df[df[first_column] == selected_first_value]
else:
    filtered_df = df[df[first_column] == 'Game 1']

# Sidebar for column selection
st.sidebar.header("Select Columns to Display")
# Allow users to choose which columns to display, pre-select the default columns
columns_to_show = st.sidebar.multiselect("Select columns to display", df.columns.tolist(), default=default_columns)

# Sidebar for further filtering based on the filtered DataFrame
st.sidebar.header("Further Filter by Other Columns")
other_columns = df.columns[df.columns != first_column]  # Exclude the first column from further filters

# Allow filtering on the remaining columns based on the current filtered data
for column in other_columns:
    if filtered_df[column].dtype == "object":  # For categorical columns
        unique_vals = filtered_df[column].dropna().unique()
        selected_val = st.sidebar.selectbox(f"Filter by {column}", options=[None] + list(unique_vals))
        if selected_val:
            filtered_df = filtered_df[filtered_df[column] == selected_val]
    elif filtered_df[column].dtype in ['int64', 'float64']:  # For numeric columns
        min_val, max_val = filtered_df[column].min(), filtered_df[column].max()
        selected_range = st.sidebar.slider(f"Filter by {column} range", min_val, max_val, (min_val, max_val))
        filtered_df = filtered_df[filtered_df[column].between(selected_range[0], selected_range[1])]

# Display the filtered DataFrame with selected columns
st.write(f"Filtered Data ({filtered_df.shape[0]} rows)")
st.dataframe(filtered_df[columns_to_show], use_container_width=True)

# Show summary statistics for the filtered DataFrame
st.write("Summary Table")
st.write(filtered_df[columns_to_show].describe(include="all"))