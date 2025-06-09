import streamlit as st
import pandas as pd
import glob
import os
import numpy as np  # Make sure this is at the top of your file
import datetime

st.set_page_config(layout="wide")

# Directory containing the CSV files
DATA_DIR = "./trading/full_analysis"

st.title("Portfolio Value Over Time")

# Find all CSV files in the data directory
csv_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))

if not csv_files:
    st.error(f"No CSV files found in {DATA_DIR}")
else:
    all_data = {}
    min_date = None
    max_date = None

    for file in csv_files:
        try:
            df = pd.read_csv(file)
            df["Date"] = pd.to_datetime(df["Date"])

            # Update min and max dates
            if min_date is None or df["Date"].min() < min_date:
                min_date = df["Date"].min()
            if max_date is None or df["Date"].max() > max_date:
                max_date = df["Date"].max()

            all_data[os.path.basename(file)] = df[["Date", "Portfolio Value"]]
        except Exception as e:
            st.warning(f"Could not read file {os.path.basename(file)}: {e}")

    if all_data:

        # Initialize session state for the current date if it doesn't exist
        if "current_date" not in st.session_state:
            st.session_state.current_date = max_date.date()

        # Calculate the total number of days in the date range
        total_days = (max_date.date() - min_date.date()).days

        # Calculate the step size (0.4% of total days)
        # Ensure step_size is at least 1 day
        step_size = max(1, int(total_days * 0.004))

        # Checkbox before buttons or reruns
        use_sliding_window = st.checkbox("Sliding Window (Last 9 Months)", value=True)

        # Add buttons for stepping and resetting
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("Step Backward"):
                new_date = st.session_state.current_date - datetime.timedelta(
                    days=step_size
                )
                st.session_state.current_date = max(min_date.date(), new_date)
                st.rerun()

        with col2:
            if st.button("Reset"):
                st.session_state.current_date = min_date.date()
                st.rerun()

        with col3:
            if st.button("Step Forward"):
                new_date = st.session_state.current_date + datetime.timedelta(
                    days=step_size
                )
                st.session_state.current_date = min(max_date.date(), new_date)
                st.rerun()

        # Date slider for end date
        end_date = st.slider(
            "Select End Date",
            min_value=min_date.date(),
            max_value=max_date.date(),
            value=st.session_state.current_date,
            format="YYYY-MM-DD",
        )
        # Prepare data for charting
        chart_data = pd.DataFrame()

        for name, df in all_data.items():
            if use_sliding_window:
                start_date = pd.to_datetime(end_date) - pd.DateOffset(months=9)
            else:
                start_date = min_date

            filtered_df = df[
                (df["Date"] >= start_date) & (df["Date"] <= pd.to_datetime(end_date))
            ].copy()

            # Replace zeroes with NaN so they are not plotted
            filtered_df["Portfolio Value"] = filtered_df["Portfolio Value"].replace(
                0, np.nan
            )

            # Rename column for legend
            filtered_df = filtered_df.rename(columns={"Portfolio Value": name})

            if chart_data.empty:
                chart_data = filtered_df
            else:
                # Merge on Date
                chart_data = pd.merge(chart_data, filtered_df, on="Date", how="outer")

        # Set Date as index for Streamlit chart
        chart_data = chart_data.set_index("Date")
        chart_data = chart_data.astype("float")

        # Fill gaps by interpolating missing values (linearly by default)
        chart_data = chart_data.interpolate()

        # Display the chart
        st.line_chart(chart_data, height=800, use_container_width=True)

    else:
        st.error("No valid data could be loaded from the CSV files.")
