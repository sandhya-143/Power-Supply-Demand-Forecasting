import streamlit as st
import pandas as pd
import numpy as np
import joblib
import holidays
import matplotlib.pyplot as plt


# ==========================================================
# 1. PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="PJM Energy Consumption Forecast",
    page_icon="⚡",
    layout="wide"
)


# ==========================================================
# 2. APPLICATION TITLE
# ==========================================================

st.title("⚡ PJM Hourly Energy Consumption Forecast")

st.caption(
    "Machine Learning-Based Hourly Electricity Demand Forecasting "
    "using XGBoost"
)


# ==========================================================
# 3. LOAD MODEL
# ==========================================================

@st.cache_resource
def load_model():

    return joblib.load(
        "xgboost_power_forecasting.pkl"
    )


try:

    model = load_model()

except FileNotFoundError:

    st.error(
        "The model file 'xgboost_power_forecasting.pkl' "
        "was not found."
    )

    st.stop()


# ==========================================================
# 4. LOAD DATA
# ==========================================================

@st.cache_data
def load_data():

    data = pd.read_excel(
        "PJMW_MW_Hourly.xlsx"
    )

    # Convert first column to datetime
    data.iloc[:, 0] = pd.to_datetime(
        data.iloc[:, 0]
    )

    # Set datetime as index
    data.set_index(
        data.columns[0],
        inplace=True
    )

    # Sort by datetime
    data = data.sort_index()

    # Keep target column
    data = data[["PJMW_MW"]].copy()

    return data


try:

    raw_df = load_data()

except FileNotFoundError:

    st.error(
        "The dataset 'PJMW_MW_Hourly.xlsx' "
        "was not found."
    )

    st.stop()


# ==========================================================
# 5. FEATURE ENGINEERING
# ==========================================================

df = raw_df.copy()


# ----------------------------------------------------------
# Calendar features
# ----------------------------------------------------------

df["Hour"] = df.index.hour

df["DayOfWeek"] = df.index.dayofweek

df["Month"] = df.index.month

df["Year"] = df.index.year


# ----------------------------------------------------------
# Holiday feature
# ----------------------------------------------------------

us_holidays = holidays.US()

df["IsHoliday"] = (
    df.index.to_series()
    .apply(
        lambda x:
        1 if x.date() in us_holidays
        else 0
    )
)


# ----------------------------------------------------------
# Lag features
# ----------------------------------------------------------

df["Lag_1"] = (
    df["PJMW_MW"].shift(1)
)

df["Lag_2"] = (
    df["PJMW_MW"].shift(2)
)

df["Lag_3"] = (
    df["PJMW_MW"].shift(3)
)

df["Lag_24"] = (
    df["PJMW_MW"].shift(24)
)

df["Lag_48"] = (
    df["PJMW_MW"].shift(48)
)

df["Lag_168"] = (
    df["PJMW_MW"].shift(168)
)


# ----------------------------------------------------------
# Rolling features
# ----------------------------------------------------------

df["Rolling_Mean_24"] = (
    df["PJMW_MW"]
    .shift(1)
    .rolling(24)
    .mean()
)

df["Rolling_Mean_168"] = (
    df["PJMW_MW"]
    .shift(1)
    .rolling(168)
    .mean()
)


# ==========================================================
# 6. MODEL FEATURES
# ==========================================================

features = [
    "Lag_1",
    "Lag_2",
    "Lag_3",
    "Lag_24",
    "Lag_48",
    "Lag_168",
    "Rolling_Mean_24",
    "Rolling_Mean_168",
    "Hour",
    "DayOfWeek",
    "Month",
    "Year",
    "IsHoliday"
]


# ==========================================================
# 7. SIDEBAR NAVIGATION
# ==========================================================

st.sidebar.title("⚡ PJM Forecast")

st.sidebar.markdown("### Navigation")

page = st.sidebar.radio(
    "Select a section",
    [
        "Overview",
        "Data Analysis",
        "Trend Analysis",
        "Model Evaluation",
        "Forecast"
    ]
)


st.sidebar.markdown("---")

st.sidebar.markdown(
    """
    **Model:** XGBoost  
    **Target:** PJMW_MW  
    **Forecast:** Up to 30 days  
    **Frequency:** Hourly
    """
)


# ==========================================================
# 8. OVERVIEW
# ==========================================================

if page == "Overview":

    st.header("Project Overview")

    st.write(
        """
        This project focuses on forecasting hourly electricity
        consumption using historical PJM energy demand data.

        An XGBoost regression model is used to learn patterns
        in historical energy consumption. The model uses lag
        features, rolling averages, calendar features and a
        holiday indicator to predict future electricity demand.
        """
    )


    # ------------------------------------------------------
    # Dataset information
    # ------------------------------------------------------

    st.subheader("Dataset Information")

    total_records = len(raw_df)

    start_date = raw_df.index.min()

    end_date = raw_df.index.max()

    average_demand = raw_df[
        "PJMW_MW"
    ].mean()

    maximum_demand = raw_df[
        "PJMW_MW"
    ].max()


    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Total Records",
            f"{total_records:,}"
        )

    with col2:

        st.metric(
            "Average Demand",
            f"{average_demand:,.0f} MW"
        )

    with col3:

        st.metric(
            "Maximum Demand",
            f"{maximum_demand:,.0f} MW"
        )

    with col4:

        st.metric(
            "Forecast Horizon",
            "30 Days"
        )


    st.markdown("---")


    # ------------------------------------------------------
    # Dataset period
    # ------------------------------------------------------

    st.subheader("Dataset Period")

    col1, col2 = st.columns(2)

    with col1:

        st.write("**Start:**")

        st.write(
            start_date.strftime(
                "%d %B %Y, %H:%M"
            )
        )

    with col2:

        st.write("**End:**")

        st.write(
            end_date.strftime(
                "%d %B %Y, %H:%M"
            )
        )


    # ------------------------------------------------------
    # Project workflow
    # ------------------------------------------------------

    st.subheader("Project Workflow")

    st.write(
        """
        **Historical Data**
        → **Data Preprocessing**
        → **Feature Engineering**
        → **Model Training**
        → **Model Evaluation**
        → **XGBoost**
        → **Energy Demand Forecast**
        """
    )


    # ------------------------------------------------------
    # Features used
    # ------------------------------------------------------

    st.subheader("Features Used by the Model")

    feature_df = pd.DataFrame(
        {
            "Feature": features,
            "Description": [
                "Previous hour consumption",
                "Consumption two hours earlier",
                "Consumption three hours earlier",
                "Consumption 24 hours earlier",
                "Consumption 48 hours earlier",
                "Consumption 168 hours earlier",
                "24-hour rolling mean",
                "168-hour rolling mean",
                "Hour of day",
                "Day of week",
                "Month",
                "Year",
                "US holiday indicator"
            ]
        }
    )

    st.dataframe(
        feature_df,
        use_container_width=True,
        hide_index=True
    )


# ==========================================================
# 9. DATA ANALYSIS
# ==========================================================

elif page == "Data Analysis":

    st.header("📊 Data Analysis")

    st.write(
        "Explore the structure and statistical characteristics "
        "of the PJM energy consumption dataset."
    )


    # ------------------------------------------------------
    # Dataset preview
    # ------------------------------------------------------

    st.subheader("Dataset Preview")

    st.dataframe(
        raw_df.head(10),
        use_container_width=True
    )


    # ------------------------------------------------------
    # Dataset shape
    # ------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Rows",
            f"{raw_df.shape[0]:,}"
        )

    with col2:

        st.metric(
            "Columns",
            raw_df.shape[1]
        )


    # ------------------------------------------------------
    # Missing values
    # ------------------------------------------------------

    st.subheader("Missing Values")

    missing_values = raw_df.isnull().sum()

    missing_df = (
        missing_values
        .reset_index()
    )

    missing_df.columns = [
        "Column",
        "Missing Values"
    ]


    st.dataframe(
        missing_df,
        use_container_width=True,
        hide_index=True
    )


    if missing_values.sum() == 0:

        st.success(
            "No missing values are present in the target data."
        )

    else:

        st.warning(
            "Missing values are present in the dataset."
        )


    # ------------------------------------------------------
    # Summary statistics
    # ------------------------------------------------------

    st.subheader("Summary Statistics")

    st.dataframe(
        raw_df["PJMW_MW"]
        .describe()
        .to_frame()
        .T,
        use_container_width=True
    )


    # ------------------------------------------------------
    # Distribution
    # ------------------------------------------------------

    st.subheader(
        "Energy Consumption Distribution"
    )

    fig, ax = plt.subplots(
        figsize=(12, 5)
    )

    ax.hist(
        raw_df["PJMW_MW"].dropna(),
        bins=50
    )

    ax.set_xlabel(
        "Power Consumption (MW)"
    )

    ax.set_ylabel(
        "Frequency"
    )

    ax.set_title(
        "Distribution of PJM Energy Consumption"
    )

    ax.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    st.pyplot(fig)

    st.caption(
        "The histogram shows how frequently different "
        "energy consumption levels occur."
    )


# ==========================================================
# 10. TREND ANALYSIS
# ==========================================================

elif page == "Trend Analysis":

    st.header("📈 Trend Analysis")

    st.write(
        "Analyze long-term, daily and weekly patterns in "
        "electricity consumption."
    )


    # ------------------------------------------------------
    # Trend period selection
    # ------------------------------------------------------

    st.subheader(
        "Historical Consumption Trend"
    )

    trend_period = st.selectbox(
        "Select time period",
        [
            "Last 7 Days",
            "Last 30 Days",
            "Last 1 Year",
            "Entire Dataset"
        ]
    )


    if trend_period == "Last 7 Days":

        trend_data = raw_df.tail(
            24 * 7
        )

    elif trend_period == "Last 30 Days":

        trend_data = raw_df.tail(
            24 * 30
        )

    elif trend_period == "Last 1 Year":

        trend_data = raw_df[
            raw_df.index >=
            raw_df.index.max()
            - pd.Timedelta(days=365)
        ]

    else:

        trend_data = raw_df


    # ------------------------------------------------------
    # Historical trend chart
    # ------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(14, 5)
    )

    ax.plot(
        trend_data.index,
        trend_data["PJMW_MW"]
    )

    ax.set_xlabel(
        "Date"
    )

    ax.set_ylabel(
        "Power Consumption (MW)"
    )

    ax.set_title(
        "Historical PJM Energy Consumption"
    )

    ax.grid(
        True,
        alpha=0.3
    )

    plt.xticks(
        rotation=45
    )

    plt.tight_layout()

    st.pyplot(fig)


    st.caption(
        f"Showing {trend_period.lower()} of energy consumption."
    )


    # ------------------------------------------------------
    # Hourly pattern
    # ------------------------------------------------------

    st.subheader(
        "Average Consumption by Hour"
    )

    hourly_pattern = (
        raw_df
        .groupby(raw_df.index.hour)
        ["PJMW_MW"]
        .mean()
    )

    hourly_pattern.index.name = "Hour"


    fig, ax = plt.subplots(
        figsize=(12, 5)
    )

    ax.plot(
        hourly_pattern.index,
        hourly_pattern.values,
        marker="o"
    )

    ax.set_xlabel(
        "Hour of Day"
    )

    ax.set_ylabel(
        "Average Consumption (MW)"
    )

    ax.set_title(
        "Average Energy Consumption by Hour"
    )

    ax.set_xticks(
        range(24)
    )

    ax.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    st.pyplot(fig)


    st.info(
        "This trend helps identify peak and low-demand hours "
        "during a typical day."
    )


    # ------------------------------------------------------
    # Weekly pattern
    # ------------------------------------------------------

    st.subheader(
        "Average Consumption by Day of Week"
    )

    day_names = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday"
    ]


    weekly_pattern = (
        raw_df
        .groupby(raw_df.index.dayofweek)
        ["PJMW_MW"]
        .mean()
    )


    weekly_pattern.index = [
        day_names[i]
        for i in weekly_pattern.index
    ]


    fig, ax = plt.subplots(
        figsize=(12, 5)
    )

    ax.bar(
        weekly_pattern.index,
        weekly_pattern.values
    )

    ax.set_xlabel(
        "Day of Week"
    )

    ax.set_ylabel(
        "Average Consumption (MW)"
    )

    ax.set_title(
        "Average Energy Consumption by Day of Week"
    )

    ax.grid(
        True,
        axis="y",
        alpha=0.3
    )

    plt.xticks(
        rotation=30
    )

    plt.tight_layout()

    st.pyplot(fig)


    st.info(
        "The weekly pattern helps identify differences "
        "between weekday and weekend electricity demand."
    )


    # ------------------------------------------------------
    # Rolling mean
    # ------------------------------------------------------

    st.subheader(
        "Rolling Mean Analysis"
    )

    rolling_data = pd.DataFrame(
        {
            "Actual Consumption":
                raw_df["PJMW_MW"].tail(720),

            "24-Hour Rolling Mean":
                raw_df["PJMW_MW"]
                .shift(1)
                .rolling(24)
                .mean()
                .tail(720),

            "168-Hour Rolling Mean":
                raw_df["PJMW_MW"]
                .shift(1)
                .rolling(168)
                .mean()
                .tail(720)
        }
    )


    st.line_chart(
        rolling_data,
        use_container_width=True
    )


    st.caption(
        "Rolling means smooth short-term fluctuations and "
        "help reveal the underlying consumption trend."
    )


    # ------------------------------------------------------
    # Lag correlation
    # ------------------------------------------------------

    st.subheader(
        "Lag Correlation Analysis"
    )


    lag_data = pd.DataFrame(
        {
            "Lag": [
                "Lag_1",
                "Lag_2",
                "Lag_24",
                "Lag_48",
                "Lag_168"
            ],

            "Correlation": [
                0.974679,
                0.910154,
                0.877163,
                0.750401,
                0.759284
            ]
        }
    )


    st.dataframe(
        lag_data,
        use_container_width=True,
        hide_index=True
    )


    st.info(
        "Lag-1 has the strongest correlation with the target, "
        "showing that the previous hour's consumption is highly "
        "useful for predicting the next hour."
    )


# ==========================================================
# 11. MODEL EVALUATION
# ==========================================================

elif page == "Model Evaluation":

    st.header("🤖 Model Evaluation")

    st.write(
        "Comparison of the forecasting models evaluated "
        "during model development."
    )


    # ------------------------------------------------------
    # Model results
    # ------------------------------------------------------

    results = pd.DataFrame(
        {
            "Model": [
                "Baseline",
                "ARIMA",
                "SARIMA",
                "Random Forest",
                "XGBoost"
            ],

            "MAE": [
                126.8155,
                838.8658,
                891.0108,
                53.8745,
                51.0083
            ],

            "RMSE": [
                177.7681,
                977.8665,
                1057.8766,
                70.0434,
                64.0635
            ]
        }
    )


    st.subheader(
        "Model Performance Comparison"
    )


    st.dataframe(
        results.style.format(
            {
                "MAE": "{:.2f}",
                "RMSE": "{:.2f}"
            }
        ),
        use_container_width=True,
        hide_index=True
    )


    # ------------------------------------------------------
    # Best model
    # ------

