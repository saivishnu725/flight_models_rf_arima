import calendar
import gzip
import pickle
from datetime import date, datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

BASE = Path(__file__).parent
MODELS = BASE / "models"
DATA = BASE / "flights.csv"

FEATURES = ["origin", "destination", "departure_date", "cabin", "airline",
            "departure_time", "arrival_time", "arrival_date", "duration_min", "stops"]


# the pickled pipelines refer to these functions by name, so they must be defined before loading
def split_airline(df):
    df = df.copy()
    df[["airline1", "airline2"]] = df["airline"].str.split(" and ", n=1, expand=True)
    df["airline2"] = df["airline2"].fillna("NA")
    return df.drop(columns=["airline"])


def split_dates(df):
    df = df.copy()
    departure = pd.to_datetime(df["departure_date"], format="%d-%m-%Y")
    arrival = pd.to_datetime(df["arrival_date"], format="%d-%m-%Y")
    df["departure_day"] = departure.dt.day
    df["departure_month"] = departure.dt.month
    df["departure_year"] = departure.dt.year
    df["arrival_day"] = arrival.dt.day
    df["arrival_month"] = arrival.dt.month
    df["arrival_year"] = arrival.dt.year
    return df.drop(columns=["departure_date", "arrival_date"])


def split_times(df):
    df = df.copy()
    departure = pd.to_datetime(df["departure_time"], format="%I:%M %p")
    arrival = pd.to_datetime(df["arrival_time"], format="%I:%M %p")
    df["departure_hour"] = departure.dt.hour
    df["departure_minute"] = departure.dt.minute
    df["arrival_hour"] = arrival.dt.hour
    df["arrival_minute"] = arrival.dt.minute
    return df.drop(columns=["departure_time", "arrival_time"])


@st.cache_resource
def load_models():
    models = {}
    for name in ["rf_price", "rf_co2"]:
        with gzip.open(MODELS / f"{name}.pkl.gz", "rb") as f:
            models[name] = pickle.load(f)
    with open(MODELS / "march_forecast.pkl", "rb") as f:
        models["march_forecast"] = pickle.load(f)
    return models


@st.cache_data
def load_data():
    return pd.read_csv(DATA)


models = load_models()
rf_price, rf_co2 = models["rf_price"], models["rf_co2"]
march_forecast = models["march_forecast"]
df = load_data()

# lookup tables for building the candidate flights
route_duration = df.groupby(["origin", "destination"])["duration_min"].mean()

TIME_SLOTS = {
    "morning": ("05:00", "11:30"),
    "afternoon": ("12:00", "16:30"),
    "evening": ("17:00", "20:30"),
    "night": ("21:00", "04:30"),
}


def slot_label(slot):
    start, end = (datetime.strptime(t, "%H:%M").strftime("%I:%M %p").lstrip("0") for t in TIME_SLOTS[slot])
    return f"{slot.capitalize()} ({start} to {end})"


# min-max scale price and co2 to 0-1 and combine them with the user's weight
# co2_weight = 0 -> only price matters, 1 -> only co2 matters
def rank_flights(result, co2_weight, top=5):
    result = result.drop_duplicates(["airline", "departure_time", "stops"]).copy()

    def scale(col):
        spread = col.max() - col.min()
        return (col - col.min()) / spread if spread > 0 else col * 0

    result["score"] = ((1 - co2_weight) * scale(result["predicted_price"])
                       + co2_weight * scale(result["predicted_co2_kg"]))
    return result.sort_values(["score", "predicted_price"]).head(top).reset_index(drop=True)


def slot_times(slot):
    start, end = (datetime.strptime(t, "%H:%M") for t in TIME_SLOTS[slot])
    if end < start:
        end += timedelta(days=1)
    times, t = [], start
    while t <= end:
        times.append(t.strftime("%I:%M %p"))
        t += timedelta(minutes=30)
    return times


def route_airlines(origin, destination, cabin):
    mask = (df["origin"] == origin) & (df["destination"] == destination) & (df["cabin"] == cabin)
    return sorted(df.loc[mask, "airline"].unique())


def route_stops(origin, destination, departure_date, cabin, airline):
    mask = ((df["origin"] == origin) & (df["destination"] == destination)
            & (df["cabin"] == cabin) & (df["airline"] == airline))
    stops = df.loc[mask & (df["departure_date"] == departure_date), "stops"].unique()
    if len(stops) == 0:
        stops = df.loc[mask, "stops"].unique()
    return sorted(stops)


def predict_flights(origin, destination, departure_date, cabin, slot):
    duration = route_duration.loc[(origin, destination)]
    rows = []
    for airline in route_airlines(origin, destination, cabin):
        for stops in route_stops(origin, destination, departure_date, cabin, airline):
            for dep_time in slot_times(slot):
                departure = datetime.strptime(f"{departure_date} {dep_time}", "%d-%m-%Y %I:%M %p")
                arrival = departure + timedelta(minutes=float(duration))
                rows.append({
                    "origin": origin,
                    "destination": destination,
                    "departure_date": departure_date,
                    "cabin": cabin,
                    "airline": airline,
                    "departure_time": dep_time,
                    "arrival_time": arrival.strftime("%I:%M %p"),
                    "arrival_date": arrival.strftime("%d-%m-%Y"),
                    "duration_min": duration,
                    "stops": stops,
                })

    candidates = pd.DataFrame(rows, columns=FEATURES)
    if candidates.empty:
        return candidates
    candidates["predicted_price"] = rf_price.predict(candidates[FEATURES]).round(0)
    candidates["predicted_co2_kg"] = rf_co2.predict(candidates[FEATURES]).round(1)
    return candidates.sort_values(["predicted_price", "predicted_co2_kg"]).reset_index(drop=True)


def plot_calendar(origin, destination, year=2027, month=3):
    fc = march_forecast[(march_forecast["origin"] == origin) &
                        (march_forecast["destination"] == destination)].set_index("date")["predicted_price"]
    weeks = calendar.monthcalendar(year, month)
    cmap = plt.get_cmap("RdPu")
    low, high = fc.min(), fc.max()
    if (high - low) / fc.mean() < 0.02:
        low = high

    fig, ax = plt.subplots(figsize=(12, 1.4 * len(weeks) + 1))
    ax.set_xlim(0, 7)
    ax.set_ylim(len(weeks), -0.5)
    ax.axis("off")
    for col, name in enumerate(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]):
        ax.text(col + 0.5, -0.25, name, ha="center", fontweight="bold")
    for row, week in enumerate(weeks):
        for col, day in enumerate(week):
            if day == 0:
                continue
            price = fc[pd.Timestamp(year, month, day)]
            shade = 0.15 + 0.5 * (price - low) / (high - low) if high > low else 0.3
            ax.add_patch(plt.Rectangle((col + 0.03, row + 0.03), 0.94, 0.94, color=cmap(shade)))
            ax.text(col + 0.1, row + 0.25, str(day), fontsize=10)
            ax.text(col + 0.5, row + 0.65, f"₹{price:,.0f}", ha="center", fontsize=11, fontweight="bold")
    ax.set_title(f"Forecast mean price, {calendar.month_name[month]} {year} - {origin} → {destination}", pad=20)
    return fig


# ---------------------------------------------------------------- app
st.set_page_config(page_title="Sustainable Flight Price Projection", layout="wide")
st.title("Sustainable Flight Price Projection")

tab_predict, tab_data, tab_results = st.tabs(["Predict", "Dataset", "Model Results"])

with tab_predict:
    airports = sorted(df["origin"].unique())
    c1, c2, c3 = st.columns(3)
    origin = c1.selectbox("Origin", airports, index=airports.index("DEL"))
    destination = c2.selectbox("Destination", [a for a in airports if a != origin])
    cabin = c3.selectbox("Cabin", sorted(df["cabin"].unique()))
    c4, c5 = st.columns(2)
    travel_date = c4.date_input("Departure date", value=date(2026, 12, 15),
                                min_value=date(2026, 11, 1), max_value=date(2027, 3, 31),
                                format="DD-MM-YYYY")
    slot = c5.selectbox("Preferred time", list(TIME_SLOTS), format_func=slot_label)

    DEFAULT_CO2_WEIGHT = 65
    if "co2_weight" not in st.session_state:
        st.session_state["co2_weight"] = DEFAULT_CO2_WEIGHT

    def set_weight(value):
        st.session_state["co2_weight"] = value

    st.markdown("**Priority**")
    left, mid, right = st.columns([1, 6, 1])
    left.markdown("<div style='text-align:right'>Cheaper price</div>", unsafe_allow_html=True)
    co2_weight = mid.slider("Priority", 0, 100, key="co2_weight", label_visibility="collapsed",
                            help="Left prioritises cheaper flights, right prioritises lower CO2 emissions") / 100
    right.markdown("Lower CO2")
    st.caption(f"Weights - price: {1 - co2_weight:.0%}, CO2: {co2_weight:.0%}")
    b1, b2, _ = st.columns([1, 1, 4])
    b1.button("Default weights", on_click=set_weight, args=(DEFAULT_CO2_WEIGHT,), use_container_width=True)
    b2.button("Equal weights", on_click=set_weight, args=(50,), use_container_width=True)

    if st.button("Predict", type="primary"):
        st.session_state["result"] = predict_flights(origin, destination,
                                                     travel_date.strftime("%d-%m-%Y"), cabin, slot)

    # kept in session state so moving the slider re-ranks without predicting again
    result = st.session_state.get("result")
    if result is not None:
        if result.empty:
            st.warning("No airlines found for this route and cabin.")
        else:
            top5 = rank_flights(result, co2_weight)
            r = result.iloc[0]
            st.subheader(f"Top 5 flights - {r['origin']} → {r['destination']}, "
                         f"{r['departure_date']}, {r['cabin'].replace('_', ' ')}")
            m1, m2 = st.columns(2)
            m1.metric("Best match price", f"₹{top5.loc[0, 'predicted_price']:,.0f}")
            m2.metric("Best match CO2", f"{top5.loc[0, 'predicted_co2_kg']:.1f} kg")
            st.dataframe(top5[["airline", "departure_time", "arrival_time", "arrival_date",
                               "stops", "predicted_price", "predicted_co2_kg"]]
                         .rename(columns={"predicted_price": "price (INR)", "predicted_co2_kg": "CO2 (kg)"}),
                         use_container_width=True, hide_index=True)

    st.subheader(f"March 2027 price forecast (ARIMA) - {origin} → {destination}")
    st.pyplot(plot_calendar(origin, destination))

with tab_data:
    st.subheader("flights.csv")
    st.caption(f"{df.shape[0]:,} rows × {df.shape[1]} columns")
    st.dataframe(df, use_container_width=True)

with tab_results:
    st.subheader("Random Forest (test set, 20%)")
    st.table(pd.DataFrame({
        "MAE": [2158.271, 6.490],
        "RMSE": [4107.293, 16.642],
        "MAPE (%)": [9.631, 2.790],
        "R2": [0.952, 0.986],
        "Train R2": [0.961, 0.989],
    }, index=["Price (INR)", "CO2 (kg)"]))

    st.subheader("ARIMA(1,1,1) - February 2027 forecast error (one day ahead)")
    st.table(pd.DataFrame({
        "MAE (INR)": [2129.46],
        "RMSE (INR)": [2992.08],
        "MAPE (%)": [11.96],
    }, index=["ARIMA"]))

# footer
st.divider()
st.markdown(
    "<div style='text-align:center; color:gray; font-size:0.9em'>"
    "Made by Kavya Anil (251090700172) &amp; Sai Vishnu M (251090700133) · III MCA A2"
    "</div>",
    unsafe_allow_html=True)
