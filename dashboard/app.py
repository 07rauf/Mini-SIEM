
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# =====================================================
# CONFIGURATION
# =====================================================

BASE_DIR = Path(__file__).resolve().parents[1]
FEATURE_DIR = BASE_DIR / "data" / "features"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

st.set_page_config(
    page_title="Mini-SIEM | SOC Operations Center",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# SOC-style visual theme
st.markdown(
    """
    <style>
    .stApp {
        background: #0B0F17;
        color: #E8EDF5;
    }
    [data-testid="stSidebar"] {
        background: #111824;
        border-right: 1px solid #263244;
    }
    [data-testid="stMetric"] {
        background: linear-gradient(145deg, #151F2D, #101722);
        border: 1px solid #26364A;
        border-radius: 10px;
        padding: 13px 15px;
        min-height: 105px;
    }
    [data-testid="stMetricLabel"] {
        color: #9EADC0;
        font-size: 0.82rem;
    }
    [data-testid="stMetricValue"] {
        color: #F4F7FB;
        font-size: 1.75rem;
        font-weight: 650;
    }
    h1, h2, h3, h4 {
        color: #F3F6FA;
    }
    .soc-subtitle {
        color: #91A0B4;
        font-size: 0.9rem;
        margin-top: -12px;
    }
    .panel-caption {
        color: #92A2B7;
        font-size: 0.78rem;
        margin-top: -7px;
    }
    .status-online {
        color: #43D9A3;
        font-weight: 700;
        text-align: right;
        padding-top: 22px;
    }
    div[data-testid="stHorizontalBlock"] {
        gap: 1rem;
    }
    hr {
        border-color: #263244;
    }
    div[data-testid="stDataFrame"] {
        border: 1px solid #263244;
        border-radius: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

PLOT_TEMPLATE = "plotly_dark"
PLOT_BG = "rgba(0,0,0,0)"


# =====================================================
# DATA LOADING
# =====================================================

@st.cache_data
def load_data():
    web = pd.read_csv(FEATURE_DIR / "web_kmeans_results.csv")
    firewall = pd.read_csv(PROCESSED_DIR / "firewall_clean.csv")
    dbscan = pd.read_csv(FEATURE_DIR / "web_dbscan_results.csv")

    web["timestamp"] = pd.to_datetime(web["timestamp"], utc=True)
    firewall["timestamp"] = pd.to_datetime(firewall["timestamp"], utc=True)

    if "row_index" not in web.columns:
        web = web.reset_index().rename(columns={"index": "row_index"})

    if "row_index" not in dbscan.columns:
        raise ValueError(
            "web_dbscan_results.csv faylında row_index sütunu yoxdur."
        )

    dbscan = dbscan[["row_index", "dbscan_cluster"]].drop_duplicates(
        subset=["row_index"]
    )

    web = web.merge(dbscan, on="row_index", how="left")

    return web, firewall


try:
    web_df, firewall_df = load_data()
except Exception as error:
    st.error(f"Məlumat faylları yüklənmədi: {error}")
    st.stop()


# =====================================================
# HEADER
# =====================================================

head_left, head_right = st.columns([5, 1])

with head_left:
    st.title("🔐 MINI-SIEM | SECURITY OPERATIONS CENTER")
with head_right:
    st.markdown(
        "<div class='status-online'>● SYSTEM ONLINE</div>",
        unsafe_allow_html=True
    )

st.divider()


# =====================================================
# SIDEBAR FILTERS
# =====================================================

st.sidebar.header("Monitorinq filterləri")
st.sidebar.caption("Seçilən filterlər uyğun qrafik və cədvəllərə tətbiq olunur.")

min_date = min(
    web_df["timestamp"].min().date(),
    firewall_df["timestamp"].min().date()
)
max_date = max(
    web_df["timestamp"].max().date(),
    firewall_df["timestamp"].max().date()
)

date_range = st.sidebar.date_input(
    "Tarix aralığı",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date, end_date = min_date, max_date

web_date_filtered = web_df[
    web_df["timestamp"].dt.date.between(start_date, end_date)
].copy()

firewall_filtered = firewall_df[
    firewall_df["timestamp"].dt.date.between(start_date, end_date)
].copy()

methods = sorted(web_date_filtered["method"].dropna().unique())
selected_methods = st.sidebar.multiselect(
    "HTTP Method",
    methods,
    default=methods
)

statuses = sorted(web_date_filtered["status"].dropna().unique())
selected_statuses = st.sidebar.multiselect(
    "HTTP Status",
    statuses,
    default=statuses
)

agents = sorted(web_date_filtered["user_agent_type"].dropna().unique())
selected_agents = st.sidebar.multiselect(
    "User-Agent",
    agents,
    default=agents
)

ip_search = st.sidebar.text_input(
    "IP ünvanı üzrə axtarış",
    placeholder="Məsələn: 87.111..."
)

firewall_actions = sorted(firewall_filtered["action"].dropna().unique())
selected_actions = st.sidebar.multiselect(
    "Firewall Action",
    firewall_actions,
    default=firewall_actions
)

web_filtered = web_date_filtered[
    web_date_filtered["method"].isin(selected_methods)
    & web_date_filtered["status"].isin(selected_statuses)
    & web_date_filtered["user_agent_type"].isin(selected_agents)
].copy()

if ip_search.strip():
    web_filtered = web_filtered[
        web_filtered["src_ip"].astype(str).str.contains(
            ip_search.strip(), case=False, na=False
        )
    ].copy()

firewall_filtered = firewall_filtered[
    firewall_filtered["action"].isin(selected_actions)
].copy()

# =====================================================
# COMMON METRICS
# =====================================================

total_requests = len(web_filtered)
suspicious_df = web_filtered[web_filtered["is_suspicious"]].copy()
suspicious_count = len(suspicious_df)
normal_count = total_requests - suspicious_count

suspicious_pct = (
    suspicious_count / total_requests * 100 if total_requests else 0
)

firewall_total = len(firewall_filtered)
firewall_deny = int((firewall_filtered["action"] == "DENY").sum())
firewall_allow = int((firewall_filtered["action"] == "ALLOW").sum())

dbscan_filtered = web_filtered.dropna(subset=["dbscan_cluster"]).copy()
if not dbscan_filtered.empty:
    dbscan_filtered["dbscan_cluster"] = (
        dbscan_filtered["dbscan_cluster"].astype(int)
    )
noise_count = int((dbscan_filtered["dbscan_cluster"] == -1).sum())
noise_pct = (
    noise_count / len(dbscan_filtered) * 100 if len(dbscan_filtered) else 0
)


# =====================================================
# KPI STRIP
# =====================================================

st.subheader("SECURITY OVERVIEW")

k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("WEB REQUESTS", f"{total_requests:,}")
k2.metric("SUSPICIOUS", f"{suspicious_count:,}")
k3.metric("NORMAL", f"{normal_count:,}")
k4.metric("SQLi INDICATOR RATE", f"{suspicious_pct:.2f}%")
k5.metric("FIREWALL DENY", f"{firewall_deny:,}")
k6.metric("DBSCAN NOISE", f"{noise_count:,}")

st.divider()


# =====================================================
# WORLD MAP: FIREWALL DESTINATION COUNTRIES
# =====================================================

st.subheader("GLOBAL NETWORK ACTIVITY")
st.markdown(
    "<div class='panel-caption'>Firewall qeydlərindəki dst_country sahəsinə əsasən ölkələr üzrə hadisə paylanması</div>",
    unsafe_allow_html=True
)

# Country-code centroids used to place activity bubbles on the world map.
# The dataset contains ISO-2 country codes in dst_country.
country_geo = {
    "GB": {"name": "United Kingdom", "lat": 55.3781, "lon": -3.4360},
    "CN": {"name": "China", "lat": 35.8617, "lon": 104.1954},
    "TR": {"name": "Türkiye", "lat": 38.9637, "lon": 35.2433},
    "FR": {"name": "France", "lat": 46.2276, "lon": 2.2137},
    "US": {"name": "United States", "lat": 37.0902, "lon": -95.7129},
    "RU": {"name": "Russia", "lat": 61.5240, "lon": 105.3188},
    "JP": {"name": "Japan", "lat": 36.2048, "lon": 138.2529},
    "NL": {"name": "Netherlands", "lat": 52.1326, "lon": 5.2913},
    "DE": {"name": "Germany", "lat": 51.1657, "lon": 10.4515},
    "AZ": {"name": "Azerbaijan", "lat": 40.1431, "lon": 47.5769},
}

if not firewall_filtered.empty and "dst_country" in firewall_filtered.columns:
    country_activity = (
        firewall_filtered["dst_country"]
        .fillna("Unknown")
        .value_counts()
        .rename_axis("country_code")
        .reset_index(name="events")
    )

    country_activity["country_code"] = (
        country_activity["country_code"].astype(str).str.upper()
    )

    country_activity["country_name"] = country_activity[
        "country_code"
    ].map(
        lambda code: country_geo.get(code, {}).get("name", code)
    )
    country_activity["latitude"] = country_activity[
        "country_code"
    ].map(
        lambda code: country_geo.get(code, {}).get("lat")
    )
    country_activity["longitude"] = country_activity[
        "country_code"
    ].map(
        lambda code: country_geo.get(code, {}).get("lon")
    )

    mapped_countries = country_activity.dropna(
        subset=["latitude", "longitude"]
    ).copy()

    map_col, map_side = st.columns([2.15, 1])

    with map_col:
        if not mapped_countries.empty:
            fig = px.scatter_geo(
                mapped_countries,
                lat="latitude",
                lon="longitude",
                size="events",
                color="events",
                hover_name="country_name",
                hover_data={
                    "country_code": True,
                    "events": ":,",
                    "latitude": False,
                    "longitude": False
                },
                size_max=42,
                projection="natural earth",
                color_continuous_scale=[
                    "#163A63", "#087EBD", "#24C6DC", "#A5F3FC"
                ],
                template=PLOT_TEMPLATE
            )
            fig.update_geos(
                showland=True,
                landcolor="#182332",
                showocean=True,
                oceancolor="#0B1420",
                showcountries=True,
                countrycolor="#34465C",
                showcoastlines=True,
                coastlinecolor="#52647A",
                showframe=False,
                bgcolor=PLOT_BG
            )
            fig.update_layout(
                height=470,
                margin=dict(l=0, r=0, t=8, b=0),
                paper_bgcolor=PLOT_BG,
                geo=dict(bgcolor=PLOT_BG),
                coloraxis_colorbar=dict(
                    title="Hadisələr",
                    thickness=10,
                    len=0.65
                )
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Xəritədə göstərmək üçün tanınan ölkə kodu yoxdur.")

    with map_side:
        st.markdown("#### Ölkələr üzrə hadisələr")
        country_top = country_activity[
            ["country_name", "events"]
        ].head(10).copy()
        country_top.columns = ["Ölkə", "Hadisə sayı"]
        st.dataframe(
            country_top,
            use_container_width=True,
            hide_index=True,
            height=340
        )

        if not mapped_countries.empty:
            st.metric(
                "Xəritədə göstərilən ölkə",
                f"{mapped_countries['country_code'].nunique()}"
            )
            st.metric(
                "Firewall hadisələri",
                f"{firewall_total:,}"
            )
else:
    st.info("Seçilmiş tarix və action filterləri üzrə firewall ölkə məlumatı yoxdur.")

st.caption(
    "Vacib: dst_country təyinat ölkəsini bildirir. Bu xəritə source IP-lərin coğrafi mənşəyini göstərmir."
)

st.divider()


# =====================================================
# FIREWALL + WEB TRAFFIC PANELS
# =====================================================

st.subheader("TRAFFIC & FIREWALL MONITORING")
traffic_col, firewall_col = st.columns([1.55, 1])

with traffic_col:
    st.markdown("#### Web sorğularının zaman üzrə dinamikası")

    if not web_filtered.empty:
        daily = (
            web_filtered.set_index("timestamp")
            .resample("D")
            .size()
            .reset_index(name="requests")
        )
        fig = px.area(
            daily,
            x="timestamp",
            y="requests",
            markers=True,
            labels={"timestamp": "Tarix", "requests": "Sorğu sayı"},
            template=PLOT_TEMPLATE
        )
        fig.update_traces(
            line=dict(color="#35B6E8", width=2),
            fillcolor="rgba(53,182,232,0.15)"
        )
        fig.update_layout(
            height=330,
            margin=dict(l=10, r=10, t=20, b=10),
            paper_bgcolor=PLOT_BG,
            plot_bgcolor=PLOT_BG
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Web trafik məlumatı yoxdur.")

with firewall_col:
    st.markdown("#### Firewall ALLOW / DENY")

    if not firewall_filtered.empty:
        action_data = (
            firewall_filtered["action"]
            .value_counts()
            .rename_axis("Action")
            .reset_index(name="Events")
        )
        fig = px.pie(
            action_data,
            names="Action",
            values="Events",
            hole=0.62,
            color="Action",
            color_discrete_map={
                "ALLOW": "#21C997",
                "DENY": "#F05D6C"
            },
            template=PLOT_TEMPLATE
        )
        fig.update_layout(
            height=330,
            margin=dict(l=10, r=10, t=20, b=10),
            paper_bgcolor=PLOT_BG,
            legend_title_text=""
        )
        st.plotly_chart(fig, use_container_width=True)
        a1, a2 = st.columns(2)
        a1.metric("ALLOW", f"{firewall_allow:,}")
        a2.metric("DENY", f"{firewall_deny:,}")
    else:
        st.info("Firewall məlumatı yoxdur.")

st.divider()


# =====================================================
# ENDPOINTS + HTTP STATUS + USER AGENT
# =====================================================

st.subheader("WEB APPLICATION ACTIVITY")
ep_col, status_col, agent_col = st.columns([1.35, 1, 1])

with ep_col:
    st.markdown("#### Top 10 endpoint")
    if not web_filtered.empty:
        endpoint_data = (
            web_filtered["path"]
            .value_counts()
            .head(10)
            .rename_axis("Endpoint")
            .reset_index(name="Requests")
            .sort_values("Requests", ascending=True)
        )
        fig = px.bar(
            endpoint_data,
            x="Requests",
            y="Endpoint",
            orientation="h",
            text="Requests",
            template=PLOT_TEMPLATE,
            color="Requests",
            color_continuous_scale=["#164E78", "#35B6E8"]
        )
        fig.update_layout(
            height=360,
            margin=dict(l=8, r=8, t=15, b=8),
            paper_bgcolor=PLOT_BG,
            plot_bgcolor=PLOT_BG,
            coloraxis_showscale=False,
            yaxis=dict(tickfont=dict(size=9))
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Endpoint məlumatı yoxdur.")

with status_col:
    st.markdown("#### HTTP status bölgüsü")
    if not web_filtered.empty:
        status_data = (
            web_filtered["status"]
            .value_counts()
            .sort_index()
            .rename_axis("Status")
            .reset_index(name="Events")
        )

        # Status dəyərlərini kateqoriyaya çeviririk
        status_data["Status"] = status_data["Status"].astype(str)

        fig = px.bar(
            status_data,
            x="Status",
            y="Events",
            text="Events",
            template=PLOT_TEMPLATE,
            color="Events",
            color_continuous_scale=["#164E78", "#35B6E8"]
        )
        fig.update_layout(
            height=360,
            margin=dict(l=5, r=5, t=15, b=8),
            paper_bgcolor=PLOT_BG,
            plot_bgcolor=PLOT_BG,
            coloraxis_showscale=False
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("HTTP status məlumatı yoxdur.")

with agent_col:
    st.markdown("#### User-Agent bölgüsü")
    if not web_filtered.empty:
        agent_data = (
            web_filtered["user_agent_type"]
            .value_counts()
            .rename_axis("User-Agent")
            .reset_index(name="Events")
        )
        fig = px.pie(
            agent_data,
            names="User-Agent",
            values="Events",
            hole=0.48,
            template=PLOT_TEMPLATE,
            color_discrete_sequence=[
                "#35B6E8", "#21C997", "#A78BFA", "#F5B942", "#F05D6C"
            ]
        )
        fig.update_layout(
            height=360,
            margin=dict(l=5, r=5, t=15, b=8),
            paper_bgcolor=PLOT_BG,
            legend_title_text=""
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("User-Agent məlumatı yoxdur.")

st.divider()


# =====================================================
# SQLi HEURISTIC PANELS
# =====================================================

st.subheader("SUSPICIOUS ACTIVITY | SQLi HEURISTICS")
sqli_status_col, sqli_indicator_col = st.columns(2)

with sqli_status_col:
    st.markdown("#### Şübhəli sorğuların HTTP status bölgüsü")
    if not suspicious_df.empty:
        suspicious_status = (
            suspicious_df["status"]
            .value_counts()
            .sort_index()
            .rename_axis("HTTP Status")
            .reset_index(name="Count")
        )

        # Status dəyərlərini kateqoriyaya çeviririk
        suspicious_status["HTTP Status"] = (
            suspicious_status["HTTP Status"].astype(str)
        )

        fig = px.bar(
            suspicious_status,
            x="HTTP Status",
            y="Count",
            text="Count",
            template=PLOT_TEMPLATE,
            color="Count",
            color_continuous_scale=["#6D2932", "#F05D6C"]
        )
        fig.update_layout(
            height=310,
            margin=dict(l=8, r=8, t=15, b=8),
            paper_bgcolor=PLOT_BG,
            plot_bgcolor=PLOT_BG,
            coloraxis_showscale=False
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Şübhəli sorğu tapılmadı.")

with sqli_indicator_col:
    st.markdown("#### SQLi heuristic indikatorları")
    if not suspicious_df.empty:
        indicator_data = pd.DataFrame({
            "Indicator": [
                "UNION SELECT",
                "DROP keyword",
                "SQL comment",
                "Boolean condition"
            ],
            "Count": [
                int(suspicious_df["has_union_select"].sum()),
                int(suspicious_df["has_drop_keyword"].sum()),
                int(suspicious_df["has_sql_comment"].sum()),
                int(suspicious_df["has_boolean_condition"].sum())
            ]
        })
        fig = px.bar(
            indicator_data,
            x="Indicator",
            y="Count",
            text="Count",
            template=PLOT_TEMPLATE,
            color="Count",
            color_continuous_scale=["#6D2932", "#F05D6C"]
        )
        fig.update_layout(
            height=310,
            margin=dict(l=8, r=8, t=15, b=8),
            paper_bgcolor=PLOT_BG,
            plot_bgcolor=PLOT_BG,
            coloraxis_showscale=False,
            xaxis=dict(tickangle=-10)
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("İndikator məlumatı yoxdur.")


st.divider()


# =====================================================
# K-MEANS + DBSCAN PANELS
# =====================================================

st.subheader("UNSUPERVISED CLUSTER ANALYTICS")
km_col, db_col = st.columns(2)

with km_col:
    st.markdown("#### K-Means klaster analizi")

    if not web_filtered.empty:
        km_total = len(web_filtered)
        km_count = web_filtered["kmeans_cluster"].nunique()
        km_suspicious = int(web_filtered["is_suspicious"].sum())

        m1, m2, m3 = st.columns(3)
        m1.metric("Sorğular", f"{km_total:,}")
        m2.metric("Klaster sayı", f"{km_count}")
        m3.metric("Şübhəli", f"{km_suspicious:,}")

        km_counts = (
            web_filtered["kmeans_cluster"]
            .value_counts()
            .sort_index()
            .rename_axis("Cluster")
            .reset_index(name="Events")
        )
        fig = px.bar(
            km_counts,
            x="Cluster",
            y="Events",
            text="Events",
            template=PLOT_TEMPLATE,
            color="Events",
            color_continuous_scale=["#174A70", "#35B6E8"]
        )
        fig.update_layout(
            height=310,
            margin=dict(l=8, r=8, t=15, b=8),
            paper_bgcolor=PLOT_BG,
            plot_bgcolor=PLOT_BG,
            coloraxis_showscale=False
        )
        st.plotly_chart(fig, use_container_width=True)

        km_summary = web_filtered.groupby("kmeans_cluster").agg(
            total_requests=("is_suspicious", "size"),
            suspicious_requests=("is_suspicious", "sum")
        ).reset_index()
        km_summary["suspicious_percentage"] = (
            km_summary["suspicious_requests"]
            / km_summary["total_requests"] * 100
        ).round(2)
        st.dataframe(
            km_summary,
            use_container_width=True,
            hide_index=True,
            height=175
        )
    else:
        st.info("K-Means məlumatı yoxdur.")

with db_col:
    st.markdown("#### DBSCAN klaster analizi")

    if not dbscan_filtered.empty:
        db_total = len(dbscan_filtered)
        db_noise = int((dbscan_filtered["dbscan_cluster"] == -1).sum())
        db_noise_pct = db_noise / db_total * 100 if db_total else 0
        db_clusters = dbscan_filtered[
            dbscan_filtered["dbscan_cluster"] != -1
        ]["dbscan_cluster"].nunique()

        m1, m2, m3 = st.columns(3)
        m1.metric("Hadisələr", f"{db_total:,}")
        m2.metric("Noise", f"{db_noise:,}")
        m3.metric("Noise faizi", f"{db_noise_pct:.2f}%")

        db_counts = (
            dbscan_filtered["dbscan_cluster"]
            .value_counts()
            .sort_index()
            .rename_axis("Cluster")
            .reset_index(name="Events")
        )
        db_counts["Cluster Name"] = db_counts["Cluster"].apply(
            lambda x: "Noise (-1)" if x == -1 else f"Cluster {x}"
        )
        fig = px.bar(
            db_counts,
            x="Cluster Name",
            y="Events",
            text="Events",
            template=PLOT_TEMPLATE,
            color="Events",
            color_continuous_scale=["#174A70", "#35B6E8"]
        )
        fig.update_layout(
            height=310,
            margin=dict(l=8, r=8, t=15, b=8),
            paper_bgcolor=PLOT_BG,
            plot_bgcolor=PLOT_BG,
            coloraxis_showscale=False
        )
        st.plotly_chart(fig, use_container_width=True)

        db_summary = dbscan_filtered.groupby("dbscan_cluster").agg(
            total_events=("is_suspicious", "size"),
            suspicious_events=("is_suspicious", "sum")
        ).reset_index()
        db_summary["suspicious_percentage"] = (
            db_summary["suspicious_events"]
            / db_summary["total_events"] * 100
        ).round(2)
        st.dataframe(
            db_summary,
            use_container_width=True,
            hide_index=True,
            height=175
        )
    else:
        st.info("DBSCAN məlumatı yoxdur.")

st.divider()


# =====================================================
# SUSPICIOUS EVENTS + DBSCAN NOISE TABLES
# =====================================================

st.subheader("SECURITY EVENT INVESTIGATION")
tab_suspicious, tab_noise, tab_firewall = st.tabs(
    ["Şübhəli web hadisələri", "DBSCAN Noise", "Firewall hadisələri"]
)

display_columns = [
    "timestamp",
    "src_ip",
    "method",
    "path",
    "status",
    "user_agent_type",
    "sqli_indicator_count",
    "kmeans_cluster",
    "dbscan_cluster"
]

with tab_suspicious:
    if not suspicious_df.empty:
        suspicious_view = suspicious_df[display_columns].sort_values(
            "timestamp", ascending=False
        )
        st.dataframe(
            suspicious_view,
            use_container_width=True,
            hide_index=True,
            height=390
        )
        csv_data = suspicious_view.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "⬇️ Şübhəli hadisələri CSV kimi yüklə",
            data=csv_data,
            file_name="suspicious_events.csv",
            mime="text/csv"
        )
    else:
        st.info("Seçilmiş filterlər üzrə şübhəli hadisə yoxdur.")

with tab_noise:
    if not dbscan_filtered.empty:
        noise_view = dbscan_filtered[
            dbscan_filtered["dbscan_cluster"] == -1
        ].copy()
        noise_columns = [
            "timestamp",
            "src_ip",
            "method",
            "path",
            "status",
            "user_agent_type",
            "sqli_indicator_count",
            "dbscan_cluster"
        ]
        if not noise_view.empty:
            st.dataframe(
                noise_view[noise_columns].sort_values(
                    "timestamp", ascending=False
                ),
                use_container_width=True,
                hide_index=True,
                height=390
            )
        else:
            st.success("Seçilmiş filterlərdə DBSCAN noise hadisəsi yoxdur.")
    else:
        st.info("DBSCAN nəticəsi yoxdur.")

with tab_firewall:
    if not firewall_filtered.empty:
        firewall_columns = [
            "timestamp",
            "src_ip",
            "dst_ip",
            "dst_port",
            "protocol",
            "action",
            "bytes_sent",
            "bytes_received",
            "dst_country"
        ]
        available_columns = [
            col for col in firewall_columns if col in firewall_filtered.columns
        ]
        st.dataframe(
            firewall_filtered[available_columns].sort_values(
                "timestamp", ascending=False
            ),
            use_container_width=True,
            hide_index=True,
            height=390
        )
    else:
        st.info("Firewall hadisəsi yoxdur.")


# =====================================================
# FOOTER
# =====================================================

st.divider()
