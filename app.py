import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import HistGradientBoostingRegressor, IsolationForest
from datetime import datetime, time
from chatbot_simulador import responder_simulador
from ev_common import (
    GREETING,
    UNIDENTIFIED,
    apartment_label,
    delivered_kwh,
    escape_dollars,
    format_brl,
    format_number,
    format_pct,
)

warnings.filterwarnings("ignore")
st.set_page_config(page_title="EV ChargeOps", page_icon="⚡", layout="wide")

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.4rem; padding-bottom: 2rem;}
    [data-testid="stMetric"] {
        background: #F7F9FC;
        border: 1px solid #E6EAF0;
        padding: 14px;
        border-radius: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

TS_FMT = "%a, %d %b %Y %H:%M:%S GMT"


def read_sessions_json(source):
    if isinstance(source, (bytes, bytearray)):
        txt = source.decode("utf-8")
    elif hasattr(source, "getvalue"):
        raw = source.getvalue()
        txt = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else raw
    elif hasattr(source, "read"):
        raw = source.read()
        txt = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else raw
    else:
        txt = Path(source).read_text(encoding="utf-8")

    match = re.search(r'"site":\s*"(\w+)"', txt)
    site = match.group(1) if match else "?"

    try:
        return json.loads(txt)["_items"], False, site
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        whitespace = re.compile(r"\s*,?\s*")
        position = txt.index("[", txt.index('"_items"')) + 1
        items = []

        while True:
            position = whitespace.match(txt, position).end()
            if position >= len(txt) or txt[position] == "]":
                break
            try:
                obj, position = decoder.raw_decode(txt, position)
            except json.JSONDecodeError:
                break
            items.append(obj)

        return items, True, site


@st.cache_data(show_spinner="Preparando sessões...")
def load_sessions(raw, timezone, rated_kw, outlier_factor):
    items, truncated, site = read_sessions_json(raw)
    data = pd.DataFrame(items)

    required_columns = {
        "connectionTime",
        "disconnectTime",
        "doneChargingTime",
        "kWhDelivered",
        "sessionID",
        "stationID",
        "userID",
    }
    missing = required_columns.difference(data.columns)
    if missing:
        raise ValueError(f"Colunas obrigatórias ausentes no JSON: {sorted(missing)}")

    for column in ["connectionTime", "disconnectTime", "doneChargingTime"]:
        data[column] = pd.to_datetime(
            data[column], format=TS_FMT, utc=True, errors="coerce"
        )

    data = data.rename(
        columns={
            "connectionTime": "conn",
            "disconnectTime": "disc",
            "doneChargingTime": "done",
            "kWhDelivered": "kwh",
            "sessionID": "session_id",
            "stationID": "station",
        }
    )

    # userInputs (lista) não é usado e impede o hash rápido do DataFrame no cache do Streamlit.
    data = data.drop(columns=["userInputs"], errors="ignore")

    data = (
        data.drop_duplicates("session_id")
        .dropna(subset=["conn", "disc", "kwh"])
        .sort_values("conn")
        .reset_index(drop=True)
    )
    data["station"] = data["station"].astype(str)

    data["apartment"] = data["userID"].map(apartment_label)

    one_minute = pd.Timedelta(minutes=1)
    data["f_no_done"] = data["done"].isna()
    data["f_bad_window"] = data["done"].notna() & (
        data["done"] <= data["conn"] + one_minute
    )

    charge_end = data["done"].where(
        ~(data["f_no_done"] | data["f_bad_window"]), data["disc"]
    )
    data["chg_end"] = charge_end.where(charge_end <= data["disc"], data["disc"])

    def hours_between(later, earlier):
        return (later - earlier).dt.total_seconds() / 3600

    data["dur_h"] = hours_between(data["disc"], data["conn"])
    data["chg_h"] = hours_between(data["chg_end"], data["conn"])
    data["idle_h"] = hours_between(data["disc"], data["chg_end"])
    data["avg_kw"] = data["kwh"] / data["chg_h"].replace(0, np.nan)
    data["conn_local"] = data["conn"].dt.tz_convert(timezone)
    data["disc_local"] = data["disc"].dt.tz_convert(timezone)
    data["f_power_outlier"] = data["avg_kw"] > outlier_factor * rated_kw
    data["f_long"] = data["dur_h"] > 24
    data["f_no_energy"] = (data["kwh"] < 1) & (data["dur_h"] >= 4)

    return data, truncated, site


@st.cache_data(show_spinner=False)
def spread_energy(data, step_min):
    step = pd.Timedelta(minutes=step_min)
    rows = []

    for index, (start, finish, kwh) in enumerate(
        zip(data["conn"], data["chg_end"], data["kwh"])
    ):
        total_seconds = (finish - start).total_seconds()
        current = start
        if total_seconds <= 0:
            continue

        while current < finish:
            next_time = min(current.floor(f"{step_min}min") + step, finish)
            block_kwh = kwh * (next_time - current).total_seconds() / total_seconds
            rows.append((index, current.floor(f"{step_min}min"), block_kwh))
            current = next_time

    return pd.DataFrame(rows, columns=["i", "ts", "kwh"])


def format_datetime(series):
    return series.dt.strftime("%d/%m/%Y %H:%M")


st.sidebar.title("⚙️ Configuração")
upload = st.sidebar.file_uploader("Dataset ACN-Data (.json)", type="json")
default_path = Path("acndata_sessions.json")

if upload is not None:
    raw = upload.getvalue()
elif default_path.exists():
    raw = default_path.read_bytes()
else:
    st.info("Envie o arquivo acndata_sessions.json na barra lateral.")
    st.stop()

timezone = st.sidebar.selectbox(
    "Fuso do site", ["America/Los_Angeles", "America/Sao_Paulo"], index=0
)
rated_kw = st.sidebar.number_input("Potência nominal (kW)", 1.0, 50.0, 6.6, 0.1)
outlier_factor = st.sidebar.number_input(
    "Fator de alerta de potência", 1.0, 3.0, 1.1, 0.1
)

st.sidebar.markdown("**Tarifas ilustrativas**")
t_fora = st.sidebar.number_input("Horário econômico (R$/kWh)", 0.0, 10.0, 0.60, 0.05)
t_inter = st.sidebar.number_input(
    "Horário intermediário (R$/kWh)", 0.0, 10.0, 0.95, 0.05
)
t_ponta = st.sidebar.number_input("Horário mais caro (R$/kWh)", 0.0, 10.0, 1.60, 0.05)
mgmt_fee = st.sidebar.number_input("Taxa de gestão (R$/kWh)", 0.0, 10.0, 0.08, 0.01)
idle_grace = st.sidebar.number_input(
    "Tolerância após o fim da carga (h)", 0.0, 24.0, 0.5, 0.25
)
idle_fee = st.sidebar.number_input(
    "Permanência adicional (R$/h)", 0.0, 100.0, 2.0, 0.5
)
demand_charge = st.sidebar.number_input(
    "Custo de demanda (R$/kW/mês)", 0.0, 1000.0, 45.0, 5.0
)

st.sidebar.caption("Horários considerados na simulação")
st.sidebar.markdown(
    """
    - **Econômico:** 00:00–16:59 e 22:00–23:59
    - **Intermediário:** 17:00–17:59 e 21:00–21:59
    - **Mais caro:** 18:00–20:59
    - Finais de semana: horário econômico
    """
)

try:
    df, truncated, site = load_sessions(raw, timezone, rated_kw, outlier_factor)
except Exception as error:
    st.error(f"Não foi possível preparar o dataset: {error}")
    st.stop()

seg = spread_energy(df, 60)
if seg.empty:
    st.error("Não foi possível construir a curva de consumo a partir do dataset.")
    st.stop()

local_times = seg["ts"].dt.tz_convert(timezone)
weekday = local_times.dt.dayofweek < 5
hour = local_times.dt.hour
seg["price_period"] = np.where(
    weekday & hour.isin([18, 19, 20]),
    "Horário mais caro",
    np.where(
        weekday & hour.isin([17, 21]),
        "Horário intermediário",
        "Horário econômico",
    ),
)

tariff = {
    "Horário econômico": t_fora,
    "Horário intermediário": t_inter,
    "Horário mais caro": t_ponta,
}
seg["energy_cost"] = seg["kwh"] * seg["price_period"].map(tariff)

billing_sessions = df.copy()
billing_sessions["energy_cost"] = (
    seg.groupby("i")["energy_cost"]
    .sum()
    .reindex(billing_sessions.index, fill_value=0)
)
billing_sessions["billable_idle_hours"] = (
    billing_sessions["idle_h"] - idle_grace
).clip(lower=0)
billing_sessions["billing_month"] = billing_sessions["disc_local"].dt.strftime(
    "%Y-%m"
)

daily_kwh = df.groupby(df["conn_local"].dt.date)["kwh"].sum()

apartment_bills = (
    billing_sessions.groupby(["billing_month", "apartment"])
    .agg(
        recargas=("kwh", "size"),
        consumo_kwh=("kwh", "sum"),
        custo_energia=("energy_cost", "sum"),
        horas_excedentes=("billable_idle_hours", "sum"),
    )
    .reset_index()
)
apartment_bills["taxa_gestao"] = apartment_bills["consumo_kwh"] * mgmt_fee
apartment_bills["custo_ocupacao_excedente"] = (
    apartment_bills["horas_excedentes"] * idle_fee
)
apartment_bills["valor_total"] = (
    apartment_bills["custo_energia"]
    + apartment_bills["taxa_gestao"]
    + apartment_bills["custo_ocupacao_excedente"]
)

st.title("⚡ EV ChargeOps")
st.caption(
    "Gestão operacional de recarga de veículos elétricos para condomínios | Enterprise Challenge 2026"
)

if truncated:
    st.warning(
        "O JSON está truncado. O aplicativo recuperou somente as sessões completas."
    )

st.caption(
    "Demonstração: cada userID do dataset foi tratado como um apartamento. "
    "Em produção, a associação dependeria do cadastro do condomínio."
)

tabs = st.tabs(
    [
        "Resumo do condomínio",
        "Gastos por apartamento",
        "Controle de demanda",
        "Previsão de consumo",
        "Manutenção e alertas",
        "Monitoramento",
        "Assistente virtual"
    ]
)

with tabs[0]:
    total_cost = apartment_bills["valor_total"].sum()
    unidentified_kwh = df.loc[df["apartment"] == UNIDENTIFIED, "kwh"].sum()
    unidentified_pct = 100 * unidentified_kwh / df["kwh"].sum()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Recargas realizadas", format_number(len(df), 0))
    c2.metric("Consumo total", f"{format_number(df['kwh'].sum(), 0)} kWh")
    c3.metric("Gasto estimado", format_brl(total_cost))
    c4.metric(
        "Apartamentos identificados",
        df.loc[df["apartment"] != UNIDENTIFIED, "apartment"].nunique(),
    )
    c5.metric("Consumo não identificado", format_pct(unidentified_pct))

    st.caption(
        f"Site: {site} | Período local: {df['conn_local'].min():%d/%m/%Y %H:%M} "
        f"até {df['conn_local'].max():%d/%m/%Y %H:%M}"
    )

    chart_col1, chart_col2 = st.columns(2)

    arrivals = (
        df.assign(
            hora=df["conn_local"].dt.hour,
            tipo=np.where(
                df["conn_local"].dt.dayofweek < 5,
                "Dia útil",
                "Fim de semana",
            ),
        )
        .groupby(["hora", "tipo"])
        .size()
        .reset_index(name="Recargas")
    )
    fig_arrivals = px.bar(
        arrivals,
        x="hora",
        y="Recargas",
        color="tipo",
        barmode="group",
        title="Horários com mais chegadas",
        labels={"hora": "Hora do dia", "tipo": "Tipo de dia"},
    )
    fig_arrivals.update_xaxes(tickmode="linear", dtick=2)
    chart_col1.plotly_chart(fig_arrivals, width="stretch")

    daily = daily_kwh.rename_axis("Data").reset_index(name="Consumo (kWh)")
    fig_daily = px.bar(
        daily,
        x="Data",
        y="Consumo (kWh)",
        title="Consumo diário do condomínio",
    )
    chart_col2.plotly_chart(fig_daily, width="stretch")

    period_consumption = (
        seg.groupby("price_period")["kwh"].sum().reset_index(name="Consumo (kWh)")
    )
    period_consumption["Participação (%)"] = (
        100 * period_consumption["Consumo (kWh)"] / period_consumption["Consumo (kWh)"].sum()
    )
    st.plotly_chart(
        px.bar(
            period_consumption,
            x="price_period",
            y="Consumo (kWh)",
            color="price_period",
            title="Consumo por faixa de preço da energia",
            labels={"price_period": "Faixa de preço"},
        ),
        width="stretch",
    )

    st.subheader("Qualidade dos dados")
    quality = pd.DataFrame(
        {
            "Indicador": [
                "Recargas sem apartamento identificado",
                "Sem sinal de fim da carga",
                "Janela de carga inválida",
                "Potência média acima do limite esperado",
                "Conexões com mais de 24 horas",
                "Conectado por 4 horas ou mais com menos de 1 kWh",
            ],
            "Valor": [
                format_pct(100 * df["userID"].isna().mean()),
                format_number(df["f_no_done"].sum(), 0),
                format_number(df["f_bad_window"].sum(), 0),
                format_number(df["f_power_outlier"].sum(), 0),
                format_number(df["f_long"].sum(), 0),
                format_number(df["f_no_energy"].sum(), 0),
            ],
        }
    )
    st.dataframe(quality, width="stretch", hide_index=True)

with tabs[1]:
    st.header("Gastos por apartamento")
    st.info(
        "O valor de cada apartamento é formado pelo consumo de energia, "
        "pela taxa de gestão e, quando aplicável, pela permanência na vaga "
        "após o término da recarga."
    )

    month_options = sorted(apartment_bills["billing_month"].unique())
    selected_month = st.selectbox(
        "Mês da cobrança", month_options, index=len(month_options) - 1
    )

    monthly_bills = apartment_bills[
        apartment_bills["billing_month"] == selected_month
    ].sort_values("valor_total", ascending=False)
    identified = monthly_bills[
        monthly_bills["apartment"] != UNIDENTIFIED
    ]

    total_month_kwh = monthly_bills["consumo_kwh"].sum()
    unidentified_month_kwh = monthly_bills.loc[
        monthly_bills["apartment"] == UNIDENTIFIED, "consumo_kwh"
    ].sum()
    unidentified_month_pct = (
        100 * unidentified_month_kwh / total_month_kwh if total_month_kwh else 0
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Consumo do mês", f"{format_number(total_month_kwh, 0)} kWh")
    c2.metric("Gasto do mês", format_brl(monthly_bills["valor_total"].sum()))
    c3.metric("Apartamentos com cobrança", identified["apartment"].nunique())
    c4.metric("Consumo não identificado", format_pct(unidentified_month_pct))

    chart_data = identified.head(12).melt(
        id_vars="apartment",
        value_vars=[
            "custo_energia",
            "taxa_gestao",
            "custo_ocupacao_excedente",
        ],
        var_name="componente",
        value_name="valor",
    )
    component_names = {
        "custo_energia": "Energia consumida",
        "taxa_gestao": "Taxa de gestão",
        "custo_ocupacao_excedente": "Permanência adicional na vaga",
    }
    chart_data["componente"] = chart_data["componente"].map(component_names)

    st.plotly_chart(
        px.bar(
            chart_data,
            x="apartment",
            y="valor",
            color="componente",
            barmode="stack",
            title="Apartamentos com os maiores gastos",
            labels={
                "apartment": "Apartamento",
                "valor": "Valor estimado (R$)",
                "componente": "Composição do gasto",
            },
        ),
        width="stretch",
    )

    display_bills = monthly_bills.rename(
        columns={
            "apartment": "Apartamento",
            "recargas": "Recargas realizadas",
            "consumo_kwh": "Consumo (kWh)",
            "custo_energia": "Energia (R$)",
            "taxa_gestao": "Gestão (R$)",
            "horas_excedentes": "Permanência adicional (h)",
            "custo_ocupacao_excedente": "Permanência adicional (R$)",
            "valor_total": "Total estimado (R$)",
        }
    )
    display_columns = [
        "Apartamento",
        "Recargas realizadas",
        "Consumo (kWh)",
        "Energia (R$)",
        "Gestão (R$)",
        "Permanência adicional (h)",
        "Permanência adicional (R$)",
        "Total estimado (R$)",
    ]
    st.dataframe(
        display_bills[display_columns].round(2),
        width="stretch",
        hide_index=True,
    )
    st.download_button(
        "Baixar gastos por apartamento",
        apartment_bills.round(2).to_csv(index=False).encode("utf-8-sig"),
        "gastos_por_apartamento.csv",
        "text/csv",
    )


def allocate(requests, cap_kw, policy):
    demands = {request["id"]: request["demand_kw"] for request in requests}
    if policy == "uncontrolled" or sum(demands.values()) <= cap_kw:
        return demands

    output = {}
    remaining = cap_kw

    if policy == "equal":
        order = sorted(demands, key=demands.get)
        for position, request_id in enumerate(order):
            count_left = len(order) - position
            output[request_id] = min(demands[request_id], remaining / count_left)
            remaining -= output[request_id]
        return output

    priority = {
        "fcfs": lambda request: request["arrival"],
        "llf": lambda request: request["laxity_h"],
    }[policy]

    for request in sorted(requests, key=priority):
        request_id = request["id"]
        output[request_id] = min(demands[request_id], max(remaining, 0))
        remaining -= output[request_id]

    return output


@st.cache_data(show_spinner="Simulando políticas de controle...")
def simulate(data, cap_kw, policy, rated_power_kw, step_min=15):
    step_hours = step_min / 60
    initial_time = data["conn"].min().floor("h")
    connections = (
        (data["conn"] - initial_time).dt.total_seconds() / 3600
    ).to_numpy()
    disconnections = (
        (data["disc"] - initial_time).dt.total_seconds() / 3600
    ).to_numpy()
    remaining_energy = data["kwh"].to_numpy(copy=True)
    original_energy = data["kwh"].to_numpy(copy=True)
    load = np.zeros(int(np.ceil(disconnections.max() / step_hours)) + 1)

    for step_index in range(len(load)):
        start = step_index * step_hours
        finish = (step_index + 1) * step_hours
        active = np.where(
            (connections < finish)
            & (disconnections > start)
            & (remaining_energy > 1e-9)
        )[0]

        requests = []
        for session_index in active:
            overlap = min(disconnections[session_index], finish) - max(
                connections[session_index], start
            )
            requests.append(
                {
                    "id": session_index,
                    "demand_kw": min(
                        rated_power_kw * overlap, remaining_energy[session_index]
                    )
                    / step_hours,
                    "laxity_h": disconnections[session_index]
                    - max(connections[session_index], start)
                    - remaining_energy[session_index] / rated_power_kw,
                    "arrival": connections[session_index],
                }
            )

        effective_cap = cap_kw if cap_kw is not None else 1e12
        allocation = allocate(requests, effective_cap, policy)
        for session_index, power in allocation.items():
            remaining_energy[session_index] -= power * step_hours
            load[step_index] += power

    delivered = original_energy - remaining_energy
    load_index = initial_time + pd.to_timedelta(
        np.arange(len(load)) * step_min, unit="min"
    )
    return (
        pd.Series(load, index=load_index),
        float(load.max()),
        float(delivered.sum() / original_energy.sum()),
        int((delivered < 0.95 * original_energy).sum()),
    )


with tabs[2]:
    st.header("Controle de demanda elétrica")
    st.write(
        "Compare formas de distribuir a potência disponível entre os veículos "
        "e veja o impacto no pico do condomínio."
    )

    base_load, base_peak, _, base_short = simulate(
        df, None, "uncontrolled", rated_kw
    )
    fraction = st.slider(
        "Limite de potência em relação ao pico sem controle",
        min_value=20,
        max_value=100,
        value=30,
        step=5,
        format="%d%%",
    )
    cap_kw = base_peak * fraction / 100

    labels = {
        "equal": "Divisão igual",
        "fcfs": "Ordem de chegada",
        "llf": "Menor folga primeiro",
    }
    result_rows = []
    curves = {"Sem controle": base_load}

    for policy in ["equal", "fcfs", "llf"]:
        load, peak, fulfilment, short_sessions = simulate(
            df, cap_kw, policy, rated_kw
        )
        curves[labels[policy]] = load
        result_rows.append(
            {
                "Política": labels[policy],
                "Energia atendida (%)": 100 * fulfilment,
                "Recargas abaixo de 95%": short_sessions,
                "Recargas adicionais afetadas": max(short_sessions - base_short, 0),
                "Pico (kW)": peak,
            }
        )

    c1, c2, c3 = st.columns(3)
    c1.metric("Pico sem controle", f"{base_peak:.1f} kW")
    c2.metric("Limite escolhido", f"{cap_kw:.1f} kW")
    c3.metric(
        "Redução indicativa de demanda",
        format_brl(max(base_peak - cap_kw, 0) * demand_charge) + "/mês",
    )

    st.dataframe(
        pd.DataFrame(result_rows).round(2),
        hide_index=True,
        width="stretch",
    )

    busy_day = daily_kwh.idxmax()
    start_plot = pd.Timestamp(busy_day, tz=timezone) + pd.Timedelta(hours=6)
    end_plot = start_plot + pd.Timedelta(hours=14)

    figure = go.Figure()
    for name, curve in curves.items():
        visible_curve = curve.tz_convert(timezone).loc[start_plot:end_plot]
        figure.add_scatter(
            x=visible_curve.index,
            y=visible_curve.values,
            name=name,
            line_shape="hv",
        )
    figure.add_hline(y=cap_kw, line_dash="dash", annotation_text="Limite")
    figure.update_layout(
        title=f"Demanda no dia mais carregado: {busy_day:%d/%m/%Y}",
        yaxis_title="Potência (kW)",
        xaxis_title="Horário",
    )
    st.plotly_chart(figure, width="stretch")


@st.cache_data(show_spinner="Treinando previsão para o próximo dia...")
def forecast_next(data, timezone):
    hourly_blocks = spread_energy(data, 60)
    hourly_energy = hourly_blocks.groupby("ts")["kwh"].sum()
    start = (
        data["conn"].min().tz_convert(timezone).normalize()
        + pd.DateOffset(days=2)
    ).tz_convert("UTC")
    # O histórico vai até a última hora COMPLETA observada. A hora da última conexão fica de fora, pois
    # sessões posteriores a ela não estão no arquivo (por exemplo, em exportações truncadas).
    end = data["conn"].max().floor("h")
    series = hourly_energy.reindex(
        pd.date_range(start, end, freq="h", inclusive="left"), fill_value=0.0
    )

    def features(values):
        local_index = values.index.tz_convert(timezone)
        feature_data = pd.DataFrame(index=values.index)
        feature_data["hour"] = local_index.hour
        feature_data["dow"] = local_index.dayofweek
        feature_data["weekend"] = (local_index.dayofweek >= 5).astype(int)
        lags = {day: values.shift(24 * day) for day in range(1, 8)}
        feature_data["lag_24h"] = lags[1]
        feature_data["lag_48h"] = lags[2]
        feature_data["lag_168h"] = lags[7]
        feature_data["mean_same_hour_7d"] = pd.concat(
            lags.values(), axis=1
        ).mean(axis=1)
        feature_data["mean_last_7d"] = values.shift(24).rolling(
            168, min_periods=24
        ).mean()
        return feature_data

    feature_data = features(series)
    valid = feature_data.notna().all(axis=1)
    if valid.sum() < 50:
        raise ValueError("Histórico insuficiente para treinar a previsão.")

    model_parameters = {
        "max_iter": 180,
        "learning_rate": 0.04,
        "max_depth": 3,
        "min_samples_leaf": 15,
        "l2_regularization": 1.0,
        "random_state": 0,
    }
    mean_model = HistGradientBoostingRegressor(
        loss="poisson", **model_parameters
    ).fit(feature_data.loc[valid], series.loc[valid])
    lower_model = HistGradientBoostingRegressor(
        loss="quantile", quantile=0.1, **model_parameters
    ).fit(feature_data.loc[valid], series.loc[valid])
    upper_model = HistGradientBoostingRegressor(
        loss="quantile", quantile=0.9, **model_parameters
    ).fit(feature_data.loc[valid], series.loc[valid])

    next_index = pd.date_range(
        series.index[-1] + pd.Timedelta(hours=1), periods=24, freq="h"
    )
    extended = pd.concat([series, pd.Series(np.nan, index=next_index)])
    next_features = features(extended).loc[next_index]

    prediction = np.clip(mean_model.predict(next_features), 0, None)
    lower = np.minimum(
        np.clip(lower_model.predict(next_features), 0, None), prediction
    )
    upper = np.maximum(upper_model.predict(next_features), prediction)

    forecast = pd.DataFrame(
        {"Previsão (kWh)": prediction, "P10": lower, "P90": upper},
        index=next_index.tz_convert(timezone),
    )
    return forecast


with tabs[3]:
    st.header("Previsão de consumo")
    try:
        next_day = forecast_next(df, timezone)
        c1, c2, c3 = st.columns(3)
        c1.metric("Consumo previsto", f"{next_day['Previsão (kWh)'].sum():.0f} kWh")
        c2.metric("Pico previsto", f"{format_number(next_day['Previsão (kWh)'].max(), 1)} kWh/h")
        c3.metric(
            "Horário estimado do pico",
            f"{next_day['Previsão (kWh)'].idxmax():%H:%M}",
        )

        figure = go.Figure()
        figure.add_bar(
            x=next_day.index,
            y=next_day["Previsão (kWh)"],
            name="Consumo previsto",
        )
        figure.add_scatter(
            x=list(next_day.index) + list(next_day.index[::-1]),
            y=list(next_day["P90"]) + list(next_day["P10"][::-1]),
            fill="toself",
            line={"color": "rgba(0,0,0,0)"},
            fillcolor="rgba(29,158,117,.2)",
            name="Faixa P10–P90",
        )
        figure.update_layout(
            title=f"Previsão para as 24 horas a partir de {next_day.index[0]:%d/%m/%Y %H:%M}",
            yaxis_title="Consumo por hora (kWh)",
            xaxis_title="Horário",
        )
        st.plotly_chart(figure, width="stretch")
        st.info(
            "A faixa P10–P90 representa um intervalo estimado pelo modelo. "
            "Ela não é uma garantia operacional."
        )
        st.download_button(
            "Baixar previsão",
            next_day.round(2).to_csv().encode("utf-8-sig"),
            "previsao_proximo_dia.csv",
            "text/csv",
        )
    except Exception as error:
        st.warning(f"A previsão não pôde ser calculada: {error}")


ALERT_COLUMNS = ["Data", "Carregador", "Apartamento", "Sessão", "Tipo", "Severidade", "Descrição"]


def alert_row(record, kind, severity, message):
    return {
        "Data": record.disc_local,
        "Carregador": record.station,
        "Apartamento": record.apartment,
        "Sessão": record.session_id,
        "Tipo": kind,
        "Severidade": severity,
        "Descrição": message,
    }


def rule_alerts(record, rated_power_kw, outlier_limit):
    """Alertas por regra de uma sessão: gera (tipo, severidade, descrição)."""
    if record.f_no_done:
        yield (
            "Sem sinal de fim da carga",
            "Atenção",
            f"Sem horário de término ({record.kwh:.1f} kWh em {record.dur_h:.1f} h).",
        )
    if record.f_bad_window:
        yield (
            "Janela de carga inválida",
            "Atenção",
            "O término registrado ocorreu antes ou muito próximo do início.",
        )
    if record.f_power_outlier:
        yield (
            "Potência acima do esperado",
            "Atenção",
            f"Média de {record.avg_kw:.1f} kW; limite de referência {outlier_limit * rated_power_kw:.1f} kW.",
        )
    if record.f_no_energy:
        yield (
            "Conexão com pouca energia",
            "Atenção",
            f"Conectado por {record.dur_h:.1f} h e consumo de {record.kwh:.2f} kWh.",
        )
    if record.f_long:
        yield (
            "Permanência prolongada",
            "Atenção" if record.dur_h > 72 else "Informativo",
            f"Veículo conectado por {record.dur_h:.0f} horas.",
        )


@st.cache_data(show_spinner=False)
def build_alerts(data, rated_power_kw, outlier_limit):
    rows = [
        alert_row(record, *alert)
        for record in data.itertuples()
        for alert in rule_alerts(record, rated_power_kw, outlier_limit)
    ]

    features = pd.DataFrame(
        {
            "kwh": data["kwh"],
            "dur_h": data["dur_h"],
            "chg_h": data["chg_h"],
            "idle_h": data["idle_h"].clip(lower=0),
            "avg_kw": data["avg_kw"].fillna(0).clip(0, 4 * rated_power_kw),
            "hour": data["conn_local"].dt.hour,
        }
    )
    scores = -IsolationForest(n_estimators=200, random_state=0).fit(
        features
    ).score_samples(features)
    threshold = np.quantile(scores, 0.98)

    rows.extend(
        alert_row(
            record,
            "Padrão estatístico incomum",
            "Informativo",
            f"Sessão estatisticamente atípica, score {score:.2f}.",
        )
        for record, score in zip(data.itertuples(), scores)
        if score >= threshold
    )

    return pd.DataFrame(rows, columns=ALERT_COLUMNS).sort_values("Data", ascending=False)


with tabs[4]:
    st.header("Manutenção e alertas")
    alerts = build_alerts(df, rated_kw, outlier_factor)

    c1, c2, c3 = st.columns(3)
    c1.metric("Sinais encontrados", len(alerts))
    c2.metric("Atenção", int((alerts["Severidade"] == "Atenção").sum()))
    c3.metric(
        "Informativos", int((alerts["Severidade"] == "Informativo").sum())
    )

    counts = (
        alerts.groupby(["Tipo", "Severidade"])
        .size()
        .reset_index(name="Quantidade")
    )
    st.plotly_chart(
        px.bar(
            counts,
            x="Quantidade",
            y="Tipo",
            color="Severidade",
            orientation="h",
            title="Sinais para inspeção",
        ),
        width="stretch",
    )

    severities = sorted(alerts["Severidade"].unique())
    selected_severities = st.multiselect(
        "Severidade", severities, default=severities
    )
    filtered_alerts = alerts[alerts["Severidade"].isin(selected_severities)].copy()
    filtered_alerts["Data"] = format_datetime(filtered_alerts["Data"])
    st.dataframe(filtered_alerts, width="stretch", hide_index=True)
    st.caption(
        "Os registros são sinais para inspeção, não falhas confirmadas. "
        "O dataset não inclui códigos reais de erro dos carregadores."
    )
    st.download_button(
        "Baixar alertas",
        alerts.to_csv(index=False).encode("utf-8-sig"),
        "alertas.csv",
        "text/csv",
    )


with tabs[5]:
    st.header("Monitoramento dos carregadores")
    st.write(
        "Escolha uma data e um horário para visualizar a operação do condomínio."
    )

    valid_dates = sorted(df["conn_local"].dt.date.unique())
    selected_date = st.select_slider(
        "Data do monitoramento",
        options=valid_dates,
        value=valid_dates[len(valid_dates) // 2],
        format_func=lambda value: value.strftime("%d/%m/%Y"),
    )

    monitoring_times = [
        f"{hour_value:02d}:{minute_value:02d}"
        for hour_value in range(24)
        for minute_value in (0, 15, 30, 45)
    ]
    monitoring_times.append("23:59")

    selected_time = st.select_slider(
        "Horário do monitoramento",
        options=monitoring_times,
        value="08:00",
    )
    selected_hour, selected_minute = map(int, selected_time.split(":"))

    local_moment = pd.Timestamp(selected_date, tz=timezone) + pd.Timedelta(
        hours=selected_hour, minutes=selected_minute
    )
    utc_moment = local_moment.tz_convert("UTC")

    active_sessions = df[
        (df["conn"] <= utc_moment) & (df["disc"] > utc_moment)
    ].copy()
    active_sessions["status"] = np.where(
        utc_moment < active_sessions["chg_end"],
        "Carregando",
        "Conectado após o término",
    )
    active_sessions["energy_so_far"] = delivered_kwh(active_sessions, utc_moment)
    active_sessions["estimated_power_kw"] = np.where(
        active_sessions["status"] == "Carregando",
        np.minimum(active_sessions["avg_kw"].fillna(0), rated_kw),
        0,
    )

    charging_count = int((active_sessions["status"] == "Carregando").sum())
    idle_count = int(
        (active_sessions["status"] == "Conectado após o término").sum()
    )
    available_count = max(
        int(df["station"].nunique() - active_sessions["station"].nunique()), 0
    )
    estimated_power = active_sessions["estimated_power_kw"].sum()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Carregadores em uso", charging_count)
    c2.metric("Após o fim da carga", idle_count)
    c3.metric("Carregadores disponíveis", available_count)
    c4.metric("Demanda estimada", f"{estimated_power:.1f} kW")

    if active_sessions.empty:
        st.info("Não havia veículos conectados no horário selecionado.")
    else:
        monitoring_table = active_sessions[
            [
                "station",
                "apartment",
                "status",
                "estimated_power_kw",
                "energy_so_far",
                "conn_local",
                "disc_local",
            ]
        ].rename(
            columns={
                "station": "Carregador",
                "apartment": "Apartamento",
                "status": "Situação",
                "estimated_power_kw": "Potência estimada (kW)",
                "energy_so_far": "Energia entregue até o momento (kWh)",
                "conn_local": "Horário de conexão",
                "disc_local": "Horário de saída",
            }
        )
        for column in ["Horário de conexão", "Horário de saída"]:
            monitoring_table[column] = format_datetime(monitoring_table[column])
        monitoring_table = monitoring_table.sort_values(
            ["Situação", "Potência estimada (kW)"], ascending=[True, False]
        )
        st.dataframe(
            monitoring_table.round(2), width="stretch", hide_index=True
        )

    st.caption(
        "A potência instantânea é estimada a partir dos dados da sessão. "
        "O dataset não possui medição instantânea real."
    )

with tabs[6]:
    st.header("Assistente virtual")

    st.caption(
        "Consulte gastos, consumo e status de recarga dos apartamentos."
    )

    st.info(
        "Este é um assistente demonstrativo. As respostas são produzidas "
        "por regras e consultas diretas aos dados calculados pelo dashboard, "
        "sem uso de uma API externa."
    )

    # Inicializa o histórico somente na primeira execução
    if "ev_chat_messages" not in st.session_state:
        st.session_state.ev_chat_messages = [
            {
                "role": "assistant",
                "content": GREETING,
            }
        ]

    st.markdown(
        """
        **Exemplos de perguntas**

        - Quanto o apartamento 891 gastou em novembro?
        - Qual foi o consumo do apartamento 1111?
        - Qual é o status do apartamento 891?
        - Qual foi o gasto total do condomínio em novembro?
        - Qual apartamento teve o maior gasto?
        """
    )

    # Consultas de status precisam de uma data e horário,
    # pois o dataset representa um histórico de 2018.
    with st.expander(
        "Data e horário para consultas de status",
        expanded=False,
    ):
        chatbot_date = st.date_input(
            "Data de referência",
            value=df["conn_local"].dt.date.max(),
            min_value=df["conn_local"].dt.date.min(),
            max_value=df["conn_local"].dt.date.max(),
            key="chatbot_status_date",
        )

        chatbot_time = st.time_input(
            "Horário de referência",
            value=time(8, 0),
            step=900,
            key="chatbot_status_time",
        )

    selected_moment = pd.Timestamp(
        datetime.combine(
            chatbot_date,
            chatbot_time,
        ),
        tz=timezone,
    )

    st.caption(
        "As consultas de status considerarão "
        f"{selected_moment:%d/%m/%Y às %H:%M}."
    )

    # Área onde as mensagens são exibidas
    chat_container = st.container(
        height=420,
        border=True,
    )

    with chat_container:
        for message in st.session_state.ev_chat_messages:
            with st.chat_message(message["role"]):
                st.markdown(escape_dollars(message["content"]))

    # Campo de entrada do chatbot
    question = st.chat_input(
        "Digite uma pergunta sobre um apartamento...",
        key="ev_simulated_chat_input",
    )

    if question:
        # Salva a mensagem do usuário
        st.session_state.ev_chat_messages.append(
            {
                "role": "user",
                "content": question,
            }
        )

        try:
            answer = responder_simulador(
                pergunta=question,
                faturas=apartment_bills,
                sessoes=df,
                momento_local=selected_moment,
            )

        except Exception as error:
            answer = (
                "Não consegui processar essa consulta. "
                f"Detalhes técnicos: {error}"
            )

        # Salva a resposta do assistente
        st.session_state.ev_chat_messages.append(
            {
                "role": "assistant",
                "content": answer,
            }
        )

        # Executa novamente para mostrar as mensagens atualizadas
        st.rerun()

    button_col, information_col = st.columns([1, 3])

    with button_col:
        if st.button(
            "Limpar conversa",
            key="clear_ev_simulated_chat",
            width="stretch",
        ):
            st.session_state.ev_chat_messages = [
                {
                    "role": "assistant",
                    "content": GREETING,
                }
            ]

            st.rerun()

    with information_col:
        st.caption(
            "O assistente consulta diretamente as tabelas de faturamento "
            "e o histórico de recargas do dashboard."
        )

st.divider()

st.caption(
    "EV ChargeOps | Valores tarifários e limites são premissas configuráveis. "
    "Valide-os antes de qualquer aplicação real."
)
