"""Constantes e funções compartilhadas entre app.py e chatbot_simulador.py."""
import pandas as pd

UNIDENTIFIED = "NÃO IDENTIFICADO"

GREETING = (
    "Olá! Sou o assistente demonstrativo do EV ChargeOps. "
    "Posso consultar gastos, consumo e status de recarga dos apartamentos."
)


def apartment_label(user_id):
    """Converte o userID do dataset no rótulo de apartamento (ex.: 891 -> 'APTO 891')."""
    if pd.isna(user_id):
        return UNIDENTIFIED
    try:
        return f"APTO {int(user_id)}"
    except (ValueError, TypeError):
        return f"APTO {user_id}"


def format_number(value, decimals=2):
    """Número no padrão brasileiro: 1.234,56."""
    text = f"{float(value):,.{decimals}f}"
    return text.replace(",", "X").replace(".", ",").replace("X", ".")


def format_brl(value):
    return f"R$ {format_number(value, 2)}"


def format_pct(value, decimals=1):
    return f"{format_number(value, decimals)}%"


def escape_dollars(text):
    """O Streamlit interpreta '$...$' como LaTeX. Use só na hora de exibir texto com valores em R$."""
    return text.replace("$", r"\$")


def delivered_kwh(active, moment_utc):
    """kWh já entregues até `moment_utc` em cada sessão ativa.

    A energia da sessão é distribuída de forma uniforme na janela de carga (conn -> chg_end).
    Evita mostrar o total final de uma sessão que ainda está em andamento.
    """
    window = (active["chg_end"] - active["conn"]).dt.total_seconds().clip(lower=1)
    elapsed = (moment_utc - active["conn"]).dt.total_seconds()
    return active["kwh"] * (elapsed / window).clip(0, 1)
