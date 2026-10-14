"""Testes do assistente virtual. Rodar na raiz do projeto:  pip install pytest && pytest"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chatbot_simulador import (  # noqa: E402
    classificar_intencao,
    extrair_apartamento,
    interpretar_mes,
    responder_simulador,
)
from ev_common import UNIDENTIFIED, delivered_kwh, format_brl, format_number  # noqa: E402

TZ = "America/Los_Angeles"


@pytest.fixture
def faturas():
    linhas = [
        # mês, apartamento, recargas, kWh, energia, gestão, permanência, total
        ("2018-10", "APTO 257", 10, 100.0, 60.0, 8.0, 5.0, 400.48),
        ("2018-10", "APTO 891", 2, 20.0, 12.0, 1.6, 0.0, 13.6),
        ("2018-10", UNIDENTIFIED, 50, 500.0, 300.0, 40.0, 0.0, 340.0),
        ("2018-11", "APTO 1111", 12, 150.0, 90.0, 12.0, 20.0, 724.71),
        ("2018-11", "APTO 891", 3, 30.0, 18.0, 2.4, 0.0, 20.4),
    ]
    cols = ["billing_month", "apartment", "recargas", "consumo_kwh", "custo_energia",
            "taxa_gestao", "custo_ocupacao_excedente", "valor_total"]
    df = pd.DataFrame(linhas, columns=cols)
    df["horas_excedentes"] = 0.0
    return df


@pytest.fixture
def sessoes():
    conn = pd.Timestamp("2018-11-14 06:00", tz=TZ).tz_convert("UTC")
    return pd.DataFrame({
        "apartment": ["APTO 891"],
        "station": ["2-39-1-1"],
        "conn": [conn],
        "chg_end": [conn + pd.Timedelta(hours=4)],
        "disc": [conn + pd.Timedelta(hours=6)],
        "kwh": [12.0],
    })


MOMENTO = pd.Timestamp("2018-11-14 08:00", tz=TZ)


def test_maior_nao_e_confundido_com_maio(faturas):
    assert interpretar_mes("qual apartamento teve o maior gasto?", ["2019-05", "2019-06"]) == (None, None)
    assert interpretar_mes("maior gasto em junho", ["2019-05", "2019-06"]) == ("2019-06", None)


def test_maior_gasto_respeita_o_mes_pedido(faturas):
    resposta = responder_simulador("Qual apartamento teve o maior gasto em outubro?", faturas, None, MOMENTO)
    assert "2018-10" in resposta and "APTO 257" in resposta
    resposta = responder_simulador("Qual apartamento teve o maior gasto em novembro?", faturas, None, MOMENTO)
    assert "2018-11" in resposta and "APTO 1111" in resposta


def test_mes_inexistente_avisa_em_vez_de_responder_outro_mes(faturas):
    r1 = responder_simulador("quanto o apartamento 891 gastou em maio?", faturas, None, MOMENTO)
    r2 = responder_simulador("quanto o apartamento 891 gastou em 2018-05?", faturas, None, MOMENTO)
    for resposta in (r1, r2):
        assert resposta.startswith("Não encontrei")
        assert "2018-10, 2018-11" in resposta
    assert "Não encontrei" in responder_simulador("gasto total do condomínio em março", faturas, None, MOMENTO)


def test_mes_do_apartamento_sem_cobranca_naquele_mes(faturas):
    resposta = responder_simulador("quanto o apartamento 1111 gastou em outubro?", faturas, None, MOMENTO)
    assert resposta.startswith("Não encontrei")


def test_sem_mes_usa_o_mais_recente(faturas):
    assert "2018-11" in responder_simulador("quanto o apartamento 891 gastou?", faturas, None, MOMENTO)


def test_ranking_por_consumo(faturas):
    resposta = responder_simulador("qual apartamento mais consumiu em outubro?", faturas, None, MOMENTO)
    assert "maior consumo" in resposta and "APTO 257" in resposta
    assert UNIDENTIFIED not in resposta   # o balde "não identificado" nunca entra no ranking


@pytest.mark.parametrize("pergunta, esperado", [
    ("Como funciona a energia solar?", "desconhecida"),     # 'ola' dentro de 'solar'
    ("Oi", "saudacao"),
    ("Bom dia!", "saudacao"),
    ("quanto o apto 891 pagou?", "faturamento"),
    ("qual apartamento gastou mais?", "condominio"),
    ("consumo total do apartamento 891", "faturamento"),
    ("consumo total do condomínio", "condominio"),
    ("qual o status do apartamento 891?", "status"),
    ("blablabla", "desconhecida"),
])
def test_classificacao(pergunta, esperado):
    assert classificar_intencao(pergunta) == esperado


def test_status_mostra_energia_ate_o_momento_e_nao_o_total(sessoes):
    resposta = responder_simulador("status do apartamento 891", None, sessoes, MOMENTO)
    # 2 h de uma janela de carga de 4 h -> metade de 12 kWh
    assert "carregando" in resposta and "6,00 kWh" in resposta and "12,00" not in resposta


def test_status_sem_veiculo(sessoes):
    resposta = responder_simulador("status do apartamento 891", None, sessoes, MOMENTO + pd.Timedelta(hours=10))
    assert "não tinha veículo conectado" in resposta


def test_status_exige_fuso(sessoes):
    with pytest.raises(ValueError):
        responder_simulador("status do apartamento 891", None, sessoes, pd.Timestamp("2018-11-14 08:00"))


def test_energia_entregue_e_limitada_ao_total(sessoes):
    depois = (MOMENTO + pd.Timedelta(hours=3)).tz_convert("UTC")
    assert delivered_kwh(sessoes, depois).iloc[0] == pytest.approx(12.0)


def test_formatacao_sem_barra_invertida():
    assert format_brl(1234.5) == "R$ 1.234,50"
    assert format_number(0.5, 0) in ("0", "1")
    assert extrair_apartamento("apto 0891") == "APTO 891"


def test_respostas_nao_contem_escape_de_markdown(faturas):
    assert "\\" not in responder_simulador("quanto o apartamento 891 gastou?", faturas, None, MOMENTO)
