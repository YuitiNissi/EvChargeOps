"""Assistente virtual por regras do EV ChargeOps (sem API externa)."""
import re
import unicodedata

from ev_common import (
    GREETING,
    UNIDENTIFIED,
    apartment_label,
    delivered_kwh,
    format_brl,
    format_number,
)

MESES = {
    "janeiro": "01", "fevereiro": "02", "marco": "03", "abril": "04",
    "maio": "05", "junho": "06", "julho": "07", "agosto": "08",
    "setembro": "09", "outubro": "10", "novembro": "11", "dezembro": "12",
}

# Termos já normalizados (minúsculas, sem acento). A busca usa palavras inteiras.
TERMOS_CONDOMINIO = ["condominio", "predio", "total geral", "gasto total", "consumo total"]
TERMOS_STATUS = ["status", "carregando", "conectado", "carregador", "terminou", "disponivel", "livre", "situacao"]
TERMOS_FATURAMENTO = [
    "fatura", "faturamento", "gasto", "gastos", "gastou", "gastei", "gasta", "custo", "custou",
    "cobranca", "cobrado", "valor", "pagar", "pagou", "paguei", "pagamento",
    "consumo", "consumiu", "kwh",
]
TERMOS_SAUDACAO = ["oi", "ola", "bom dia", "boa tarde", "boa noite", "ajuda"]

FORMATADORES = {
    "brl": format_brl,
    "kwh": lambda valor: f"{format_number(valor)} kWh",
    "int": lambda valor: str(int(valor)),
}

# Rankings do condomínio: um único lugar define os termos e os campos exibidos.
RANKINGS = {
    "valor_total": {
        "termos": ["maior gasto", "mais gastou", "gastou mais", "gasta mais", "maior valor"],
        "descricao": "gasto",
        "campos": [
            ("Gasto total", "valor_total", "brl"),
            ("Consumo", "consumo_kwh", "kwh"),
            ("Recargas realizadas", "recargas", "int"),
            ("Custo da energia", "custo_energia", "brl"),
            ("Taxa de gestão", "taxa_gestao", "brl"),
            ("Permanência adicional", "custo_ocupacao_excedente", "brl"),
        ],
    },
    "consumo_kwh": {
        "termos": ["maior consumo", "mais consumiu", "consumiu mais", "consome mais"],
        "descricao": "consumo",
        "campos": [
            ("Consumo", "consumo_kwh", "kwh"),
            ("Gasto total", "valor_total", "brl"),
            ("Recargas realizadas", "recargas", "int"),
        ],
    },
}
RANKING_GENERICO = re.compile(r"\bapartamento\s+(teve|tem|com)\s+(o\s+)?maior\b")


def normalizar_texto(texto):
    texto = unicodedata.normalize("NFKD", texto or "")
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    return texto.lower().strip()


def contem_termo(texto, termos):
    """True se alguma expressão aparece como palavra inteira (evita 'maio' em 'maior', 'ola' em 'solar')."""
    return any(re.search(rf"\b{re.escape(termo)}\b", texto) for termo in termos)


def extrair_apartamento(pergunta):
    texto = normalizar_texto(pergunta)
    for padrao in (
        r"\bapartamento\s*(\d+)\b",
        r"\bapto\.?\s*(\d+)\b",
        r"\bapt\.?\s*(\d+)\b",
        r"\bunidade\s*(\d+)\b",
    ):
        encontrado = re.search(padrao, texto)
        if encontrado:
            return apartment_label(encontrado.group(1))
    return None


def interpretar_mes(pergunta, meses_disponiveis):
    """Retorna (mes, mes_indisponivel).

    - sem mês na pergunta        -> (None, None)
    - mês pedido e disponível    -> ('2018-11', None)
    - mês pedido e sem dados     -> (None, 'maio')   para a resposta avisar o usuário
    """
    texto = normalizar_texto(pergunta)
    disponiveis = sorted(str(m) for m in meses_disponiveis)

    iso = re.search(r"\b(20\d{2})[-/](0[1-9]|1[0-2])\b", texto)
    if iso:
        mes = f"{iso.group(1)}-{iso.group(2)}"
        return (mes, None) if mes in disponiveis else (None, mes)

    achados = []
    for nome, numero in MESES.items():
        encontrado = re.search(rf"\b{nome}\b(?:\s*(?:de|/|-)?\s*(20\d{{2}}))?", texto)
        if encontrado:
            achados.append((encontrado.start(), nome, numero, encontrado.group(1)))
    if not achados:
        return None, None

    _, nome, numero, ano = min(achados)
    candidatos = [
        m for m in disponiveis
        if m.endswith(f"-{numero}") and (ano is None or m.startswith(ano))
    ]
    if candidatos:
        return candidatos[-1], None
    return None, (f"{nome} de {ano}" if ano else nome)


def resolver_mes(pergunta, meses_disponiveis):
    """Mês a consultar: o pedido, ou o mais recente se a pergunta não cita mês."""
    mes, indisponivel = interpretar_mes(pergunta, meses_disponiveis)
    if indisponivel:
        return None, indisponivel
    return (mes or max(str(m) for m in meses_disponiveis)), None


def detectar_ranking(texto):
    """Coluna do ranking pedido ('valor_total' ou 'consumo_kwh') ou None."""
    if contem_termo(texto, RANKINGS["consumo_kwh"]["termos"]):
        return "consumo_kwh"
    if contem_termo(texto, RANKINGS["valor_total"]["termos"]) or RANKING_GENERICO.search(texto):
        return "valor_total"
    return None


def classificar_intencao(pergunta):
    texto = normalizar_texto(pergunta)

    if detectar_ranking(texto):
        return "condominio"

    # Com apartamento citado, a consulta é sobre ele ("consumo total do apartamento 5").
    if extrair_apartamento(pergunta):
        if contem_termo(texto, TERMOS_STATUS):
            return "status"
        if contem_termo(texto, TERMOS_FATURAMENTO):
            return "faturamento"

    if contem_termo(texto, TERMOS_CONDOMINIO):
        return "condominio"
    if contem_termo(texto, TERMOS_STATUS):
        return "status"
    if contem_termo(texto, TERMOS_FATURAMENTO):
        return "faturamento"
    if contem_termo(texto, TERMOS_SAUDACAO):
        return "saudacao"
    return "desconhecida"


def responder_faturamento(pergunta, faturas):
    apartamento = extrair_apartamento(pergunta)
    if not apartamento:
        return "Informe o número do apartamento. Por exemplo: quanto o apartamento 891 gastou em novembro?"

    dados_apto = faturas[faturas["apartment"] == apartamento]
    if dados_apto.empty:
        return f"Não encontrei dados de cobrança para o {apartamento}."

    meses = sorted(str(m) for m in dados_apto["billing_month"].unique())
    mes, indisponivel = resolver_mes(pergunta, meses)
    if indisponivel:
        return (
            f"Não encontrei cobrança do {apartamento} em {indisponivel}. "
            f"Meses disponíveis: {', '.join(meses)}."
        )

    linha = dados_apto[dados_apto["billing_month"] == mes].iloc[0]
    return (
        f"No mês **{mes}**, o **{apartamento}** realizou "
        f"**{int(linha['recargas'])} recargas** e consumiu "
        f"**{format_number(linha['consumo_kwh'])} kWh**.\n\n"
        f"### Composição do gasto\n\n"
        f"- **Energia consumida:** {format_brl(linha['custo_energia'])}\n"
        f"- **Taxa de gestão:** {format_brl(linha['taxa_gestao'])}\n"
        f"- **Permanência adicional na vaga:** {format_brl(linha['custo_ocupacao_excedente'])}\n\n"
        f"**Total estimado: {format_brl(linha['valor_total'])}**"
    )


def responder_status(pergunta, sessoes, momento_local):
    apartamento = extrair_apartamento(pergunta)
    if not apartamento:
        return "Informe o número do apartamento. Por exemplo: qual é o status do apartamento 891?"
    if getattr(momento_local, "tzinfo", None) is None:
        raise ValueError("momento_local precisa ter fuso horário.")

    momento_utc = momento_local.tz_convert("UTC")
    ativas = sessoes[
        (sessoes["apartment"] == apartamento)
        & (sessoes["conn"] <= momento_utc)
        & (sessoes["disc"] > momento_utc)
    ].copy()

    horario = momento_local.strftime("%d/%m/%Y às %H:%M")
    if ativas.empty:
        return f"Em {horario}, o {apartamento} não tinha veículo conectado."

    ativas["entregue_kwh"] = delivered_kwh(ativas, momento_utc)
    respostas = []
    for _, linha in ativas.iterrows():
        if momento_utc < linha["chg_end"]:
            respostas.append(
                f"estava carregando no carregador {linha['station']}, com cerca de "
                f"{format_number(linha['entregue_kwh'])} kWh entregues até esse horário"
            )
        else:
            respostas.append(
                f"estava conectado após o término da carga no carregador {linha['station']}, "
                f"com {format_number(linha['kwh'])} kWh entregues na sessão"
            )
    return f"Em {horario}, o {apartamento} " + "; e também ".join(respostas) + "."


def responder_ranking(identificados, mes, coluna):
    config = RANKINGS[coluna]
    linha = identificados.sort_values(coluna, ascending=False).iloc[0]
    itens = "\n".join(
        f"- **{rotulo}:** {FORMATADORES[tipo](linha[campo])}"
        for rotulo, campo, tipo in config["campos"]
    )
    return (
        f"No mês **{mes}**, o apartamento com o maior {config['descricao']} "
        f"foi o **{linha['apartment']}**.\n\n{itens}"
    )


def responder_condominio(pergunta, faturas):
    meses = sorted(str(m) for m in faturas["billing_month"].unique())
    if not meses:
        return "Não há dados de faturamento disponíveis."

    mes, indisponivel = resolver_mes(pergunta, meses)
    if indisponivel:
        return (
            f"Não encontrei dados consolidados para {indisponivel}. "
            f"Meses disponíveis: {', '.join(meses)}."
        )

    dados = faturas[faturas["billing_month"] == mes]
    identificados = dados[dados["apartment"] != UNIDENTIFIED]
    if identificados.empty:
        return f"Não encontrei apartamentos identificados no mês {mes}."

    ranking = detectar_ranking(normalizar_texto(pergunta))
    if ranking:
        return responder_ranking(identificados, mes, ranking)

    sem_identificacao = dados.loc[dados["apartment"] == UNIDENTIFIED, "consumo_kwh"].sum()
    return (
        f"No mês **{mes}**, o condomínio consumiu "
        f"**{format_number(dados['consumo_kwh'].sum())} kWh**, com gasto total estimado de "
        f"**{format_brl(dados['valor_total'].sum())}**.\n\n"
        f"- **Apartamentos identificados:** {identificados['apartment'].nunique()}\n"
        f"- **Consumo sem identificação:** {format_number(sem_identificacao)} kWh"
    )


def responder_simulador(pergunta, faturas, sessoes, momento_local):
    intencao = classificar_intencao(pergunta)
    if intencao == "faturamento":
        return responder_faturamento(pergunta, faturas)
    if intencao == "status":
        return responder_status(pergunta, sessoes, momento_local)
    if intencao == "condominio":
        return responder_condominio(pergunta, faturas)
    if intencao == "saudacao":
        return f"{GREETING} Experimente: 'Quanto o apartamento 891 gastou em novembro?'"
    return (
        "Não consegui identificar a consulta. Posso responder sobre faturamento, consumo, "
        "gasto total do condomínio ou status de um apartamento. "
        "Inclua o número do apartamento quando necessário."
    )
