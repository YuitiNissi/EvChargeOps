# ⚡ EV ChargeOps

Sistema de gestão operacional de recarga de veículos elétricos para condomínios, desenvolvido para o **Enterprise Challenge 2026 (GoodWe + FIAP)**.

O projeto transforma sessões de recarga em dados estruturados e inteligência acionável, atacando três problemas do dia a dia de um condomínio:

| Problema | Como o EV ChargeOps responde |
|---|---|
| **Rateio de custos** | Cobrança automática por apartamento, por faixa de preço da energia |
| **Sobrecarga da rede** | Simulação de políticas de controle de demanda sob um limite de potência |
| **Experiência do usuário** | Painel de monitoramento, alertas de manutenção e um assistente virtual para consultas |

Esta versão é uma **demonstração em Streamlit** construída sobre o dataset público [ACN-Data](https://ev.caltech.edu/dataset) (Caltech).

---

## O que o dashboard faz

O aplicativo tem sete abas:

| Aba | O que entrega |
|---|---|
| **Resumo do condomínio** | Indicadores gerais, horários de chegada, consumo diário, consumo por faixa de preço e qualidade dos dados |
| **Gastos por apartamento** | Fatura mensal por apartamento (energia + taxa de gestão + permanência na vaga), gráfico dos maiores gastos e download em CSV |
| **Controle de demanda** | Compara três políticas de distribuição de potência sob um limite escolhido e estima a economia de demanda |
| **Previsão de consumo** | Previsão horária para 24 h com faixa P10–P90 e download em CSV |
| **Manutenção e alertas** | Sinais para inspeção por regras e por detecção de anomalia, com filtro de severidade e download em CSV |
| **Monitoramento** | Replay do histórico: escolha uma data e um horário e veja quais carregadores estavam em uso |
| **Assistente virtual** | Chatbot por regras que responde sobre gastos, consumo e status de recarga |

## Como executar

Requer **Python 3.10 ou superior** e **Streamlit 1.51 ou superior**.

```bash
pip install -r requirements.txt
streamlit run app.py
```

Mantenha `acndata_sessions.json` na mesma pasta do `app.py`. Também é possível enviar outro arquivo ACN-Data pela barra lateral.

Testado com Python 3.12, Streamlit 1.65, Plotly 7.1, pandas 3.0 e scikit-learn 1.8.

## Estrutura do repositório

```
.
├── app.py                  # dashboard Streamlit (dados, rateio, simulação, previsão, alertas, UI)
├── chatbot_simulador.py    # assistente virtual por regras (usado pela aba "Assistente virtual")
├── ev_common.py            # constantes e funções compartilhadas (formatação pt-BR, rótulo de apartamento, energia entregue)
├── tests/
│   └── test_chatbot.py     # testes automatizados do assistente
├── requirements.txt        # dependências
├── acndata_sessions.json   # dataset de exemplo (ACN-Data, site Caltech)
├── .gitignore
└── README.md
```

## Dados

O **ACN-Data** (Adaptive Charging Network) registra uma linha por conexão de veículo, com horários de conexão, desconexão e fim de carga (em UTC), kWh entregue, carregador, `userID` e, quando o motorista informou, o pedido de recarga (`userInputs`).

**Arquivo de exemplo.** O `acndata_sessions.json` incluído é do site `caltech` e está **truncado no final**. A função `read_sessions_json` percorre o arquivo e recupera todas as sessões completas em vez de falhar, e o app exibe um aviso quando isso acontece. Foram recuperadas **2.799 sessões**, de 07/10 a 20/11/2018, em **54 carregadores** e **147 usuários**. Se você tiver o arquivo completo, basta substituí-lo.

**Cada `userID` é tratado como um apartamento** (`APTO 891`). Sessões sem `userID` ficam em `NÃO IDENTIFICADO`. Em produção, essa associação viria do cadastro do condomínio.

O dataset **não traz** código de falha nem potência instantânea. Por isso os alertas são *inferidos* e a potência do monitoramento é *estimada*.

## Como cada módulo funciona

### Preparação dos dados (`load_sessions`)
Converte os horários para o fuso escolhido, remove duplicatas e calcula, por sessão: duração da conexão, janela de carga (até o fim de carga informado), tempo ocioso após o fim da carga e potência média. Também marca problemas de qualidade: sem fim de carga, janela inválida, potência acima do esperado, conexão acima de 24 h e conexão longa com quase nenhuma energia.

### Rateio por apartamento
1. O kWh de cada sessão é distribuído hora a hora ao longo da janela de carga (`spread_energy`).
2. Cada hora é precificada pela faixa tarifária. Em dias úteis, 18h–20h59 é a faixa mais cara e 17h e 21h são intermediárias. O restante, e todo o fim de semana, é econômico.
3. A fatura mensal soma **energia + taxa de gestão (R$/kWh) + permanência adicional** (horas ociosas acima da tolerância × R$/h).
4. O mês da cobrança é o mês da desconexão.

### Controle de demanda (`allocate` e `simulate`)
A simulação avança em passos de 15 minutos. A cada passo, as sessões conectadas pedem potência e a política decide quanto cada uma recebe sem ultrapassar o limite do condomínio:

| Política | Critério |
|---|---|
| **Divisão igual** | Reparte a potência igualmente (quem pede menos é atendido por inteiro) |
| **Ordem de chegada** | Prioriza quem conectou primeiro |
| **Menor folga primeiro (LLF)** | Prioriza quem tem menos tempo sobrando até a saída, dado o que ainda falta carregar |

O limite é definido como porcentagem do pico sem controle. As métricas são a energia atendida, as recargas que ficaram abaixo de 95% do kWh original e o pico. A economia indicativa é `(pico sem controle − limite) × custo de demanda`.

### Previsão de consumo (`forecast_next`)
Gradient boosting (scikit-learn) sobre a série de energia por hora. As variáveis são hora do dia, dia da semana, fim de semana, consumo 24 h, 48 h e 168 h antes, média da mesma hora nos 7 dias anteriores e média das últimas 168 h. Três modelos são treinados: um com perda Poisson para a previsão central e dois quantílicos (10% e 90%) para a faixa. A série de treino termina na última hora completa observada, e a previsão cobre as 24 h seguintes (no dataset de exemplo, de 20/11/2018 11:00 a 21/11/2018 10:00).

### Alertas (`build_alerts`)
- **Regras:** sem sinal de fim de carga, janela de carga inválida, potência média acima do limite, conexão acima de 24 h (de "informativa" a "atenção" acima de 72 h) e conexão longa com menos de 1 kWh.
- **Anomalia:** Isolation Forest sobre energia, duração, tempo de carga, ociosidade, potência média e hora de chegada. Os 2% de sessões mais atípicas viram sinais informativos (por construção, sempre há esse percentual de sinais).

Os alertas são **sinais para inspeção, não falhas confirmadas**.

### Monitoramento
Para a data e o horário escolhidos, lista as sessões ativas, classifica cada uma como *Carregando* ou *Conectado após o término* e estima a demanda total. A potência por sessão é a média da janela de carga, limitada à potência nominal. A energia exibida é a entregue **até o horário escolhido**, não o total da sessão.

### Assistente virtual (`chatbot_simulador.py`)
Funciona por regras, sem API externa. Normaliza o texto, identifica a intenção (faturamento, status, visão do condomínio ou saudação), extrai o apartamento (`apartamento 891`, `apto 891`) e o mês (nome ou `AAAA-MM`) e responde consultando as tabelas calculadas pelo dashboard. Se o mês pedido não tem dados, o assistente avisa e lista os meses disponíveis, em vez de responder outro mês. Nas consultas de status, informa os kWh entregues até o horário de referência. Consultas de status usam a data e o horário de referência escolhidos no painel, porque o dataset é um histórico.

Os textos e os formatadores (R$, kWh, datas) são compartilhados com o dashboard por `ev_common.py`.

Exemplos de perguntas:

- Quanto o apartamento 891 gastou em novembro?
- Qual foi o consumo do apartamento 1111?
- Qual é o status do apartamento 891?
- Qual foi o gasto total do condomínio em novembro?
- Qual apartamento teve o maior gasto?

## Testes

```bash
pip install pytest
pytest
```

Os 21 testes cobrem o assistente (interpretação de mês, classificação de intenção, rankings, status) e o cálculo de energia entregue.

## Parâmetros configuráveis (barra lateral)

| Parâmetro | Padrão | Observação |
|---|---|---|
| Fuso do site | `America/Los_Angeles` | Para dados do Brasil, use `America/Sao_Paulo` |
| Potência nominal | 6,6 kW | Deduzida dos dados; confirmar com a GoodWe |
| Fator de alerta de potência | 1,1 | Alerta se a média passar de 1,1 × a nominal |
| Tarifa econômica / intermediária / mais cara | R$ 0,60 / 0,95 / 1,60 por kWh | **Ilustrativas** |
| Taxa de gestão | R$ 0,08 por kWh | |
| Tolerância após o fim da carga | 0,5 h | |
| Permanência adicional | R$ 2,00 por h | Cobrada acima da tolerância |
| Custo de demanda | R$ 45 por kW/mês | **Ilustrativo** |

## Resultados de referência

Com o dataset de exemplo e os parâmetros padrão (valores dependem das premissas ilustrativas):

| Indicador | Valor |
|---|---|
| Recargas / consumo total | 2.799 recargas, 25.579 kWh |
| Gasto estimado (rateio) | R$ 36.284,84 |
| Consumo sem usuário identificado | 66,7% em outubro e 3,9% em novembro (46,5% no total) |
| Pico sem controle | 211,2 kW |
| Limite a 30% do pico (63,4 kW) | LLF entrega **99,84%** da energia, contra 98,00% (ordem de chegada) e 97,76% (divisão igual) |
| Limite a 20% do pico (42,2 kW) | LLF 94,17%; as demais ficam perto de 89% |
| Redução indicativa de demanda a 30% | R$ 6.653,43 por mês |
| Alertas | 197 sinais (116 de atenção e 81 informativos) |

**Leitura principal:** em outubro, dois terços da energia não tinham dono. Para ratear em condomínio, a **identificação do usuário na conexão (RFID ou app) é requisito do produto**.

## Limitações

- **Período curto.** São 6 semanas de dados; a previsão e as conclusões de sazonalidade são limitadas.
- **Previsão sem validação no app.** Não há backtest, e a faixa P10–P90 vem de modelos quantílicos treinados nos próprios dados, sem calibração.
- **Simulador com visão perfeita.** Ele conhece o kWh e a hora de saída reais de cada sessão. Em produção, esses dados precisam ser estimados, e o pedido do motorista sozinho não é confiável.
- **Sem rede elétrica real.** A simulação ignora perdas e limites por fase.
- **Potência estimada.** A telemetria instantânea não existe no dataset.
- **Alertas sem rótulo.** Devem ser calibrados com logs reais dos carregadores.
- **Assistente por regras.** Não entende sinônimos livres nem guarda contexto entre perguntas.
- **Tarifas e demanda ilustrativas.** Substitua pelas da concessionária do condomínio.

## Próximos passos

1. Trocar as tarifas ilustrativas pelas reais e o fuso por `America/Sao_Paulo`.
2. Separar a lógica (rateio, controle, previsão, alertas) da interface e expô-la como API.
3. Substituir a potência estimada por telemetria real dos carregadores.
4. Validar os alertas com logs de falha reais.
5. Validar a previsão com backtest em um histórico mais longo.

## Créditos

- Dados: [ACN-Data](https://ev.caltech.edu/dataset), Caltech.
- Projeto desenvolvido para o Enterprise Challenge 2026 (GoodWe + FIAP).
