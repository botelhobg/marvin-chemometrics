# Marvin Quimiometria 0.1.0

Primeira versão executável: importação, organização, exploração visual e projetos. Implementação independente em Python/Streamlit. Ainda não inclui PCA, PLS, MCR ou outros ajustes de modelos. Não requer MATLAB nem PLS Toolbox.

## Abrir no Windows

1. Extraia a pasta inteira do ZIP.
2. Com Python 3.12 ou 3.13 instalado e o lançador `py` disponível, dê dois cliques em `iniciar_windows.bat`.
3. Na primeira execução, aguarde a criação do ambiente e a instalação das dependências (internet necessária).
4. O Streamlit abre no navegador. Se necessário, acesse o endereço local informado no terminal.
5. Mantenha o terminal aberto; use Ctrl+C para encerrar.

Se a porta padrão estiver ocupada, execute no terminal da pasta:

```bat
.venv\Scripts\python.exe -m streamlit run app.py --server.port 8502
```

O atalho Windows foi incluído, mas os testes desta entrega foram executados em Linux/Python 3.12.

## Instalação manual

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Primeiro teste

Clique em **Carregar exemplo sintético** e abra **Explorar**. Há 36 amostras, 161 comprimentos de onda, uma resposta e dois rótulos de classe; os dados são artificiais. O arquivo `exemplo.marvin.zip` permite testar a reabertura de um projeto em uma sessão nova.

Para dados próprios:

- Importe CSV/TSV/TXT ou Excel XLSX. Cada linha representa uma amostra.
- Em Organizar, defina X, Y, Identificação, Classe, Grupo ou Ignorar.
- Identificadores são preservados como texto (por exemplo, 001). Não são automaticamente reconhecidos como IDs.
- Clique **Aplicar organização**. Alterações ainda não aplicadas não entram nos gráficos/projetos.
- Visualize sinais, distribuições, respostas e estatísticas. Escolher amostras no gráfico não as exclui da análise.
- Em Projeto, baixe o ZIP para guardar tudo e retomar depois.

Em CSV, selecione separador, decimal e codificação. Em Excel, escolha a aba: células numéricas são lidas pelo valor armazenado, não pela aparência. Recalcule fórmulas no Excel e salve antes de importar. Texto numérico com vírgulas pode exigir ajuste do separador decimal em Organizar. Arquivos XLS antigos devem ser salvos como XLSX.

Limites: 20 MB por upload, 64 MB descompactado, 20 mil linhas, 10 mil colunas e 2 milhões de células. Cabeçalhos devem ser únicos; milhares não são interpretados. Campos numéricos vazios permanecem ausentes. Estatísticas descritivas ignoram ausentes; gráficos mantêm lacunas. Nenhum pré-processamento ou imputação é aplicado. A aplicação avisa sobre X constante.

## Rodar online

O código está preparado para hospedagem Streamlit, mas **esta entrega ainda não possui URL pública**.

No Streamlit Community Cloud, publique os arquivos desta pasta em um repositório seu, incluindo `core/`, `requirements.txt` e `.streamlit/config.toml`; selecione `app.py` como arquivo principal e Python 3.12. Não inclua seus dados de pesquisa no repositório. A configuração e a capacidade da conta de hospedagem precisam ser verificadas antes da publicação.

Alternativamente, há um Dockerfile para servidor com Python:

```bash
docker build -t marvin-chemometrics .
docker run --rm -p 8501:8501 marvin-chemometrics
```

Docker não foi executado neste ambiente. Para exposição em produção, configurar HTTPS e controle de acesso na hospedagem. O aplicativo não possui autenticação própria nesta versão. Sessões mantêm dados separados em memória; isolamento funcional foi testado, não constitui auditoria de segurança multiusuário. Recarregar ou encerrar a sessão pode perder o estado; baixe o projeto. Não há persistência automática de uploads no disco nem cache compartilhado de dados.

## Estrutura e validação

`app.py`: interface. `core/data.py`: importação, validação e formato de projetos. `tests/`: testes funcionais e do núcleo. `PLANO.md`: etapas do projeto. `MCR_PLANO.md`: escolha do toolbox do Romà e mapeamento inicial. Os arquivos originais dos toolboxes não são incluídos.

Teste com:

```bash
python -m pip install pytest
python -m pytest -q
```

Referências de infraestrutura consultadas em 19/10/2026:
- https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy
- https://docs.streamlit.io/develop/api-reference/app-testing/st.testing.v1.apptest
- https://docs.streamlit.io/develop/api-reference/caching-and-state/st.session_state

A validação atual verifica software de dados, não equivalência científica com o PLS Toolbox. As versões futuras terão critérios por método. Para MCR, a referência funcional passa a ser o pacote do Romà enviado pelo usuário, conforme MCR_PLANO.md.


## v0.2.0 — regressão PLS

Área **PLS** com PLS1/PLS2, centralização ou autoescalamento, validação cruzada K-fold embaralhada com semente fixa ou GroupKFold por lote, métricas R² / RMSEC / R² CV / RMSECV, scores, VIP e exportação CSV. Os ajustes de validação são treinados separadamente em cada fold. Sem imputação automática. Os modelos ainda não são salvos no arquivo .marvin.zip. Para iniciar instale as dependências de requirements.txt e execute `streamlit run app.py`.


## v0.4 — Classificação (PLS-DA, SIMCA, DD-SIMCA)
- **PLS-DA:** classificação multiclasses por codificação indicadora, regressão PLS e decisão por maior escore; validação cruzada estratificada com reajuste em cada partição. Exporta predições fora dos folds e matriz de confusão.
- **SIMCA:** PCA da classe-alvo e limites individuais aproximados para SD e OD, via momentos qui-quadrado.
- **DD-SIMCA:** preserva o modelo de distância combinada; agora mostra sensibilidade e especificidade exploratórias em CV.
- Para SIMCA/DD-SIMCA, as classes diferentes da classe-alvo não participam da calibração.
- **Limitações:** CV aleatória/estratificada não substitui validação externa e não controla replicatas do mesmo lote. Limites clássicos de SIMCA são aproximados. Parâmetros e resultados do modelo não são persistidos no arquivo de projeto.

## v0.5 — versão de pesquisa multivariada (módulo exploratório)

Nova área **Pré-processamento e PCA**: SNV, MSC com referência média do conjunto,
Savitzky–Golay (suavização e derivadas), correção linear de linha de base por extremos,
PCA centralizada ou autoescalada, scores, loadings, variância explicada,
distâncias Q e T² descritivas, e exportação CSV. A matriz original permanece intacta.

**Limitações importantes:** pré-processamento nesta seção é exploratório e não
é ainda encadeado aos modelos PLS/classificação; não aplicar a uma validação
cruzada após ajustar MSC em todos os dados, para evitar vazamento. Não há
correção automática de delta de eixo nas derivadas; elas são expressas por
intervalo entre colunas, não por nm/cm⁻¹. Q e T² ainda não têm limites
estatísticos implementados. Esta versão não contém MCR-ALS funcional.

Testes numéricos do novo módulo em `python -m unittest discover -s tests -p 'test_exploratory.py'`.
O teste de integração Streamlit requer instalação das dependências de `requirements.txt`.

## Beta de pesquisa: pipeline validado (v0.6)
Na aba **Pipeline validado**, execute PLS1/PLS2 ou PLS-DA usando
linha de base linear, Savitzky-Golay (derivada), SNV e MSC. O ajuste
MSC é exclusivo do conjunto de treinamento em cada fold. A validação por
grupo mantém replicatas identificadas com o mesmo grupo juntas.

**Limitações científicas:** a seleção de hiperparâmetros (janela, derivada,
número de componentes) é manual; portanto resultados de CV usados para
selecionar parâmetros não são estimativas imparciais. Reserve teste externo
independente ou faça CV aninhada para relatar desempenho final. A nova aba
não contempla ainda pipelines combinados para SIMCA/DD-SIMCA; essas rotinas
continuam disponíveis na aba Classificação com ajuste próprio por fold.

## Publicação no Streamlit Community Cloud
1. Envie a pasta `marvin_chemometrics` a um repositório GitHub autorizado,
   com `app.py`, `requirements.txt` e `.streamlit/config.toml` na raiz.
2. No Streamlit Community Cloud, escolha **Create app**, indique repositório,
   branch e entrypoint `app.py`.
3. Se o acesso deve ser restrito, verifique a política de visibilidade e os
   usuários permitidos antes de compartilhar qualquer link.
4. Faça teste com 10–20 sessões simultâneas e dados sintéticos antes de usar
   em aula. Não envie dados sigilosos para serviço sem autorização institucional.
5. Se o deployment falhar por incompatibilidade de versões, confira os logs
   e ajuste as dependências no repositório.

**Status:** pacote preparado para publicação; não foi publicado nem teve URL verificada.
