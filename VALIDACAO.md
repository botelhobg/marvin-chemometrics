# Validação da versão 0.1.0 — 19/10/2026

Resultado: **11 testes aprovados** (`python -m pytest -q`, Linux/Python 3.12).

Cobertura: CSV brasileiro com IDs 001/002 e dados ausentes; XLSX numérico e seleção de aba; rejeição de cabeçalhos duplicados/vazios e linhas irregulares; rejeição de textos/infinitos em X/Y e identificadores duplicados; exclusões; salvamento/reabertura preservando tabela, configurações, matrizes, eixos e grupos; rejeição de pacote com entrada extra e schema incompatível; navegação das quatro áreas com exemplo; isolamento entre duas sessões simuladas; upload CSV e reabertura de projeto pela interface simulada, seguidos de visualização.

Interface verificada com Streamlit AppTest. Não houve inspeção visual em navegador real, teste nativo Windows, build Docker ou publicação em servidor externo. Não há cálculos de PCA/PLS/MCR nesta versão, logo não há comparação numérica com os toolboxes.
