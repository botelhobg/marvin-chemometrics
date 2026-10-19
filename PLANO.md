# Plataforma de quimiometria em Python — etapa 0

Data: 19/10/2026. Nome de trabalho: Marvin Chemometrics.

## Decisão de arquitetura

Aplicativo Streamlit para uso no navegador, com núcleo científico Python independente. A mesma aplicação poderá rodar localmente ou em servidor. Esta entrega é o inventário inicial e o plano de implementação; não contém aplicativo executável e não publica um serviço.

## Evidência obtida

Inspeção estática, sem executar o instalador. Pacote NSIS-3, Deflate, 80.664.037 bytes. O catálogo principal identifica versão 8.9.2 (23162), de 14/05/2021. O arquivo inventario.json registra hash SHA-256, caminhos do pacote e entradas dos dez catálogos Contents.m extraídos.

Foram listadas 3.708 entradas: 1.791 arquivos .m, 7 .p, 1.147 .html, 68 .mat e 4 .pdf, além de imagens, bibliotecas e outros recursos. Há 783 entradas nos catálogos, com 734 nomes distintos. Esses números NÃO representam 734 algoritmos: incluem utilitários, demonstrações, dados e duplicatas entre catálogos. Extensão .m não comprova que um arquivo contém uma implementação completa.

O catálogo documenta PCA/MLPCA, PLS/PCR/MLR/CLS, SIMPLS/NIPALS, redes neurais, SVM e árvores; PLS-DA/SIMCA/KNN/regressão logística; MCR/EFA; PARAFAC/PARAFAC2/Tucker/N-PLS; seleção genética/iPLS; transferência de calibração; MSPC; planejamento experimental; pré-processamento e diversos importadores instrumentais.

Limites: foram lidos os catálogos de ajuda, não todas as páginas de documentação, opções ou implementações. A presença no catálogo não comprova funcionamento. Nenhum cálculo foi comparado com MATLAB. Os arquivos originais não são redistribuídos nesta entrega. O cabeçalho do catálogo contém restrição à tradução/conversão dos M-files; o projeto será implementação independente dos métodos, apoiada na literatura e em dependências compatíveis.

## Organização técnica proposta

- core/data: matriz X, respostas Y, classes, identificadores, eixos, unidades, grupos e máscaras.
- core/preprocessing: transformadores com fit/transform, parâmetros e ordem registrados.
- core/models: ajuste, previsão, diagnósticos e serialização versionada.
- core/validation: particionamento, validação interna/externa, grupos e permutação.
- core/io: importadores, exportadores e projetos.
- ui: páginas Streamlit, formulários, estado da sessão e gráficos interativos.
- tests: dados sintéticos determinísticos, referências analíticas e comparações externas.

Cada usuário terá estado próprio. Não usar cache global para guardar dados privados ou modelos de usuários. No primeiro ciclo, projetos serão exportados e importados pelo usuário; persistência em servidor será uma função explícita, com autenticação e isolamento antes de armazenar dados de pesquisa.

Formato inicial proposto: ZIP com manifesto JSON e matrizes em formato sem execução de código. Não aceitar pickle arbitrário. Registrar versão do aplicativo, dependências, semente, divisão das amostras, exclusões e parâmetros. Validar tamanho, dimensões e nomes de caminhos na importação.

## Etapas e aceite

| Versão | Escopo | Critério de aceite |
|---|---|---|
| 0.0 — atual | Inventário estático e arquitetura | Rastreabilidade ao instalador, contagens e limites explícitos |
| 0.1 | Importação CSV/XLSX, atribuição de X/Y/classes/IDs, visualização e projeto | Reabrir preservando valores, eixos e metadados; validar entradas inválidas; duas sessões sem mistura de dados |
| 0.2 | Centrar, autoescalar, SNV, MSC, Savitzky–Golay, PCA | Reconstrução e variância verificadas; transformações reproduzíveis; constantes e dimensões insuficientes tratadas |
| 0.3 | PLS/PCR/MLR, validação e previsão | Partições reproduzíveis; parâmetros aprendidos somente no treino; RMSE/bias/R² auditáveis |
| 0.4 | PLS-DA, SIMCA, KNN, SVM e classificação | Métricas por classe, limiares treinados sem conjunto externo, classes ausentes tratadas |
| 0.5 | VIP/iPLS/seleção genética, diagnósticos e permutação | Seleção dentro da validação; estimativa final sem viés de seleção, com CV aninhada quando pertinente |
| 0.6 | MCR/EFA e métodos multiway | Restrições, convergência, múltiplas inicializações e indeterminações documentadas |
| 0.7 | Multiblocos, transferência, MSPC, métodos restantes | Aceite específico por família e referência numérica |
| 0.8 | Importadores especializados, DOE e funções auxiliares | Arquivos de referência por formato; unidades e metadados preservados |
| 1.0 | Auditoria de cobertura e operação online | Lista de equivalências/exceções explícita, recuperação de projetos e testes de carga |

A ordem pode ser ajustada por necessidade de ensino/pesquisa. Cobertura total é o objetivo do programa, não uma promessa de equivalência exata em todos os formatos proprietários ou integrações externas. Utilitários específicos de MATLAB serão substituídos pelo comportamento equivalente no ambiente Python.

## Validação científica

Usar problemas sintéticos com solução conhecida, identidades algébricas e implementações de referência independentes. Para equivalência com PLS Toolbox, obter resultados de uma instalação licenciada sob configurações idênticas. Definir tolerâncias absolutas e relativas por saída após considerar condicionamento e precisão; não usar um único limite universal.

Na PCA comparar subespaços, reconstrução e variância, considerando sinais e rotações de componentes degenerados. Em PLS fixar algoritmo, escala, centralização e número de componentes. Em SNV/autoescala declarar divisor da variância. Em derivadas declarar espaçamento do eixo e tratamento das bordas. Em classificação separar respostas contínuas, probabilidades calibradas e decisão.

Pré-processamento que aprende parâmetros, seleção de variáveis, otimização e ajuste de limiar devem ocorrer exclusivamente dentro do treino apropriado. Réplicas do mesmo indivíduo/lote devem poder permanecer no mesmo grupo. O conjunto externo não escolhe hiperparâmetros.

## Operação online

Streamlit atende à interface no navegador e permite publicação a partir de repositório. Referências consultadas em 19/10/2026:
- https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy
- https://docs.streamlit.io/get-started/fundamentals/advanced-concepts

A escolha de hospedagem dependerá de memória, concorrência e privacidade. Validação cruzada extensa, algoritmos genéticos e tensores grandes terão limites de recursos; filas de tarefas serão acrescentadas se a carga justificar. Hospedagem gratuita não é requisito de arquitetura nem garantia de capacidade.

Publicação é uma etapa separada: nenhum endereço foi criado nesta entrega. Primeiro marco executável: versão 0.1 em Streamlit, com exemplo sintético próprio e formato de projeto definido.

## Atualização de 19/10/2026 — versão 0.1 e MCR

A primeira versão de dados e exploração está implementada em Streamlit. O módulo MCR passa a seguir o toolbox enviado pelo usuário: `mcr_toolbox2_updated_070721.rar`. Ver MCR_PLANO.md para escopo e critérios; não usar a implementação do PLS Toolbox como referência principal do MCR. A v0.1 não inclui cálculos MCR.
