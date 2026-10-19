# MCR: referência específica definida pelo usuário

Em 19/10/2026, o usuário definiu o arquivo `mcr_toolbox2_updated_070721.rar`, identificado por ele como o toolbox do Romà, como referência do módulo MCR. Essa decisão substitui o PLS Toolbox como referência principal dessa família. O módulo ainda não está implementado na versão 0.1.

Inspeção: arquivo RAR com 146 arquivos e 3 diretórios, incluindo interfaces MATLAB e rotinas científicas; 93 arquivos de texto .m foram extraídos e foram examinados o fluxo principal e trechos das rotinas. Veja nomes, tamanhos e hash do arquivo em MCR_REFERENCIA.json. Não foram executadas rotinas MATLAB, e nenhuma equivalência foi verificada.

## Mapeamento inicial

| Função desejada | Evidência no pacote | Fase proposta |
|---|---|---|
| Fatoração D ≈ C Sᵀ por ALS | alsOptimization.m | MCR-1 |
| Inicialização por perfis fornecidos e variáveis puras | iniesta no otimizador; pure.m | MCR-1 |
| Estimativa de posto e EFA | svdgui.m; efa.m; efard.m | MCR-1 |
| Não negatividade em C e S | blocos do otimizador; nnls.m; fnnls.m | MCR-1 |
| Unimodalidade | unimod.m e aplicação no otimizador | MCR-2 |
| Fechamento/balanço | closure.m; closurels.m; blocos em C e S | MCR-2 |
| Modelos multi-experimento e restrições por modo | columnMultiConstraints.m; rowMultiConstraints.m; nexp no otimizador | MCR-2 |
| Restrições de correlação e cinética | correlCons.m; opt_kinglob.m; blocos no otimizador | MCR-3 |
| Trilinearidade e outras estruturas multiway | trilin.m; quadril.m; tuck2.m | MCR-3 |
| Ambiguidade de soluções | pasta mcrbandsC | MCR-4; estudar dependências e escopo antes de implementar |
| LOF, variância explicada, melhor iteração e convergência | saídas plot_lof, plot_R2, optim_niter e rotinas do otimizador | desde MCR-1 |

As linhas acima documentam presença e uso aparente, não a lista completa de opções ou garantias de funcionamento. Antes de programar cada recurso, especificar orientação das matrizes, escolha das componentes, tolerâncias, ordem das restrições e tratamento de falhas.

## Critérios de comparação

Usar os mesmos D, número de componentes e estimativas iniciais, com cada restrição isolada antes de combiná-las. Comparar reconstrução, resíduos e LOF; alinhar permutação e escala de C/S antes de comparar perfis. Separar resultados sobre D experimental e aproximação PCA. Não confundir variância explicada não centrada do MCR com R² de regressão. Guardar inicialização, histórico de convergência, limite de iterações, melhor solução e motivo de parada.

## Implementação e atribuição

O pacote fornecido não apresentou um arquivo autônomo de licença entre os nomes inventariados; isso não comprova ausência de condições de uso. Algumas rotinas auxiliares contêm avisos MathWorks. Nesta versão nenhuma rotina do pacote foi copiada ou traduzida para Python, e o pacote original não é redistribuído. Implementar métodos de forma independente com atribuição científica; verificar licença/autorização antes de incorporar diretamente qualquer código. A identificação bibliográfica exata da versão ainda deve ser confirmada.
