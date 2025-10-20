# Pré-processador de Vídeo iPPG - Aplicativo Streamlit

## Início Rápido

### Instalação

1. Instale as dependências necessárias:
```bash
pip install -r requirements.txt
```

### Executando a Aplicação

```bash
streamlit run app.py
```

O aplicativo será aberto no seu navegador padrão em `http://localhost:8501`

## Funcionalidades

### 1. Processamento de Vídeo Único

Faça upload de qualquer arquivo de vídeo (MP4, AVI, MOV, MKV) e o app irá:
- Detectar faces usando Haar Cascade
- Extrair sinais RGB da região da testa
- Calcular métricas de qualidade (SNR, brilho, estabilidade)
- Fornecer visualizações interativas
- Permitir exportação de dados para CSV

**Critérios de Qualidade:**
- ✅ **Brilho**: Y_mean entre 80-180
- ✅ **Estabilidade**: Y_mean_std_over_time ≤ 12
- ✅ **Qualidade do Sinal**: SNR ≥ 3 dB

### 2. Processamento em Lote (FaceForensics)

Processe múltiplos vídeos do dataset FaceForensics++:
- Especifique o caminho raiz do dataset
- Selecione tipos de compressão (raw, c23, c40)
- Gere CSV mestre e resumo de qualidade
- Processe vídeos em lote com acompanhamento de progresso

## Interface do Usuário

### Seções Principais

1. **Barra Lateral de Configuração**
   - Selecione o modo de processamento (Único/Lote)
   - Configure parâmetros de processamento

2. **Upload de Vídeo**
   - Arraste e solte ou navegue para arquivos de vídeo
   - Visualize o vídeo carregado
   - Veja informações do arquivo

3. **Painel de Resultados**
   - Cards de métricas de qualidade
   - Avaliação geral de qualidade
   - Gráficos interativos de sinais (RGB, Luminância, Análise de frequência)
   - Visualizador de tabela de dados
   - Opções de download em CSV

### Visualizações

**Gráfico de Canais RGB**
- Visualização de série temporal das intensidades dos canais Vermelho, Verde e Azul
- Ajuda a identificar variações de cor e artefatos

**Gráfico de Luminância**
- Y_mean (brilho) ao longo do tempo
- Y_std (variação espacial) para avaliar estabilidade da iluminação

**Análise do Canal Verde**
- Domínio do tempo: Sinal verde bruto
- Domínio da frequência: PSD com banda cardíaca (0.7-4 Hz) destacada
- Cálculo de SNR para qualidade do sinal

## Exportação de Dados

Baixe dados processados em formato CSV:
- **Dados de Sinal**: Valores RGB e luminância frame a frame
- **Métricas de Qualidade**: Estatísticas resumidas do vídeo

## Detalhes Técnicos

### Pipeline de Processamento de Sinal

1. **Detecção de Face**: Classificador Haar Cascade
2. **Extração de ROI**: Região da testa (15-33% do topo da face, 50% de largura)
3. **Suavização de Box**: EMA com alpha=0.25 para reduzir tremor
4. **Extração de Sinal**: Médias dos canais RGB e luminância (Y)
5. **Avaliação de Qualidade**: Cálculo de SNR usando PSD de Welch

### Estrutura de Arquivos

```
app.py              # Aplicação principal Streamlit
requirements.txt    # Dependências Python
README_APP.md       # Este arquivo
```

## Dicas para Melhores Resultados

1. **Qualidade do Vídeo**
   - Use vídeos com faces claras e bem iluminadas
   - Evite movimento excessivo ou oclusões
   - Duração mínima de 5 segundos para SNR confiável

2. **Detecção de Face**
   - Faces frontais funcionam melhor
   - Garanta iluminação adequada
   - Evite ângulos ou expressões extremas

3. **Qualidade do Sinal**
   - Iluminação estável melhora os resultados
   - Minimize movimento da câmera
   - Taxas de quadros mais altas (≥25 fps) são recomendadas

## Solução de Problemas

**Nenhuma face detectada:**
- Certifique-se de que a face está claramente visível e frontal
- Verifique as condições de iluminação
- Tente diferentes segmentos do vídeo

**SNR baixo:**
- Verifique a estabilidade da iluminação (Y_mean_std_over_time)
- Confira a qualidade de compressão do vídeo
- Garanta duração suficiente do vídeo (≥5s)

**Erros de processamento:**
- Verifique se o formato do arquivo de vídeo é suportado
- Confira se o arquivo não está corrompido
- Garanta memória suficiente para vídeos grandes

## Próximos Passos

Após processar os vídeos:
1. Exporte métricas de qualidade para analisar seleção de vídeos
2. Filtre vídeos de alta qualidade (todos os 3 critérios atendidos)
3. Use master_rgb.csv para engenharia de features de ML
4. Aplique janelamento (10s) para treinamento de modelo
5. Treine modelo para prever/corrigir problemas de iluminação
