<a name="portuguese"></a>
<h1 align="center">📌 Bem-vindo ao A-GRASPQ-FS Tool! 📌</h1>

<h4 align="left">
✔️ O <strong>A-GRASPQ-FS Tool</strong> (<em>Adaptive GRASPQ Feature Selection</em>) é uma ferramenta de linha de comando que implementa a metaheurística adaptativa proposta no artigo <em>"Adaptive Feature Selection with Self-Tuning Subset Size for Intrusion Detection"</em> (ISCC 2026). Diferente da versão anterior (GRASPQ-FS), esta versão <strong>elimina a necessidade de definir manualmente o número de features (k)</strong>, aprendendo automaticamente o tamanho ótimo do subconjunto por meio de um mecanismo de seleção probabilística com recalibração adaptativa.
</h4>

<h2>📁 Estrutura do Repositório</h2>
<pre><code>.
├── data/                     # Pasta para os datasets de entrada (não incluídos — veja nota abaixo)
├── results/                  # Logs e gráficos gerados durante a execução
├── Dockerfile                # Imagem Docker para execução sem dependências locais
├── main.py                   # Script principal: fases construtiva e de busca local adaptativa
├── utils.py                  # Funções auxiliares: carregamento, pré-processamento, avaliação e plots
├── priority_queue.py         # Implementação da fila de prioridade máxima (MaxPriorityQueue)
├── baselines.py              # Script para execução de baselines (SFS, RFE, RFECV)
├── requirements.txt          # Dependências Python
└── README.md                 # Este arquivo
</code></pre>

<h2>📋 Sumário</h2>
<ol>
  <li><a href="#diferencial">Diferencial: A-GRASPQ-FS vs GRASPQ-FS</a></li>
  <li><a href="#ambiente">Ambiente de Teste e Desenvolvimento</a></li>
  <li><a href="#requisitos">Requisitos</a></li>
  <li><a href="#datasets">Datasets e Direitos Autorais</a></li>
  <li><a href="#execucao">Como Executar</a></li>
  <li><a href="#parametros">Parâmetros Disponíveis</a></li>
  <li><a href="#baselines">Executando os Baselines</a></li>
  <li><a href="#saidas">Saídas Geradas</a></li>
  <li><a href="#english-version">English version</a></li>
</ol>

<h2 id="diferencial">🧠 Diferencial: A-GRASPQ-FS vs GRASPQ-FS</h2>
<table border="1">
<tr><th>Característica</th><th>GRASPQ-FS (anterior)</th><th>A-GRASPQ-FS (atual)</th></tr>
<tr><td>Tamanho do subconjunto</td><td>Fixo — definido manualmente pelo usuário (<code>-is k</code>)</td><td>Adaptativo — aprendido automaticamente no intervalo [kmin, kmax]</td></tr>
<tr><td>Fase construtiva</td><td>GRASP padrão com tamanho fixo</td><td>Reactive Construction com seleção probabilística por roleta e recalibração de pesos</td></tr>
<tr><td>Busca local</td><td>SWAP apenas (mantém tamanho fixo)</td><td>Shuffled Neighborhood com ADD, REMOVE e SWAP — ajusta tamanho dinamicamente</td></tr>
<tr><td>Dependência de modelo</td><td>Model-agnostic</td><td>Model-agnostic (mantido)</td></tr>
<tr><td>Necessidade de grid search manual</td><td>Sim, para encontrar o melhor k</td><td>Não — converge ao k ótimo em uma única execução</td></tr>
</table>

<h2 id="ambiente">🖱️ Ambiente de Teste</h2>
<table border="1">
<tr><th>Configuração</th><th>Máquina</th></tr>
<tr><td>Sistema Operacional</td><td>Windows 11</td></tr>
<tr><td>Processador</td><td>Intel(R) Core(TM) i7-13650HX @ 2.60GHz</td></tr>
<tr><td>Memória RAM</td><td>16 GB (15,7 GB utilizável)</td></tr>
<tr><td>Versão do Python</td><td>3.10.0</td></tr>
</table>

<h3>⚙️ Ambiente de Desenvolvimento</h3>
<table border="1">
<tr><th>Ferramenta</th><th>Versão</th></tr>
<tr><td>Python</td><td>3.10.0</td></tr>
<tr><td>Editor</td><td>VS Code / PyCharm</td></tr>
<tr><td>Terminal</td><td>PowerShell, CMD ou Bash</td></tr>
</table>

<h2 id="requisitos">📝 Requisitos</h2>
<p>O projeto utiliza Python 3 e as seguintes bibliotecas:</p>
<ul>
  <li>numpy ≥ 1.21</li>
  <li>pandas ≥ 1.3</li>
  <li>matplotlib ≥ 3.4</li>
  <li>scikit-learn ≥ 1.0</li>
  <li>xgboost ≥ 1.5</li>
</ul>

<h2 id="datasets">📂 Datasets e Direitos Autorais</h2>

<p>⚠️ <strong>Os datasets não estão incluídos neste repositório</strong> por questões de direitos autorais e licenciamento. Cada usuário deve obter os datasets diretamente nas fontes oficiais e colocá-los na pasta <code>data/</code>.</p>

<p>O projeto inclui suporte nativo (pré-processamento automático) para os seguintes datasets:</p>

<table border="1">
<tr><th>Identificador (<code>-d</code>)</th><th>Dataset</th><th>Arquivo esperado em <code>data/</code></th></tr>
<tr><td><code>ereninho</code></td><td>ERENO — Smart Grid / IEC-61850 (dataset padrão para testes iniciais)</td><td><code>hibrid_dataset_GOOSE_train.csv</code></td></tr>
<tr><td><code>batadal</code></td><td>BATADAL — Infraestrutura de água</td><td><code>BATADAL_dataset03.csv</code>, <code>BATADAL_dataset04.csv</code></td></tr>
<tr><td><code>wadi</code></td><td>WADI — Sistema de distribuição de água</td><td><code>WADI.csv</code></td></tr>
<tr><td><code>wustl</code></td><td>WUSTL-EHMS-2020 — Healthcare / IoMT</td><td><code>wustl-ehms-2020.csv</code></td></tr>
<tr><td><code>cic-iot</code></td><td>CIC-IoT 2023 — Tráfego IoT com ataques modernos</td><td>(configurar em <code>utils.py</code>)</td></tr>
</table>

<p>Para adicionar um novo dataset, basta implementar sua lógica de carregamento e pré-processamento na função <code>load_data()</code> em <code>utils.py</code>, seguindo o padrão dos datasets existentes.</p>

<h2 id="execucao">🚀 Como Executar</h2>

<h4>▶️ Opção 1: Execução Local (Recomendado para Desenvolvimento)</h4>
<ol>
  <li>
    <strong>Clone este repositório e entre na pasta do projeto:</strong>
    <pre><code>git clone https://github.com/this-repository.git
cd this-repository</code></pre>
  </li>
  <li>
    <strong>Crie e ative um ambiente virtual (recomendado):</strong>
    <pre><code>python -m venv venv
venv\Scripts\activate       # no Windows
source venv/bin/activate    # no Unix/Mac</code></pre>
  </li>
  <li>
    <strong>Instale as dependências:</strong>
    <pre><code>pip install -r requirements.txt</code></pre>
  </li>
  <li>
    <strong>Execute com a configuração do artigo (recomendado para reprodutibilidade):</strong>
    <pre><code>python main.py -d ereninho -a nb -rcl 10 -is 5 -pq 30 -lc 1000 -cc 100</code></pre>
    <p>ℹ️ O dataset <code>hibrid_dataset_GOOSE_train.csv</code> (ERENO) já está incluído na pasta <code>data/</code> para testes imediatos. Para outros datasets, consulte a seção <a href="#datasets">Datasets e Direitos Autorais</a>.</p>
  </li>
</ol>

<h4>⚡ Execução Rápida (para testes iniciais com menor custo computacional)</h4>
<pre><code>python main.py -d ereninho -a nb -rcl 10 -is 5 -pq 10 -lc 50 -cc 50</code></pre>

<p><strong>Outros exemplos de uso:</strong></p>
<pre><code># Usando KNN com configuração do artigo
python main.py -d ereninho -a knn -rcl 10 -is 5 -pq 30 -lc 1000 -cc 100

# Usando Decision Tree com execução rápida
python main.py -d ereninho -a dt -rcl 10 -is 5 -pq 10 -lc 50 -cc 50

# Aliases alternativos dos parâmetros também funcionam:
python main.py --dataset ereninho --algorithm nb --rcl_size 10 --init_sol 5 --pq_size 30 --ls 1000 --const 100
</code></pre>

<h4>🐳 Opção 2: Execução via Docker (Sem Dependências Locais)</h4>
<ol>
  <li>
    <strong>Construa a imagem Docker:</strong>
    <pre><code>docker build -t a-graspq-fs .</code></pre>
  </li>
  <li>
    <strong>Teste rápido (dataset ERENO já incluso no repositório):</strong>
    <pre><code>docker run --rm -v $(pwd)/results:/app/results a-graspq-fs -d ereninho -a nb -rcl 10 -is 5 -pq 10 -lc 50 -cc 50</code></pre>
  </li>
  <li>
    <strong>Execução completa com dataset externo:</strong>
    <pre><code>docker run --rm -v $(pwd)/data:/app/data -v $(pwd)/results:/app/results a-graspq-fs -d ereninho -a nb -rcl 10 -is 1 -max_is 33 -pq 30 -lc 1000 -cc 100</code></pre>
  </li>
</ol>
<p>ℹ️ O <code>-v $(pwd)/results:/app/results</code> é recomendado para que os arquivos gerados (logs, gráficos) fiquem acessíveis na sua máquina após o container encerrar. O <code>-v $(pwd)/data:/app/data</code> só é necessário ao usar datasets externos não incluídos no repositório.</p>

<h2 id="parametros">🧾 Parâmetros Disponíveis</h2>

<table border="1">
<tr><th>Flag</th><th>Aliases</th><th>Descrição</th><th>Padrão</th></tr>
<tr><td><code>-d</code></td><td><code>--dataset</code></td><td>Dataset a utilizar (<code>ereninho</code>, <code>batadal</code>, <code>wadi</code>, <code>wustl</code>, <code>cic-iot</code>)</td><td><code>ereninho</code></td></tr>
<tr><td><code>-a</code></td><td><code>--algorithm</code>, <code>--alg</code></td><td>Classificador (<code>nb</code>, <code>dt</code>, <code>knn</code>, <code>rf</code>, <code>svm</code>, <code>linear_svc</code>, <code>sgd</code>, <code>xgboost</code>)</td><td><code>nb</code></td></tr>
<tr><td><code>-rcl</code></td><td><code>--rcl_size</code>, <code>--rcl</code></td><td>Tamanho da Lista Restrita de Candidatos (RCL)</td><td><code>10</code></td></tr>
<tr><td><code>-is</code></td><td><code>--initial_solution</code>, <code>--init_sol</code></td><td>Tamanho mínimo da solução inicial (kmin)</td><td><code>5</code></td></tr>
<tr><td><code>-max_is</code></td><td><code>--max_initial_solution</code></td><td>Tamanho máximo da solução inicial (kmax)</td><td><code>30</code></td></tr>
<tr><td><code>-pq</code></td><td><code>--priority_queue</code>, <code>--pq_size</code></td><td>Capacidade da fila de prioridade (SPQ)</td><td><code>10</code></td></tr>
<tr><td><code>-cc</code></td><td><code>--constructive_iterations</code>, <code>--const</code></td><td>Número de iterações da fase construtiva (Iconst)</td><td><code>100</code></td></tr>
<tr><td><code>-lc</code></td><td><code>--local_iterations</code>, <code>--ls</code></td><td>Número máximo de iterações da busca local por solução (Ilocal)</td><td><code>50</code></td></tr>
<tr><td><code>-alpha</code></td><td><code>--alpha</code></td><td>Fator de aleatoriedade da construção semi-gulosa (0.0 a 1.0)</td><td><code>0.3</code></td></tr>
<tr><td><code>-ess</code></td><td><code>--evaluation_sample_size</code></td><td>Fração dos dados de treino usada na avaliação rápida (0.0 a 1.0)</td><td><code>0.1</code></td></tr>
</table>

<h3>🔬 Configuração do Artigo (Reprodutibilidade)</h3>
<p>Para reproduzir os experimentos do artigo (ISCC 2026), use:</p>
<pre><code>python main.py -d ereninho -a nb   -rcl 10 -is 1 -max_is 33 -pq 30 -lc 1000 -cc 100
python main.py -d ereninho -a knn  -rcl 10 -is 1 -max_is 33 -pq 30 -lc 1000 -cc 100
python main.py -d ereninho -a dt   -rcl 10 -is 1 -max_is 33 -pq 30 -lc 1000 -cc 100
python main.py -d ereninho -a linear_svc -rcl 10 -is 1 -max_is 33 -pq 30 -lc 1000 -cc 100
</code></pre>

<h2 id="baselines">📊 Executando os Baselines</h2>
<p>O script <code>baselines.py</code> implementa as linhas de base usadas no artigo (SFS Forward/Backward, RFE e RFECV) para comparação direta com o A-GRASPQ-FS.</p>

<pre><code># SFS Forward com Naive Bayes, selecionando 10 features
python baselines.py -d ereninho -m sfs_forward -a nb -nf 10

# RFE com Decision Tree, selecionando 5 features
python baselines.py -d ereninho -m rfe -a dt -nf 5

# RFECV com Random Forest (determina o número automaticamente)
python baselines.py -d ereninho -m rfecv -a rf
</code></pre>

<p><strong>Parâmetros do baselines.py:</strong></p>
<ul>
  <li><code>-d</code>, <code>--dataset</code>: Dataset a utilizar</li>
  <li><code>-m</code>, <code>--method</code>: Método (<code>sfs_forward</code>, <code>sfs_backward</code>, <code>rfe</code>, <code>rfecv</code>)</li>
  <li><code>-a</code>, <code>--algorithm</code>: Classificador (<code>knn</code>, <code>nb</code>, <code>dt</code>, <code>linear_svc</code>, <code>rf</code>)</li>
  <li><code>-nf</code>, <code>--n_features</code>: Número de features a selecionar (obrigatório para RFE; opcional para SFS; ignorado no RFECV)</li>
</ul>

<h2 id="saidas">📤 Saídas Geradas</h2>
<p>Após a execução, os seguintes arquivos são gerados automaticamente na pasta <code>results/</code> (criada pelo próprio script se não existir):</p>
<ul>
  <li><code>results/log.txt</code> — Log completo da execução (parâmetros, F1-Scores, features selecionadas, tempos)</li>
  <li><code>results/all_bestsolution.png</code> / <code>results/all_bestsolution.pdf</code> — Gráfico das soluções construídas vs. entrada na fila de prioridade vs. melhoria na busca local</li>
  <li><code>results/evolution_plot.png</code> — Gráfico de evolução do tamanho das soluções (inicial vs. final após busca local), reproduzindo a Fig. 2 do artigo</li>
  <li><code>results/evolution_plot_data.csv</code> — Dados numéricos do gráfico de evolução (para replot ou análise posterior)</li>
</ul>

---

<a name="english-version"></a>
<h1 align="center">📌 Welcome to A-GRASPQ-FS Tool! 📌</h1>

<h4 align="left">
✔️ The <strong>A-GRASPQ-FS Tool</strong> (<em>Adaptive GRASPQ Feature Selection</em>) is a command-line tool implementing the adaptive metaheuristic proposed in <em>"Adaptive Feature Selection with Self-Tuning Subset Size for Intrusion Detection"</em> (ISCC 2026). Unlike the previous version (GRASPQ-FS), this version <strong>eliminates the need to manually define the number of features (k)</strong>, autonomously learning the optimal subset size through probabilistic selection with adaptive recalibration.
</h4>

<h2>📁 Repository Structure</h2>
<pre><code>.
├── data/                     # Folder for input datasets (not included — see note below)
├── results/                  # Generated logs and plots
├── Dockerfile                # Docker image for dependency-free execution
├── main.py                   # Main script: adaptive constructive and local search phases
├── utils.py                  # Helpers: data loading, preprocessing, evaluation, and plotting
├── priority_queue.py         # Custom MaxPriorityQueue implementation
├── baselines.py              # Baseline runner (SFS, RFE, RFECV)
├── requirements.txt          # Python dependencies
└── README.md                 # This file
</code></pre>

<h2>📋 Table of Contents</h2>
<ol>
  <li><a href="#what-is-new">What's New: A-GRASPQ-FS vs GRASPQ-FS</a></li>
  <li>Test &amp; Development Environment</li>
  <li>Requirements</li>
  <li>Datasets and Licensing</li>
  <li>How to Run</li>
  <li>Available Parameters</li>
  <li>Running Baselines</li>
  <li>Generated Outputs</li>
</ol>

<h2 id="what-is-new">🧠 What's New: A-GRASPQ-FS vs GRASPQ-FS</h2>
<table border="1">
<tr><th>Feature</th><th>GRASPQ-FS (previous)</th><th>A-GRASPQ-FS (current)</th></tr>
<tr><td>Subset size</td><td>Fixed — manually set by the user (<code>-is k</code>)</td><td>Adaptive — automatically learned over range [kmin, kmax]</td></tr>
<tr><td>Construction phase</td><td>Standard GRASP with fixed size</td><td>Reactive Construction with roulette-wheel size selection and adaptive weight recalibration</td></tr>
<tr><td>Local search</td><td>SWAP only (maintains fixed size)</td><td>Shuffled Neighborhood with ADD, REMOVE, and SWAP — dynamically adjusts subset size</td></tr>
<tr><td>Model dependency</td><td>Model-agnostic</td><td>Model-agnostic (preserved)</td></tr>
<tr><td>Manual grid search for k</td><td>Required</td><td>Not needed — converges to optimal k in a single run</td></tr>
</table>

<h3>🖱️ Test Environment</h3>
<table border="1">
<tr><th>Configuration</th><th>Machine</th></tr>
<tr><td>Operating System</td><td>Windows 11</td></tr>
<tr><td>Processor</td><td>13th Gen Intel(R) Core(TM) i7-13650HX @ 2.60GHz</td></tr>
<tr><td>RAM</td><td>16 GB (15.7 GB usable)</td></tr>
<tr><td>Python Version</td><td>3.10.0</td></tr>
</table>

<h3>⚙️ Development Environment</h3>
<table border="1">
<tr><th>Tool</th><th>Version</th></tr>
<tr><td>Python</td><td>3.10.0</td></tr>
<tr><td>Editor</td><td>VS Code / PyCharm</td></tr>
<tr><td>Terminal</td><td>PowerShell, CMD, or Bash</td></tr>
</table>

<h3>📝 Requirements</h3>
<p>This project uses Python 3 and the following libraries:</p>
<ul>
  <li>numpy ≥ 1.21</li>
  <li>pandas ≥ 1.3</li>
  <li>matplotlib ≥ 3.4</li>
  <li>scikit-learn ≥ 1.0</li>
  <li>xgboost ≥ 1.5</li>
</ul>

<h2>📂 Datasets and Licensing</h2>

<p>⚠️ <strong>Datasets are not included in this repository</strong> due to licensing and copyright constraints. Each user must obtain the datasets directly from their official sources and place them in the <code>data/</code> folder.</p>

<p>The project includes native support (automatic preprocessing) for the following datasets:</p>

<table border="1">
<tr><th>Identifier (<code>-d</code>)</th><th>Dataset</th><th>Expected file(s) in <code>data/</code></th></tr>
<tr><td><code>ereninho</code></td><td>ERENO — Smart Grid / IEC-61850 (default for initial testing)</td><td><code>hibrid_dataset_GOOSE_train.csv</code></td></tr>
<tr><td><code>batadal</code></td><td>BATADAL — Water infrastructure</td><td><code>BATADAL_dataset03.csv</code>, <code>BATADAL_dataset04.csv</code></td></tr>
<tr><td><code>wadi</code></td><td>WADI — Water distribution system</td><td><code>WADI.csv</code></td></tr>
<tr><td><code>wustl</code></td><td>WUSTL-EHMS-2020 — Healthcare / IoMT</td><td><code>wustl-ehms-2020.csv</code></td></tr>
<tr><td><code>cic-iot</code></td><td>CIC-IoT 2023 — IoT traffic with modern attacks</td><td>(configure in <code>utils.py</code>)</td></tr>
</table>

<p>To add a new dataset, implement its loading and preprocessing logic in the <code>load_data()</code> function in <code>utils.py</code>, following the pattern of existing datasets.</p>

<h2>🚀 How to Run</h2>

<h4>▶️ Option 1: Run Locally (Recommended for Development)</h4>
<ol>
  <li>
    <strong>Clone this repository and enter the project folder:</strong>
    <pre><code>git clone https://github.com/this-repository.git
cd this-repository</code></pre>
  </li>
  <li>
    <strong>Create and activate a virtual environment (recommended):</strong>
    <pre><code>python -m venv venv
venv\Scripts\activate       # on Windows
source venv/bin/activate    # on Unix/Mac</code></pre>
  </li>
  <li>
    <strong>Install the dependencies:</strong>
    <pre><code>pip install -r requirements.txt</code></pre>
  </li>
  <li>
    <strong>Run with the paper configuration (recommended for reproducibility):</strong>
    <pre><code>python main.py -d ereninho -a nb -rcl 10 -is 5 -pq 30 -lc 1000 -cc 100</code></pre>
    <p>ℹ️ The <code>hibrid_dataset_GOOSE_train.csv</code> (ERENO) dataset is already included in the <code>data/</code> folder for immediate testing. For other datasets, see the <a href="#datasets">Datasets and Licensing</a> section.</p>
  </li>
</ol>

<h4>⚡ Quick Run (for initial testing with lower computational cost)</h4>
<pre><code>python main.py -d ereninho -a nb -rcl 10 -is 5 -pq 10 -lc 50 -cc 50</code></pre>

<p><strong>Other usage examples:</strong></p>
<pre><code># Using KNN with paper configuration
python main.py -d ereninho -a knn -rcl 10 -is 5 -pq 30 -lc 1000 -cc 100

# Using Decision Tree with quick run
python main.py -d ereninho -a dt -rcl 10 -is 5 -pq 10 -lc 50 -cc 50

# Alternative parameter aliases also work:
python main.py --dataset ereninho --algorithm nb --rcl_size 10 --init_sol 5 --pq_size 30 --ls 1000 --const 100
</code></pre>

<h4>🐳 Option 2: Run with Docker (No Python Installation Required)</h4>
<ol>
  <li>
    <strong>Build the Docker image:</strong>
    <pre><code>docker build -t a-graspq-fs .</code></pre>
  </li>
  <li>
    <strong>Quick test (ERENO dataset already included in the repository):</strong>
    <pre><code>docker run --rm -v $(pwd)/results:/app/results a-graspq-fs -d ereninho -a nb -rcl 10 -is 5 -pq 10 -lc 50 -cc 50</code></pre>
  </li>
  <li>
    <strong>Full run with external dataset:</strong>
    <pre><code>docker run --rm -v $(pwd)/data:/app/data -v $(pwd)/results:/app/results a-graspq-fs -d ereninho -a nb -rcl 10 -is 1 -max_is 33 -pq 30 -lc 1000 -cc 100</code></pre>
  </li>
</ol>
<p>ℹ️ The <code>-v $(pwd)/results:/app/results</code> flag is recommended so that generated files (logs, plots) are accessible on your machine after the container exits. The <code>-v $(pwd)/data:/app/data</code> flag is only needed when using external datasets not included in the repository.</p>

<h2>🧾 Available Parameters</h2>

<table border="1">
<tr><th>Flag</th><th>Aliases</th><th>Description</th><th>Default</th></tr>
<tr><td><code>-d</code></td><td><code>--dataset</code></td><td>Dataset to use (<code>ereninho</code>, <code>batadal</code>, <code>wadi</code>, <code>wustl</code>, <code>cic-iot</code>)</td><td><code>ereninho</code></td></tr>
<tr><td><code>-a</code></td><td><code>--algorithm</code>, <code>--alg</code></td><td>Classifier (<code>nb</code>, <code>dt</code>, <code>knn</code>, <code>rf</code>, <code>svm</code>, <code>linear_svc</code>, <code>sgd</code>, <code>xgboost</code>)</td><td><code>nb</code></td></tr>
<tr><td><code>-rcl</code></td><td><code>--rcl_size</code>, <code>--rcl</code></td><td>Restricted Candidate List (RCL) size</td><td><code>10</code></td></tr>
<tr><td><code>-is</code></td><td><code>--initial_solution</code>, <code>--init_sol</code></td><td>Minimum initial solution size (kmin)</td><td><code>5</code></td></tr>
<tr><td><code>-max_is</code></td><td><code>--max_initial_solution</code></td><td>Maximum initial solution size (kmax)</td><td><code>30</code></td></tr>
<tr><td><code>-pq</code></td><td><code>--priority_queue</code>, <code>--pq_size</code></td><td>Priority Queue capacity (SPQ)</td><td><code>10</code></td></tr>
<tr><td><code>-cc</code></td><td><code>--constructive_iterations</code>, <code>--const</code></td><td>Number of constructive phase iterations (Iconst)</td><td><code>100</code></td></tr>
<tr><td><code>-lc</code></td><td><code>--local_iterations</code>, <code>--ls</code></td><td>Max local search iterations per solution (Ilocal)</td><td><code>50</code></td></tr>
<tr><td><code>-alpha</code></td><td><code>--alpha</code></td><td>Semi-greedy randomness factor (0.0 to 1.0)</td><td><code>0.3</code></td></tr>
<tr><td><code>-ess</code></td><td><code>--evaluation_sample_size</code></td><td>Fraction of training data used for fast evaluation (0.0 to 1.0)</td><td><code>0.1</code></td></tr>
</table>

<h3>🔬 Paper Configuration (Reproducibility)</h3>
<p>To reproduce the experiments from the ISCC 2026 paper, use:</p>
<pre><code>python main.py -d ereninho -a nb          -rcl 10 -is 1 -max_is 33 -pq 30 -lc 1000 -cc 100
python main.py -d ereninho -a knn         -rcl 10 -is 1 -max_is 33 -pq 30 -lc 1000 -cc 100
python main.py -d ereninho -a dt          -rcl 10 -is 1 -max_is 33 -pq 30 -lc 1000 -cc 100
python main.py -d ereninho -a linear_svc  -rcl 10 -is 1 -max_is 33 -pq 30 -lc 1000 -cc 100
</code></pre>

<h2>📊 Running Baselines</h2>
<p>The <code>baselines.py</code> script implements the baselines used in the paper (SFS Forward/Backward, RFE, and RFECV) for direct comparison with A-GRASPQ-FS.</p>

<pre><code># SFS Forward with Naive Bayes, selecting 10 features
python baselines.py -d ereninho -m sfs_forward -a nb -nf 10

# RFE with Decision Tree, selecting 5 features
python baselines.py -d ereninho -m rfe -a dt -nf 5

# RFECV with Random Forest (determines number automatically)
python baselines.py -d ereninho -m rfecv -a rf
</code></pre>

<p><strong>baselines.py parameters:</strong></p>
<ul>
  <li><code>-d</code>, <code>--dataset</code>: Dataset to use</li>
  <li><code>-m</code>, <code>--method</code>: Method (<code>sfs_forward</code>, <code>sfs_backward</code>, <code>rfe</code>, <code>rfecv</code>)</li>
  <li><code>-a</code>, <code>--algorithm</code>: Classifier (<code>knn</code>, <code>nb</code>, <code>dt</code>, <code>linear_svc</code>, <code>rf</code>)</li>
  <li><code>-nf</code>, <code>--n_features</code>: Number of features to select (required for RFE; optional for SFS; ignored for RFECV)</li>
</ul>

<h2>📤 Generated Outputs</h2>
<p>After execution, the following files are generated automatically inside the <code>results/</code> folder (created by the script if it does not exist):</p>
<ul>
  <li><code>results/log.txt</code> — Full execution log (parameters, F1-Scores, selected features, timing)</li>
  <li><code>results/all_bestsolution.png</code> / <code>results/all_bestsolution.pdf</code> — Plot of constructed solutions vs. priority queue entries vs. local search improvements</li>
  <li><code>results/evolution_plot.png</code> — Size evolution plot (initial vs. final after local search), reproducing Fig. 2 from the paper</li>
  <li><code>results/evolution_plot_data.csv</code> — Numerical data from the evolution plot (for replotting or further analysis)</li>
</ul>

<hr>
