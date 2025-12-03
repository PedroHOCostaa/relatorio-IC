# Classificação de Imagens dos Simpsons (Cor + 5 Classificadores + Ensemble)

Trabalho prático de Inteligência Artificial para classificação de personagens dos Simpsons. As imagens `.bmp` são redimensionadas para `96x96`, recebem descritores HOG + histogramas de cor RGB e são avaliadas com k-NN, Árvore de Decisão, Random Forest, SVM-RBF, MLP e um ensemble de votação com 20 classificadores. Todo o treinamento é feito em 10-fold stratified cross-validation sobre o conjunto Train, e a validação final é feita no conjunto Valid fornecido.

## Visão Geral
- Classes: Bart, Homer, Lisa, Maggie e Marge.
- Base organizada em `data/simpsons/Train` (treino/CV) e `data/simpsons/Valid` (validação final).
- Extração de atributos: histogramas de cor RGB (16 bins/canal).
- Classificadores: k-NN, Decision Tree, Random Forest, SVM (kernel RBF), MLP.
- Ensemble: VotingClassifier (voting soft) com 20 estimadores variados (k-NN, SVM, RF, DT, MLP).
- Cross-validation: Stratified K-Fold (k=10) com GridSearchCV otimizando `f1_macro`.
- Métricas: Accuracy, Precision macro, Recall macro, F1-score macro, matrizes de confusão (validação) e relatórios por classe.

## Estrutura do Dataset
```
data/
└── simpsons/
    ├── Train/
    │   ├── bart/
    │   ├── homer/
    │   ├── lisa/
    │   ├── maggie/
    │   └── marge/
    └── Valid/
        ├── bart/
        ├── homer/
        ├── lisa/
        ├── maggie/
        └── marge/
```
Se seus arquivos `.bmp` estiverem soltos dentro de `Train` ou `Valid`, rode `organizar.py` para criá-las e mover as imagens para as pastas corretas.

## Pré-requisitos
- Python 3.10+.
- Pacotes: numpy, pandas, scikit-learn, matplotlib, opencv-python, scikit-image, Pillow.
- Extensão Jupyter no Visual Studio Code.

## Configuração do Ambiente
1) Criar e ativar o ambiente virtual:
```bash
python3 -m venv .venv
source .venv/bin/activate
```
2) Instalar dependências:
```bash
pip install numpy pandas scikit-learn matplotlib opencv-python scikit-image Pillow
# ou: pip install -r requirements.txt   (se você mantiver esse arquivo)
```

## Preparação dos Dados
1) Garanta a estrutura `data/simpsons/Train` e `data/simpsons/Valid` com os `.bmp`.
2) (Opcional) Automatizar a organização:
```bash
python organizar.py
```
O script cria as subpastas de classe e move as imagens cujo prefixo do arquivo indica o personagem (ex.: `homer057.bmp` → `homer/`).

## Execução (Treino e Avaliação)
Selecione o arquivo train_eval.ipynb e depois clice em "run all"


O script:
- Carrega imagens de `Train` (treino/CV) e `Valid` (validação final).
- Extrai histogramas de cor.
- Faz GridSearchCV com 10 folds estratificados para k-NN, Decision Tree, Random Forest, SVM-RBF e MLP.
- Treina um ensemble de votação (20 estimadores) e valida no conjunto Valid.
- Salva métricas, relatórios e matrizes de confusão em `outputs/`.

### Saídas geradas
```
outputs/
├── confusion_knn.png
├── confusion_svm_rbf.png
├── confusion_decision_tree.png
├── confusion_random_forest.png
├── confusion_mlp.png
├── confusion_ensemble.png
├── report_knn_valid.txt
├── report_knn_cv.txt
├── report_svm_rbf_valid.txt
├── report_svm_rbf_cv.txt
├── report_decision_tree_valid.txt
├── report_decision_tree_cv.txt
├── report_random_forest_valid.txt
├── report_random_forest_cv.txt
├── report_mlp_valid.txt
├── report_mlp_cv.txt
├── report_ensemble_valid.txt
├── report_ensemble_cv.txt
├── metrics_summary.csv
└── results.json
```

## Resultados de Referência
- Acurácias/f1 variam com a divisão, mas SVM-RBF e o ensemble tendem a ficar próximos ou acima de 97–98% na base fornecida. Compare sempre com os números gerados em `outputs/metrics_summary.csv`.
- As matrizes de confusão tendem a ser diagonais, indicando baixa confusão entre personagens.

## Estrutura do Projeto
```
simpsons/
├── data/              # imagens (Train/Valid)
├── outputs/           # resultados gerados pelo treinamento
├── organizar.py       # script para organizar as pastas de classe
└── train_eval.ipynb   # pipeline principal: carga, extração, treino, avaliação
```

## Observações
- Ajuste os hiperparâmetros em `src/train_eval.py` se quiser explorar novos valores.
- Para reprodutibilidade, os folds usam `random_state=42`. Mantenha a mesma divisão de dados para comparar resultados.
