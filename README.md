# Model Evaluation Toolkit

Projeto funcional para avaliar modelos de classificacao com Python.

## Como executar

```bash
pip install pandas numpy scikit-learn matplotlib
python src/evaluate.py --output-dir reports
```

## Saida

- `reports/threshold_metrics.csv`

## O que o projeto demonstra

- treino de modelo demonstrativo com dataset publico do scikit-learn;
- calculo de acuracia, precisao, recall, F1 e ROC AUC;
- avaliacao de diferentes thresholds;
- matriz de confusao em cada ponto de corte.

## Estrutura

```text
src/evaluate.py
reports/threshold_metrics.csv
```

## Dados

O exemplo usa dataset publico carregado diretamente pelo scikit-learn.

## Licenca

MIT License.
