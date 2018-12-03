# Bakalářská práce - Slavík
Implementace řešení bakalářské práce "**Algoritmické obchodování na burze s využitím umělých 
neuronových sítí**". Implementace obsahuje:
- zpracování dat
- 
## Obsah Package
### Interface
- Jupyter Notebook vysvětlující workflow trénování modelu
- Iterátor pro hromadné trénování neuronových sítí
### Dataset Manager
- `datasets` - složka s datasetem ve formátu .csv
- `dataset_manager.py` - package pro správu a manipulaci s daty
- `technical_indicators.py` - implementace použitých technických indikátorů
### Neural Lab
- `logger.py` - package pro logování informací o natrénovaných modelech
- `model_builder.py` - implementace sestavení modelu a jeho trénování
- `model_evaluation.py` - implementace evaluace vytrénovaného modelu
- `model_strategies.py` - implementace algoritmických strategií a jejich evaluace 

## Instalace
### Python verze 3.6.5
Práce je programována v jazyce Python 3.6.5. Je možné, že na nejnovější verzi nebudou 
spolehlivě běžet všechny použité dependencies.
Pro kontrolu verze použijte následující příkaz.
```bash
python --version
```
### Instalace virtual environment
```bash
pip install virtualenv
```
### Vytvoření virtualního prostředí
Přejděte do složky, kde chcete nový python interpreter instalovat a použijte
```bash
virtualenv ann-algotrading
```
### Aktivate virtuálního prostředí
```bash
ann-algotradin\Scripts\activate
```
### Instalace dependencies
`cd` do root složky projektu a poté nainstalujte dependencies
```bash
pip install -r requirements.txt
```
## Jupyter Notebook
Notebook s popisem workflow je napsán v jupyter notebooku
```bash
jupyter notebook
```
Poté v adresáři vyhledejte root složku a v ní interface, uvnitř které je umístěn notebook.

## Help
- https://developer.akamai.com/blog/2017/06/21/how-building-virtual-python-environment
- https://jupyter-notebook-beginner-guide.readthedocs.io/en/latest/execute.html