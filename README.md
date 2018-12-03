# Bakalářská práce - Slavík
Implementace řešení bakalářské práce "**Algoritmické obchodování na burze s využitím umělých 
neuronových sítí**". 

#### Funkcionalita:
- zpracování dat
- trénování neuronové sítě
- evaluace modelu
- optimalizace algoritmických strategií
- obchodování automatizované prahové strategie
- obchodování automatizované prahové strategie s MACD indikátorem

## Průvodce implementací
Pokud Vás zajímá proces implementace, ale nechce se Vám instalovat package lokálně, je zde 
připraven průvodce v podobě jupyter notebooku mapující workflow a funkcionalitu práce.
- [**Průvodce implementací - ONLINE**](https://github.com/NightingaleV/bakalarska-prace-ann-algotrading/blob/master/interface/Workflow_Guide.ipynb)

## Struktura
### Interface
- `Workflow_Guide.ipynb` - Jupyter Notebook vysvětlující implementaci trénování modelu a 
obchodních strategií
- `training_iterator.py` - Iterátor pro hromadné trénování neuronových sítí
### Dataset Manager
- `dataset_manager.py` - pro správu a manipulaci s daty
- `technical_indicators.py` - technické indikátory
- `datasets` - složka s datasetem ve formátu .csv
### Neural Lab
- `logger.py` - package pro logování informací o natrénovaných modelech
- `ann_clasification.py` - řízení topologie a parametrů modelu
    - `model_builder.py` - sestavení modelu a jeho trénování
    - `model_evaluation.py` - evaluace vytrénovaného modelu
    - `model_strategies.py` - algoritmických strategií a jejich evaluace 

## Instalace
Pokud chcete zprovoznit implementaci, je třeba dodržet následující pokyny.
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
### Aktivace virtuálního prostředí
```bash
ann-algotradin\Scripts\activate
```
### Instalace dependencies
`cd` do root složky projektu a poté nainstalujte dependencies
```bash
pip install -r requirements.txt
```
## Popis implementace a workflow
Notebook s popisem workflow je napsán v jupyter notebooku
```bash
jupyter notebook
```
Poté v adresáři vyhledejte root složku a v ní interface, uvnitř které je umístěn notebook 
`workflow_description.ipynb`.

## Training Iterator
Slouží k hromadnému trénování neuronových sítí za účelem optimalizace parametrů. Hromadné 
trénování je velmi časově náročné a pohybuje se v řádech hodin.

## Help
- https://developer.akamai.com/blog/2017/06/21/how-building-virtual-python-environment
- https://jupyter-notebook-beginner-guide.readthedocs.io/en/latest/execute.html