# Datasets

Small datasets are included directly in this folder. Larger datasets need to be downloaded — instructions below.

## Included in repo

| Dataset | Size | Description |
|---------|------|-------------|
| iris | 12K | Classic flower classification (3 species, 4 features) |
| diabetes | 24K | Diabetes progression prediction (10 features) |
| heart_disease | 24K | Heart disease diagnosis from clinical features |
| auto_mpg | 36K | Predict car fuel efficiency from specs |
| titanic | 64K | Passenger survival prediction |
| breast_cancer | 128K | Tumor malignancy classification from cell measurements |
| wine_quality | 348K | Red and white wine quality ratings from chemical properties |
| california_housing | 1.4M | California median house values by block group |
| movielens | 3.2M | MovieLens Latest Small — 100k ratings, 9k movies, 600 users |
| dry_bean | 3.6M | Classify 7 bean types from shape/dimension features |
| ml-100k | 16M | MovieLens 100K — classic recommendation benchmark |

## Download required

### Tabular

**adult_income** (5.8MB) — Predict income >$50K from census data (UCI)
```bash
wget https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data
wget https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test
```
Or: `sklearn.datasets.fetch_openml(name='adult', version=2)`

**bank_marketing** (11MB) — Predict term deposit subscription from phone campaigns (UCI)
```bash
wget https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank.zip
```

**census_income_kdd** (149MB) — Extended census income dataset, more features than adult_income (UCI)
```bash
wget https://archive.ics.uci.edu/ml/machine-learning-databases/census-income-mld/census-income.data.gz
wget https://archive.ics.uci.edu/ml/machine-learning-databases/census-income-mld/census-income.test.gz
```

**covertype** (132MB) — Predict forest cover type from cartographic variables, 7 classes (UCI)
```python
from sklearn.datasets import fetch_covtype
data = fetch_covtype()
```

**creditcard** (210MB) — Credit card fraud detection, PCA-transformed features (Kaggle)
```bash
kaggle datasets download -d mlg-ulb/creditcardfraud
```

**steam_games** (78MB) — Australian Steam user reviews and playtime data (Kaggle)
```bash
kaggle datasets download -d tamber/steam-video-games
```

### Images

**mnist** (53MB) — Handwritten digits 0-9, 28x28 grayscale
```python
torchvision.datasets.MNIST(root='./data', download=True)
```

**fashion_mnist** (53MB) — Zalando clothing items, 10 classes, 28x28 grayscale
```python
torchvision.datasets.FashionMNIST(root='./data', download=True)
```

**cifar10** (341MB) — 60k 32x32 color images, 10 classes
```python
torchvision.datasets.CIFAR10(root='./data', download=True)
```

**cifar100** (339MB) — 60k 32x32 color images, 100 fine-grained classes
```python
torchvision.datasets.CIFAR100(root='./data', download=True)
```

**flowers102** (348MB) — Oxford 102 Flower Categories
```python
torchvision.datasets.Flowers102(root='./data', download=True)
```

**oxford_pets** (849MB) — 37 pet breeds with segmentation masks
```python
torchvision.datasets.OxfordIIITPet(root='./data', download=True)
```

**food101** (5GB) — 101 food categories, 1000 images each
```python
torchvision.datasets.Food101(root='./data', download=True)
```

**celebfaces** (3.1GB) — CelebA, 200k+ celebrity faces with 40 attribute annotations
```bash
kaggle datasets download -d jessicali9530/celeba-dataset
```
Torchvision's `CelebA(download=True)` often fails due to Google Drive quota limits.

**coco2017** (25GB) — Object detection, segmentation, captioning (330k images)
```bash
wget http://images.cocodataset.org/zips/train2017.zip    # 18GB
wget http://images.cocodataset.org/zips/val2017.zip      # 1GB
wget http://images.cocodataset.org/zips/test2017.zip     # 6GB
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
```

### Text / NLP

**squad1.1** (34MB) — Stanford Question Answering Dataset v1.1
```bash
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json
```

**squad2.0** (45MB) — SQuAD v2.0 with unanswerable questions
```bash
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json
```

**wikitext** (13MB) — WikiText-2 language modeling benchmark
```python
from datasets import load_dataset
ds = load_dataset("wikitext", "wikitext-2-raw-v1")
```
Or: `wget https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-raw-v1.zip`

**yelp** (5.4GB) — 6M+ business reviews for sentiment/NLP tasks
Requires agreeing to terms at https://www.yelp.com/dataset (no direct download).

### Recommendation / Collaborative Filtering

**beeradvocate** (428MB) — Multi-aspect beer reviews (appearance, aroma, palate, taste)
Download from https://cseweb.ucsd.edu/~jmcauley/datasets.html#multi_aspect

**ratebeer** (381MB) — Multi-aspect beer reviews, companion to BeerAdvocate
Download from https://cseweb.ucsd.edu/~jmcauley/datasets.html#multi_aspect

**goodreads** (4.2GB) — Book reviews and metadata, split by genre
Download from https://mengtingwan.github.io/data/goodreads.html

**food_recipes** (852MB) — Food.com recipes and user interactions
```bash
kaggle datasets download shuyangli94/food-com-recipes-and-user-interactions
```

**nyctaxi** (55GB) — NYC taxi trip records (pickup/dropoff, fares, tips)
Monthly Parquet files at https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page
