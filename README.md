# Projekt NAI - Modele Podsumowujące

## Przegląd

To repozytorium zawiera kod do oceny i porównywania różnych modeli podsumowujących, korzystając z różnych modeli wytrenowanych dostępnych w bibliotece Hugging Face Transformers. Projekt skupia się na trzech różnych modelach: BART (facebook/bart-large-cnn), LED (pszemraj/led-base-book-summary) i mT5 (csebuetnlp/mT5_multilingual_XLSum).

## Modele

### Model 1 - BART (facebook/bart-large-cnn)

Skrypt `model1.py` wykorzystuje model BART do podsumowywania tekstu. Wczytuje zestaw danych, generuje podsumowania i ocenia wydajność, korzystając z precyzji, czułości i wyniku F1.

### Model 2 - mT5 (csebuetnlp/mT5_multilingual_XLSum)

Skrypt `model2.py` wykorzystuje model mT5 do podsumowywania tekstu. Wczytuje zestaw danych, generuje podsumowania i ocenia wydajność, korzystając z precyzji, czułości i wyniku F1.

### Model 3 - LED (pszemraj/led-base-book-summary) 

Skrypt `model3.py` wykorzystuje model LED do podsumowywania tekstu. Wczytuje zestaw danych, generuje podsumowania i ocenia wydajność, korzystając z precyzji, czułości i wyniku F1.

## Jak zacząć

### 1. Sklonuj repozytorium:

```bash
git clone https://github.com/twoja-nazwa-uzytkownika/ProjektNAI.git
cd ProjektNAI
```

### 2. Zainstaluj zależności:
```bash
pip install -r requirements.txt
```

### 3. Uruchom odpowiedni skrypt modelu:
```bash
python3 main.py
```

### lub
```bash
py3 main.py
```

### 4. Zbiór danych
Upewnij się, że twój zbiór danych jest w odpowiednim formacie Excel i zawiera niezbędne kolumny dla każdego skryptu.

## Współtwórcy
Kacper Krawczyk (s24547),
Wojciech Bezak (s24573)


