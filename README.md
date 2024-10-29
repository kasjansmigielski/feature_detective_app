### Zmiany w specyfikacji:
* Użytkownik może załadować plik CSV z danymi -> dodatkowo ma możliwość wczytywania plików JSON lub pobierania gotowych zbiorów danych z chmury
* Użytkownik wskazuje kolumnę docelową -> dodatkowo ma możliwość skorzystania z automatycznej detekcji kolumny (generowanej przez LLM)
* Rozpoznajemy czy mamy do czynienia z problemem klasyfikacji czy regresji (bez zmian)
* Generujemy najlepszy model dla danego problemu (bez zmian)
* Wyświetlamy najważniejsze cechy (bez zmian)
* Przesyłamy użytkownikowi opis słowny tego co znaleźliśmy -> dodatkowo model LLM na podstawie tego opisu generuje przykładowe rekomendacje dla użytkownika

### Taski:
* v1 - możliwość wczytania pliku CSV i wyświetlenie przykładowych danych
* v2 - wybór kolumny docelowej
* v3 - rozpoznanie problemu klasyfikacji/regresji i wyświetlenie informacji
* v4 - wygenerowanie modelu + wyświetlenie najważniejszych cech
* v5 - wygenerowanie opisu + podpięcie modelu LLM pod Langfuse + zastosowanie api key protection + możliwość wyboru modelu wraz z wyświetleniem kosztów zużycia
* v6 - automatyczna detekcja kolumny docelowej
* v7 - rekomendacje (jakie kroki można podjąć żeby jeszcze bardziej wpłynąć na wynik)
* v8 - możliwość wczytania gotowego datasetu (z biblioteki get_data) umieszczonego w Digital Ocean Space
* v9 - możliwość wczytania formatu JSON

### Zależności:
* streamlit
* pycaret
* pandas
* mathplotlib
* python-dotenv
* langfuse
* instructor
* pydantic
* boto3