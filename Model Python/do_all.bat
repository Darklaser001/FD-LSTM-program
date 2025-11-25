@echo off
REM Ustawia polskie znaki w konsoli
chcp 65001 > nul

ECHO.
ECHO =======================================================
ECHO ==   AUTOMATYCZNY TRENING WSZYSTKICH MODELI (20 EPOK)  ==
ECHO =======================================================
ECHO.

REM Zdefiniuj listę rodzin (oddzielone spacją)
REM Upewnij się, że są tu te same nazwy co w BASE_DIRS w preprocesData.py
set FAMILIES=RLM1 RLM2 RLR1 RLV1 RLV2

REM Pętla po każdej rodzinie
for %%f in (%FAMILIES%) do (
    ECHO.
    ECHO [INFO] --- Rozpoczynam trening dla: %%f ---
    
    REM Uruchom skrypt train_model.py dla danej rodziny
    REM --input_dim zostanie wykryty automatycznie
    REM --hidden_dim itp. zostaną wzięte z domyślnych w skrypcie
    
    python train_model.py --family %%f --epochs 20
    
    ECHO [INFO] --- Trening dla %%f zakończony ---
    ECHO.
)

ECHO =======================================================
ECHO ==      WSZYSTKIE TRENINGI ZOSTAŁY ZAKOŃCZONE      ==
ECHO =======================================================
ECHO.

ECHO.
ECHO =======================================================
ECHO ==     AUTOMATYCZNA DETEKCJA WSZYSTKICH MODELI    ==
ECHO =======================================================
ECHO.

REM Pętla po każdej rodzinie
for %%f in (%FAMILIES%) do (
    ECHO.
    ECHO [INFO] --- Rozpoczynam detekcję dla: %%f ---
    
    REM Uruchom skrypt detect.py dla danej rodziny
    REM --input_dim zostanie wykryty automatycznie
    REM --hidden_dim itp. zostaną wzięte z domyślnych w skrypcie
    
    python detect.py --family %%f
    
    ECHO [INFO] --- Detekcja dla %%f zakończony ---
    ECHO.
)

ECHO =======================================================
ECHO ==      WSZYSTKIE DETEKCJE ZOSTAŁY ZAKOŃCZONE      ==
ECHO =======================================================
ECHO.
pause