@echo off

ECHO.
ECHO =======================================================
ECHO ==   AUTOMATYCZNY TRENING WSZYSTKICH MODELI (20 EPOK)  ==
ECHO =======================================================
ECHO.

set EPOCHS=20

REM =======================================================
REM 1. RLM1 (Vi = 1.0)
REM =======================================================
ECHO [RLM1] Start treningu (Vi=1.0)...
python train_model.py --family RLM1 --epochs %EPOCHS% --vi 1.0 --load_model
ECHO.

REM =======================================================
REM 2. RLM2 (Vi = 0.9)
REM =======================================================
ECHO [RLM2] Start treningu (Vi=0.9)...
python train_model.py --family RLM2 --epochs %EPOCHS% --vi 0.9 --load_model
ECHO.

REM =======================================================
REM 3. RLR1 (Vi = 1.0)
REM =======================================================
ECHO [RLR1] Start treningu (Vi=1.0)...
python train_model.py --family RLR1 --epochs %EPOCHS% --vi 1.0 --load_model
ECHO.

REM =======================================================
REM 4. RLV1 (Vi = 1.2)
REM =======================================================
ECHO [RLV1] Start treningu (Vi=1.2)...
python train_model.py --family RLV1 --epochs %EPOCHS% --vi 1.2 --load_model
ECHO.

REM =======================================================
REM 5. RLV2 (Vi = 0.9)
REM =======================================================
ECHO [RLV2] Start treningu (Vi=0.9)...
python train_model.py --family RLV2 --epochs %EPOCHS% --vi 0.9 --load_model
ECHO.

ECHO =======================================================
ECHO ==      WSZYSTKIE TRENINGI ZOSTAŁY ZAKOŃCZONE      ==
ECHO =======================================================
ECHO.

ECHO.
ECHO =======================================================
ECHO ==     AUTOMATYCZNA DETEKCJA WSZYSTKICH MODELI    ==
ECHO =======================================================
ECHO.

set FAMILIES=RLM1 RLM2 RLR1 RLV1 RLV2

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