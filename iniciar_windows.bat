@echo off
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  py -3 -m venv .venv
  if errorlevel 1 goto erro
)
.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 goto erro
.venv\Scripts\python.exe -m streamlit run app.py
pause
exit /b
:erro
echo Nao foi possivel preparar o aplicativo. Confira Python e conexao com a internet.
pause
