@echo off
title TruthPulse - Full AI Dashboard (Y. Louati & F. Saidi)
cd /d "%~dp0"
echo Lancement du serveur Streamlit TruthPulse Complet...
python -m streamlit run app.py
pause

