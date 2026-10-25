@echo off
echo =====================================================================
echo  Starting Liver Lesion GAN-CNN Streamlit Web Interface...
echo =====================================================================
py -m streamlit run app.py
if %ERRORLEVEL% NEQ 0 (
    python -m streamlit run app.py
)
pause
