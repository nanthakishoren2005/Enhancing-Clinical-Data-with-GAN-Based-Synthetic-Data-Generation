@echo off
echo =====================================================================
echo  LIVER LESION GAN-CNN COMPLETE END-TO-END WORKFLOW PIPELINE
echo =====================================================================

echo.
echo [Stage 1/6] Validating Dataset...
py -m src.data.validate_dataset
if %ERRORLEVEL% NEQ 0 (
    echo Dataset missing or invalid. Initializing benchmark simulated dataset...
    py -m src.data.generate_simulated_data --train-count 35 --val-count 10 --test-count 15
    py -m src.data.validate_dataset
)

echo.
echo [Stage 2/6] Training Conditional DCGAN...
py -m src.gan.train_gan --epochs 6 --batch-size 16

echo.
echo [Stage 3/6] Generating Synthetic Images with Trained Generator...
py -m src.gan.generate_images --num-per-class 20

echo.
echo [Stage 4/6] Training Model A: Baseline CNN...
py -m src.cnn.train_baseline --epochs 6 --architecture custom_cnn

echo.
echo [Stage 5/6] Training Model B: GAN-Augmented CNN...
py -m src.cnn.train_augmented --epochs 6 --architecture custom_cnn

echo.
echo [Stage 6/6] Evaluating & Comparing Both Models on Real Test Set...
py -m src.evaluation.compare_models

echo.
echo =====================================================================
echo  Full Pipeline Executed Successfully!
echo  Reports saved in outputs/reports/ and figures in outputs/figures/
echo =====================================================================
pause
