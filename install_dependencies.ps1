# Script untuk menginstall semua dependensi dengan lebih reliable di Windows

Write-Host "Menginstall dependensi inti..." -ForegroundColor Green
pip install -r requirements.txt

Write-Host "Mengupgrade pip ke versi terbaru..." -ForegroundColor Green
python -m pip install --upgrade pip

$response = Read-Host "Apakah Anda ingin menginstall library AI/ML tambahan? (y/n)"

if ($response -eq "y" -or $response -eq "Y") {
    Write-Host "Menginstall dependensi AI/ML tambahan..." -ForegroundColor Green
    
    Write-Host "Menginstall XGBoost..." -ForegroundColor Cyan
    pip install xgboost==2.0.2
    
    Write-Host "Menginstall LightGBM..." -ForegroundColor Cyan
    pip install lightgbm==4.1.0
    
    Write-Host "Menginstall PyTorch..." -ForegroundColor Cyan
    pip install torch==2.1.1 torchvision==0.16.1
    
    Write-Host "Menginstall TensorFlow..." -ForegroundColor Cyan
    pip install tensorflow==2.15.0
    
    Write-Host "Menginstall CatBoost..." -ForegroundColor Cyan
    pip install catboost==1.2.3
    
    Write-Host "Menginstall time series libraries tambahan..." -ForegroundColor Cyan
    pip install tsfresh==0.20.2 sktime==0.24.0
    
    Write-Host "Semua dependensi AI/ML berhasil diinstal!" -ForegroundColor Green
}
else {
    Write-Host "Melewati instalasi library AI/ML tambahan." -ForegroundColor Yellow
}

Write-Host "Instalasi selesai!" -ForegroundColor Green
