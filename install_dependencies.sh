#!/bin/bash
# Script untuk menginstal semua dependensi dengan lebih reliable

echo "Menginstall dependensi inti..."
pip install -r requirements.txt

echo "Mengupgrade pip ke versi terbaru..."
pip install --upgrade pip

echo "Apakah Anda ingin menginstall library AI/ML tambahan? (y/n)"
read response

if [[ "$response" == "y" || "$response" == "Y" ]]; then
    echo "Menginstall dependensi AI/ML tambahan..."
    
    echo "Menginstall XGBoost..."
    pip install xgboost==2.0.2
    
    echo "Menginstall LightGBM..."
    pip install lightgbm==4.1.0
    
    echo "Menginstall PyTorch..."
    pip install torch==2.1.1 torchvision==0.16.1
    
    echo "Menginstall TensorFlow..."
    pip install tensorflow==2.15.0
    
    echo "Menginstall CatBoost..."
    pip install catboost==1.2.3
    
    echo "Menginstall time series libraries tambahan..."
    pip install tsfresh==0.20.2 sktime==0.24.0
    
    echo "Semua dependensi AI/ML berhasil diinstal!"
else
    echo "Melewati instalasi library AI/ML tambahan."
fi

echo "Instalasi selesai!"
