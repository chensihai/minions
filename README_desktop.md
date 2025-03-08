conda install -c conda-forge pygobject gtk3 pycairo mlx-lm -y

pyinstaller --onefile --windowed --name "MinionsDesktop" --add-data "static;static" --add-data "assets;assets" main.py

pyinstaller --onefile --windowed --name "MinionsDesktop" --add-data "static;static" --add-data "assets;assets" --upx-dir="C:\tools\upx-5.0.0-win64\upx" main.py
