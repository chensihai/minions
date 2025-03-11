conda install -c conda-forge pygobject gtk3 pycairo -y

pyinstaller --onefile --windowed --name "MinionsDesktop" --add-data "static;static" --add-data "assets;assets" main.py --noconsole

pyinstaller --onefile --windowed --name "MinionsDesktop" --add-data "static;static" --add-data "assets;assets" --upx-dir="C:\tools\upx-5.0.0-win64\upx" main.py

pyinstaller MinionsDesktop.spec

conda install -n base -c conda-forge mamba

mamba install -c conda-forge mlx-lm
