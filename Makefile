run:
	conda run --no-capture-output -n automl python main.py

install:
	conda create -n automl python=3.10
	conda install -n automl swig
	conda run --no-capture-output -n automl pip install -r requirements.txt

uninstall:
	conda remove -n automl --all

.PHONY: run, install, uninstall