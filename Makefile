run:
	conda run --no-capture-output -n automl python3 warmstart_template.py

install:
	conda create -n automl python=3.10
	conda install -n automl swig
	conda run --no-capture-output -n automl pip install -r requirements.txt

uninstall:
	conda remove -n automl --all

download-data:
	conda run --no-capture-output -n automl python3 datasets.py	

remove-data:
	rm -r data

.PHONY: run, install, uninstall, download-data, remove-data