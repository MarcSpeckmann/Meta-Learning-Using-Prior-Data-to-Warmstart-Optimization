install:
	conda create -n automl -y python=3.10 
	conda install -n automl -y swig
	conda run --no-capture-output -n automl pip install -r requirements.txt

uninstall:
	conda remove -n automl --all

clean:
	rm -r ray_tune
	rm -r data

.PHONY: install, uninstall, clean