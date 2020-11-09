install-deps:
	conda install -c conda-forge --file requirements-conda.txt
	pip install -r requirements-pip.txt

run-dev:
	streamlit run src/app.py

run:
	streamlit run src/app.py
