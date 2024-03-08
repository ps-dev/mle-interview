setup:
	pip install -r requirements.txt

serve-api:
	python -m flask --app api/app run --debug --port 5005 --host 0.0.0.0
	
serve-model:
	docker compose up tf-serving

train:
	TF_CPP_MIN_LOG_LEVEL=3 python -m model.main

test:
	pytest ./tests