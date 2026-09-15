PYTEST = TF_CPP_MIN_LOG_LEVEL=3 pytest
HANDLE ?= e337a675-46f5-437e-aa72-5d43643b5461

.PHONY: help setup serve-api serve-model ping-model train test \
	test-1 test-2 test-3 test-4 test-5 test-6 test-7

help:
	@printf 'Setup and servers\n'
	@printf '  make setup        Install dependencies\n'
	@printf '  make train        Train the model and export it to serving/\n'
	@printf '  make serve-model  Start TF Serving on port 8501\n'
	@printf '  make serve-api    Start the Flask API on port 5005\n'
	@printf '  make ping-model   Check that TF Serving has loaded the model\n'
	@printf '\nInterview tasks (run only the target for the task you are on)\n'
	@printf '  make test-1       Task 1: model build\n'
	@printf '  make test-2       Task 2: API starts up\n'
	@printf '  make test-3       Task 3: probability scores\n'
	@printf '  make test-4       Task 4: interests count\n'
	@printf '  make test-5       Task 5: response time\n'
	@printf '  make test-6       Task 6: probability in the response\n'
	@printf '  make test-7       Task 7: min_probability filtering\n'
	@printf '  make test         Every test at once (expect noise until all tasks are done)\n'

setup:
	pip install -r requirements.txt

serve-api:
	python -m flask --app api/app run --debug --port 5005 --host 0.0.0.0

serve-model:
	docker compose up tf-serving

ping-model:
	curl -s http://localhost:8501/v1/models/interests-model | python3 -m json.tool

train:
	TF_CPP_MIN_LOG_LEVEL=3 python -m model.main

test:
	$(PYTEST)

test-1:
	$(PYTEST) tests/test_model.py::TestModel::test_model_build

# No automated test — the API only starts once the bug is fixed.
test-2:
	@curl -fs http://localhost:5005/ \
		&& printf '\nPASS: the API is up\n' \
		|| printf 'FAIL: the API is not reachable on port 5005 (see task 2)\n'

test-3:
	$(PYTEST) tests/test_model.py::TestModel::test_probability_scores

test-4:
	$(PYTEST) tests/test_app.py::TestInterestsAPI::test_interests_count

test-5:
	$(PYTEST) tests/test_app.py::TestInterestsAPI::test_response_time

test-6:
	$(PYTEST) tests/test_app.py::TestInterestsAPI::test_with_probability_threshold

# No automated test — read the response and check every probability is >= 0.5.
test-7:
	@curl -fsS "http://localhost:5005/interests/$(HANDLE)?min_probability=0.5" \
		| python3 -m json.tool
