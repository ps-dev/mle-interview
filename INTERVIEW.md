## Coding Challenge

You are working on a user interest recommendation system. It consists of:

- A **TensorFlow model** trained on user content interactions, served via TF Serving (Docker)
- A **Flask API** that fetches user features, calls the model, and returns ranked interests
- A **feature store** backed by CSV files

There are **6 tasks** to complete. Tasks 1–5 each have a failing test that tells you exactly what to fix. Task 6 is an open-ended feature implementation. You can refer to Google or relevant documentation — no AI-assisted coding tools.

---

### Terminal setup (do this before task 3)

Once the model is trained and served, you'll need **two terminals** running simultaneously inside the container:

| Terminal | Command | Purpose |
|---|---|---|
| 1 | `make serve-model` | TF Serving on port 8501 |
| 2 | `make serve-api` | Flask API on port 5005 |

Keep both running while working on tasks 3–6.

---

### Tasks

**1. Fix model training**

Run `make test`. The `TestModel::test_model_build` test fails.

Find and fix the bug in the model code, then verify:
```
make test  # TestModel::test_model_build must pass
```

Once passing, train the model and start the model server (leave it running):
```
make train
make serve-model
```

---

**2. Fix the API startup**

Run `make serve-api`. The Flask application fails to start.

Find and fix the bug, then verify the server starts without errors:
```
make serve-api
```

Leave the server running in its terminal.

---

**3. Fix the interests count**

With both servers running, run `make test`. This test fails:
```
TestInterestsAPI::test_basic_response — AssertionError: Incorrect number of interests
```

Fix the API so it returns the correct number of interests. Verify:
```
make test  # test_basic_response must pass (interests count assertion)
```

---

**4. Fix response time**

With both servers running, run `make test`. This test fails:
```
TestInterestsAPI::test_basic_response — AssertionError: Request greater than 1 second
```

Find the performance bottleneck and fix it. Verify:
```
make test  # test_basic_response must pass (timing assertion)
```

---

**5. Add probability to the response**

With both servers running, run `make test`. This test fails:
```
TestInterestsAPI::test_with_probability_threshold
```

Update the API response so each interest includes a `probability` field. Verify:
```
make test  # test_with_probability_threshold must pass
```

---

**6. Add probability threshold filtering**

Implement a feature that lets API clients filter results by a minimum probability score. For example:

```
GET /interests/<user_handle>?min_probability=0.5
```

Should return only interests with probability ≥ 0.5. There is no automated test for this task — demonstrate it works with a curl request.
