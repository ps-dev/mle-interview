## Coding Challenge

You are working on a user interest recommendation system. It consists of:

- A **TensorFlow model** trained on user content interactions, served via TF Serving (Docker)
- A **Flask API** that fetches user features, calls the model, and returns ranked interests
- A **feature store** backed by CSV files

There are **7 tasks** to complete. Tasks 1, 3, 4, 5 and 6 each have a failing test that tells you what to fix. Tasks 2 and 7 are verified by running the app. You can refer to Google or relevant documentation — no AI-assisted coding tools.

---

### Running the tests

**Every task has its own command** — `make test-1` through `make test-7`. Run only the one for the task you are on:

```
make test-4
```

`make test` runs everything at once. Avoid it until the end: tests for tasks you have not reached yet will fail and bury the output you care about.

**Read the last few lines of the output.** Each test failure ends with the assertion message that names exactly what is wrong, for example:

```
E   AssertionError: Incorrect number of interests
```

That message is the whole signal — you do not need to read the traceback above it.

**The tests are correct.** Nearly every fix belongs in the source code, in `api/` or `model/` — not in `tests/`. **Task 4 is the only exception:** `top_k` has to be passed into the request as a query parameter, so that test does change.

---

### Terminal setup (do this before task 4)

Once the model is trained and served, you'll need **two terminals** running simultaneously inside the container:

| Terminal | Command | Purpose |
|---|---|---|
| 1 | `make serve-model` | TF Serving on port 8501 |
| 2 | `make serve-api` | Flask API on port 5005 |

Keep both running while working on tasks 4–7.

---

### Tasks

**1. Fix model training**

Run `make test-1`. The test fails.

Find and fix the bug in the model code, then verify:
```
make test-1
```

Once passing, train the model and start the model server (leave it running):
```
make train
make serve-model
```

---

**2. Fix the API startup**

Run `make serve-api`. The Flask application fails to start.

Find and fix the bug, then start the server again and leave it running in its terminal:
```
make serve-api
```

From a second terminal, verify:
```
make test-2
```

---

**3. Fix the probability scores**

The model's serving function returns raw logit scores, not probabilities. These values are not bounded between 0 and 1 and cannot be meaningfully compared or filtered.

Find where the issue originates and fix it so that the scores returned represent proper probability values. There are two valid approaches — both are acceptable.

Verify:
```
make test-3
```

This test exercises the model in memory. For the **served** model to return probabilities too, re-run `make train` and restart `make serve-model`.

---

**4. Fix the interests count**

With both servers running, run `make test-4`. It fails with:
```
AssertionError: Incorrect number of interests
```

The test asks for 15 interests but never tells the API how many it wants. Make the request send `top_k`, and make sure the API honours it. Verify:
```
make test-4
```

This is the one task where you edit a test.

---

**5. Fix response time**

With both servers running, run `make test-5`. It fails with:
```
AssertionError: Request greater than 1 second
```

Find the performance bottleneck and fix it. Verify:
```
make test-5
```

---

**6. Add probability to the response**

With both servers running, run `make test-6`. It fails with:
```
AssertionError: Interest object missing 'probability': ...
```

Update the API response so each interest includes a `probability` field. Verify:
```
make test-6
```

---

**7. Add probability threshold filtering**

Implement a feature that lets API clients filter results by a minimum probability score. For example:

```
GET /interests/<user_handle>?min_probability=0.5
```

Should return only interests with probability ≥ 0.5. There is no automated test for this task — check the response yourself:
```
make test-7
```
