
## Coding Challenge

You can refer to Google or relevant documentation to guide you in addressing the issues below. 
You are not allowed to use ChatGPT, Copilot, or any similar tools for assistance. 

1. **Model Training Error Resolution:**
   - Begin by executing `make test` to identify any issues during model training.
   - After passing the `TestModel::test_model_build` test, use `make train` followed by `make serve-model` to train and serve the model, preparing it for upcoming steps.

2. **API Startup Issue Fix:**
   - Resolve the Flask API startup issue with `make serve-api`.
   - Run `make serve-api` to start a development Flask server, then open a new terminal window without shutting down the server.

3. **Data Return and Interests Length Correction:**
   - Execute `make test` and resolve the `TestInterestsAPI::test_basic_response - AssertionError: Incorrect number of interests` test.
   - Ensure the return data structure and interests array length meet the specifications, adjusting the API as necessary.

4. **Response Time Optimization:**
   - Run `make test` and fix the `TestInterestsAPI::test_basic_response - AssertionError: Request greater than 1 second` test.
   - Enhance the API to achieve a response time of one second or less.

5. **Probability Field Inclusion:**
   - Perform `make test` and correct the `TestInterestsAPI::test_with_probability` test.
   - Update the API response to incorporate a 'probability' field.

6. **Probability Score Filtering Implementation:**
   - Create a feature allowing API clients to filter results by probability score, enabling users to specify a probability threshold for more targeted results.