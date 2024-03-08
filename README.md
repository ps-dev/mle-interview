# MLE Interview:


## Setup
1. Install [Visual Studio Code](https://code.visualstudio.com/). Also the following plugins:
    - [Dev Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)
2. Clone this repo
3. Open the repository in VSCode (`code .`)
4. VSCode will prompt you to install recommended extensions. Select `Yes`
    - If you missed this step, you can install recommended extensions using the extensions sidebar (`⇧⌘X`)
5. VSCode will prompt you to reopen the workspace in a Dev Container. Select `Yes`.
    - If you missed this step, open the command prompt palette (`⇧⌘P`) and run the command `Remote-Containers: Rebuild and Reopen in Container`


## Coding Challenge

1. **Model Training Error Resolution:**
   - First, address the model training error by executing `make test` to run the tests and identify any issues.
   - Once the `TestModel::test_model_build` test passes, proceed with `make train` to train and export the model. This step is crucial for preparing the model for subsequent processes.

2. **API Startup Issue Fix:**
   - To resolve the startup issue with the Flask API, execute `make serve-api`.
   - Carefully examine the logs for any error messages or indications of what might be causing the startup failure, and address those issues accordingly.

3. **Data Return and Interests Length Correction:**
   - Ensure that the expected return data is correctly structured and that the length of the interests array aligns with the specifications. Make adjustments to meet the expected outcomes.

4. **Response Time Optimization:**
   - Improve the response time to one second or less. This may involve identifying and resolving system bottlenecks, optimizing code, or improving resource allocation.

5. **Probability Field Inclusion:**
   - Modify the response body to include a 'probability' field. This field should provide users with additional insights by indicating the confidence level or likelihood associated with the response.

6. **Probability Score Filtering Implementation:**
   - Develop functionality that allows API clients to filter results based on the probability score. This feature should enable users to refine their results based on a specified probability threshold.
   
## Make Commands

Before you start, ensure you have all the necessary dependencies installed:

```
make setup
```

This command will install the Python dependencies listed in `requirements.txt`. 

### Serve API

To start the Flask API, use the following command:

```
make serve-api
```

This command launches the Flask application defined in `api/app` on port 5005, accessible via `0.0.0.0`. The `--debug` flag is included to enable debug mode, which provides useful debugging information in case of errors and automatically reloads the server on code changes.

### Serve Model

If you're using a TensorFlow model served via Docker, you can start the TensorFlow serving using:

```
make serve-model
```

This command uses Docker Compose to spin up a container defined in the Docker configuration, specifically targeting the TensorFlow serving service (`tf-serving`). Ensure Docker is installed and running on your system before executing this command.

### Train

To train the model, execute:

```
make train
```

This command runs the main training script located at `model.main`. 

### Test

Finally, to run the tests for your project, use:

```
make test
```

This command runs all the tests in the `./tests` directory using pytest. 


## API

Route: GET: http://localhost:5005/interests/:userHandle

API Response:

```
{
    "user_handle": "e337a675-46f5-437e-aa72-5d43643b5461",
    "interests": [
        {
            "id": "29e020bb-7a02-41db-b21f-527a8ef4dfdf",
            "label": "Accounting technician"
        },
        {
            "id": "012abcd1-3a6a-4803-a47e-42f46b402024",
            "label": "Field seismologist"
        },
        {
            "id": "a1b028dd-8464-4c63-85e8-ae29ea184fc7",
            "label": "Designer, industrial/product"
        },
        {
            "id": "686af7e4-6d58-4148-b227-3bf65ff10273",
            "label": "Materials engineer"
        },
        {
            "id": "8d1526b9-a7bf-4972-be43-7b912f149667",
            "label": "Fashion designer"
        },
        {
            "id": "d83a55a3-0143-4318-91c1-88f44ad59390",
            "label": "Journalist, newspaper"
        },
        {
            "id": "cda4441f-dba6-495c-9e2e-7429bd5e0465",
            "label": "Therapist, music"
        },
        {
            "id": "e9b23ad4-753b-4331-9cfa-525965fcf281",
            "label": "Secretary, company"
        },
        {
            "id": "ca4b03b2-0ae2-440a-afc8-5469510b19cb",
            "label": "Commissioning editor"
        },
        {
            "id": "fd3c41b8-8c15-47e2-a80d-cf3683b2d0da",
            "label": "Copywriter, advertising"
        }
    ],
    "name": "Kisiza",
    "type": "B2B"
}
```