
# Part 2: Model Inference Server - Discussion

This document outlines the design, analysis, and conclusions for the simple model inference server created in `part2.py`.

You can start the server by running `python part2.py` in the command line. The server will be available at http://127.0.0.1:8000.

### 1. Motivation and High-Level Goal

The primary goal of this task was to take the trained PyTorch model (`fraud_prevention_model.pt`) and make it accessible for real-time predictions. A script that just runs and exits is insufficient for this; we need a persistent service that can respond to requests on demand.

The chosen solution was to build a **web API**, as this is the standard, language-agnostic way to expose a machine learning model's capabilities to other services (e.g., a web application, a mobile app, or another backend service).

I chose **FastAPI** as the web framework for the following key reasons:

*   **High Performance:** It is one of the fastest Python frameworks available. It's asynchronous which is great for I/O tasks and handling many concurrent connections.
*   **Ease of Use & Readability:** FastAPI makes the full use of Python type hints.
*   **Automatic Data Validation:** By using Pydantic models, FastAPI automatically validates incoming request data. It also defines the schema for the output data.
*   **Automatic Interactive Documentation:** FastAPI automatically generates interactive OpenAPI UI.

### 2. Solution Architecture (`server.py`)

The server is a single Python script that can be broken down into a few logical components:

**a. Data Models:**
The `TransactionFeatures` class defines the expected structure and data types for an incoming prediction request. It ensures that any `POST` request to our prediction endpoint must contain a JSON object with `amount` (float), `time_of_day` (int), `mismatch` (int), and `frequency` (int).

**b. Application and Model Initialisation:**
When the server starts, it loads the TorchScript model (`fraud_prevention_model.pt`) from disk into memory and sets it to evaluation mode (`model.eval()`).

This "load-on-startup" approach is important for performance. Loading a model can be a slow operation, and doing it for every prediction request would result in unacceptably high latency. By keeping the model in memory, it's always ready for immediate inference.

**c. The `/predict` Endpoint:**
This is the core of the API.
*   It's a `POST` requests.
*   It takes the validated `TransactionFeatures` as input.
*   **Inference Logic:**
    1.  The incoming data is converted into a PyTorch tensor.
    2.  Normalisation. The tensor is normalised. **This is the most critical step for correct model performance.** A model must be fed data that has the same statistical distribution as the data it was trained on. The server uses placeholder `mean` and `std` values, but in a production system, these exact statistics would be saved as artifacts during training and loaded here.
    3.  The normalised tensor is passed directly to the loaded model (`model(n_tensor)`).
    4.  Model output is converted to probability using a sigmoid function.
*   **Response:** It returns a clear, user-friendly JSON object containing the boolean prediction (`is_fraudulent`) and the `fraud_probability`.

### 3. Analysis and Conclusions

**a. Performance and Scalability:**
*   **Latency:** The prediction latency for a single request is very low, as the main work involves a simple tensor transformation and a forward pass through a small, in-memory neural network.
*   **Throughput:** Thanks to FastAPI's async nature, the server can handle a high number of concurrent requests. While one request is waiting for network I/O, the server can process another.
*   **Scaling Strategy:** The server is **stateless**, meaning each prediction request is self-contained. This is the key to scalability. To handle more traffic, one can simply run more instances of the server behind a load balancer. A common pattern is to use a process manager like **Gunicorn** to run multiple `uvicorn` worker processes on a single machine (`gunicorn -w 4 -k uvicorn.workers.UvicornWorker server:app`), allowing the application to utilise multiple CPU cores.

**b. Readability and Maintainability:**
The code is highly readable due to the declarative syntax of FastAPI and the use of type hints and Pydantic models. The logic is straightforward, and the automatic documentation means new developers can understand how to use the API without even reading the code.

### 4. Next Steps (If More Time Were Allocated)

This solution provides a solid foundation, but for a production-grade service, the following steps would be essential:

1.  **Save and Load Normalisation Artifacts:** Modify `create_model.py` to calculate and save the mean and standard deviation of the training features to a file (e.g., `normalisation_stats.json`). The server would then load this file at startup to ensure predictions are made with the correct, non-placeholder statistics.
2.  **Add data pre-processing pipeline:** Instead of normalising the tensor in the endpoint, we can create a data pre-processing pipeline that will be applied both before training and before inference.
3.  **Containerisation with Docker:** Package the server, the model file, and all Python dependencies into a Docker image. This creates a portable, reproducible artifact that can be deployed consistently anywhere Docker is running.
4.  **Configuration Management:** Move hardcoded values like the model path from the code into environment variables or a configuration file.
5.  **Logging:** Add logging to record details about each request, its prediction, and any errors. This is invaluable for monitoring the model's behavior and debugging issues in production.
6.  **Testing:** Add unit tests for the prediction logic and integration tests to verify the API endpoint's behavior.
