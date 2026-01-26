### **Implementation Modules and Data Artifacts**

* **`aml_agents.py`**: Encapsulates the core Agentic AI logic, utilizing the OpenAI SDK to define agent personas, tool-use, and reasoning chains.
* **`aml_classifier.py`**: Defines the REST API infrastructure, serving as the interface for transaction ingestion and classification output.
* **`aml_server.py`**: Orchestrates the application lifecycle, including server initialization and environmental configuration.
* **`Rules.txt`**: A natural language repository of AML regulatory rules, decoupled from the application logic to allow for dynamic updates and LLM interpretability.
* **`reviewer_results.txt`**: Contains execution logs specifically highlighting borderline cases where the primary classifier required secondary validation.
* **`reviewer_feedback_loop_triggered.txt`**: Documented logs illustrating the iterative feedback mechanism when the "Reviewer Agent" triggers a re-evaluation of the transaction state.