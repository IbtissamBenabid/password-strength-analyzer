```mermaid
sequenceDiagram
    participant Data as Data Source
    participant Process as Data Processing
    participant Train as Model Training
    participant Eval as Model Evaluation
    participant Save as Model Storage
    
    Data->>Process: Load Raw Data
    Process->>Process: Clean & Transform
    Process->>Train: Prepare Training Data
    Train->>Train: Train Model
    Train->>Eval: Evaluate Performance
    Eval->>Save: Save Model & Metrics
``` 