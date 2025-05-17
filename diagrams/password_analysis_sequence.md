```mermaid
sequenceDiagram
    participant User
    participant UI as Streamlit UI
    participant FE as Feature Extraction
    participant Model as ML Model
    participant Viz as Visualization
    
    User->>UI: Enter Password
    UI->>FE: Extract Features
    FE->>Model: Predict Strength
    Model->>UI: Return Prediction
    UI->>Viz: Generate Visualizations
    Viz->>UI: Display Results
    UI->>User: Show Analysis
``` 