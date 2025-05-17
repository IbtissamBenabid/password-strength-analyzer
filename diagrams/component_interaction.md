```mermaid
graph LR
    subgraph Frontend
        UI[Streamlit UI]
        Viz[Visualizations]
    end
    
    subgraph Backend
        FE[Feature Extraction]
        ML[ML Model]
        PG[Password Generator]
    end
    
    subgraph Data
        Model[Model Files]
        Metrics[Performance Metrics]
    end
    
    UI --> FE
    FE --> ML
    ML --> Viz
    UI --> PG
    PG --> ML
    ML --> Model
    Model --> Metrics
    Metrics --> Viz
``` 