# 🏪 Guide 01: Dual-Storage Feature Store Architecture & Point-in-Time Joins

## 1. The Online/Offline Dual-Storage Paradigm

In production ML systems, features are consumed in two fundamentally different environments with competing operational requirements:

```mermaid
flowchart TD
    subgraph FeaturePipeline[" Streaming & Batch Ingestion "]
        Stream["Kafka / Kinesis Stream<br/>(Real-Time Events)"]
        Batch["Snowflake / Spark Batch<br/>(Daily Aggregations)"]
    end

    subgraph DualStore[" Dual-Storage Feature Store "]
        Online["Online Store (Redis / DynamoDB)<br/>• Sub-5ms key-value lookups<br/>• Entity-keyed latest state"]
        Offline["Offline Store (Parquet / Iceberg / Delta)<br/>• Terabyte/Petabyte historical logs<br/>• Immutable append-only partitions"]
    end

    subgraph Consumers[" ML Consumers "]
        Serve["Model Serving Gateway<br/>(Real-Time Prediction)"]
        Train["Training Pipeline<br/>(Point-in-Time Historical Training Sets)"]
    end

    Stream --> Online
    Stream --> Offline
    Batch --> Offline
    Batch -->|Batch Ingestion Sync| Online
    Online --> Serve
    Offline --> Train

    style FeaturePipeline fill:#1e2327,stroke:#4c72b0,stroke-width:1.5px,color:#ffffff
    style DualStore fill:#1e2327,stroke:#22c55e,stroke-width:1.5px,color:#ffffff
    style Consumers fill:#1e2327,stroke:#f59e0b,stroke-width:1.5px,color:#ffffff
```

---

## 2. Preventing Data Leakage via Point-in-Time (As-Of) Joins

The most common cause of ML models failing in production after reporting stellar offline validation scores is **Data Leakage (Lookahead Bias)**:
* Training a fraud model using a customer's `chargeback_count_30d` computed *after* the fraudulent event occurred.

### The Point-in-Time Join Algorithm
For each observation event $E_j = (\text{entity\_id}_j, t_j)$ with observation timestamp $t_j$, the feature store extracts the most recent feature record $F_{i}$ satisfying:

$$\text{timestamp}(F_i) \le t_j$$

```mermaid
sequenceDiagram
    autonumber
    participant T as Timeline
    participant F as Feature Value Updates
    participant O as Observation Event

    Note over T: 10:00 AM - Feature Value = $100
    F->>T: Ingest (user_1, balance=$100, t=10:00)
    Note over T: 11:30 AM - Prediction Event (Loan Request)
    O->>T: Observation (user_1, t=11:30)
    T-->>O: As-Of Match: balance=$100 (Correct)
    Note over T: 12:00 PM - Feature Value = $500
    F->>T: Ingest (user_1, balance=$500, t=12:00)
    Note over O: Training Join never sees the $500 balance at 11:30!
```
