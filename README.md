# Hyper-Personalized AI Recommendation System

## 1. Project Overview
The Hyper-Personalized AI Recommendation System is an AI-powered furniture recommendation application designed to provide personalized product suggestions using user interactions, product information, semantic similarity, machine learning, and conversational AI.
The system helps users discover furniture based on category, brand, material, color, style, and budget.

## 2. Objectives
- Recommend relevant furniture products.
- Apply collaborative filtering using Alternating Least Squares (ALS).
- Generate semantic product embeddings and store them in ChromaDB.
- Combine recommendation signals using a hybrid recommendation pipeline and Learning-to-Rank (LTR).
- Support natural-language furniture searches through a conversational RAG interface.
- Provide a dashboard, REST APIs, and A/B testing simulation.
- Evaluate recommendation quality and response performance.

## 3. Technology Stack
- **Language:** Python
- **Dashboard:** Streamlit
- **API:** FastAPI
- **Machine Learning:** ALS and XGBoost
- **Embeddings:** Sentence Transformers
- **Vector Database:** ChromaDB
- **Conversational AI:** RAG and Ollama (Phi-3, as configured)
- **Data Processing:** Pandas, NumPy, SciPy
- **Evaluation:** Precision@K, NDCG@K, and simulated A/B testing

## 4. System Architecture
1. Product data and user interactions are loaded.
2. Data preprocessing prepares the recommendation inputs.
3. The collaborative filtering model learns from user interactions.
4. Sentence Transformer embeddings represent product descriptions.
5. ChromaDB retrieves semantically similar products.
6. The hybrid recommender combines recommendation signals and applies LTR ranking.
7. The conversational interface processes natural-language queries and retrieves relevant products.
8. FastAPI exposes recommendation endpoints.
9. The Streamlit dashboard provides model selection, filters, recommendations, chat, and experiment results.

## 5. Project Structure
```text
HyperPersonalizedAIRecommendationSystem/
│
├── data/
│   ├── Raw and processed product data
│   └── User interaction data
│
├── embeddings/
│   ├── evaluation/
│   │   ├── embedding_evaluation_results.xlsx
│   │   └── pca_info
│   └── embedding_metadata.json
│
├── models/
│   ├── Trained recommendation models
│   ├── LTR model artifacts
│   ├── A/B testing results
│   └── Evaluation outputs
│
├── src/
│   ├── ab_testing.py
│   ├── als_model.py
│   ├── api.py
│   ├── build_ltr_dataset.py
│   ├── chat_panel.py
│   ├── chatbot_ui.py
│   ├── content_recommender.py
│   ├── conversational_rag.py
│   ├── conversational_recommender.py
│   ├── dashboard.py
│   ├── data_cleaning_user_interactions.py
│   ├── embedder.py
│   ├── embedding_evaluation.py
│   ├── embedding_model.py
│   ├── etl_pipeline.py
│   ├── evaluation.py
│   ├── feature_engineering.py
│   ├── hybrid_recommender.py
│   ├── interaction_matrix.py
│   ├── langchain_retriever.py
│   ├── ltr_evaluation.py
│   ├── ltr_holdout_evaluation.py
│   ├── ltr_model.py
│   ├── ollama_rag_test.py
│   ├── optuna_ltr_tuning.py
│   ├── personalization_prompts.py
│   ├── products_dataset.py
│   ├── query_processor.py
│   ├── rag_context.py
│   ├── rag_evaluation.py
│   ├── rag_recommender.py
│   ├── recommendation_orchestrator.py
│   └── vector_store.py
│
├── vector_db/
│   └── ChromaDB vector database
│
├── ecommerce.db
├── run_dashboard.py
├── requirements.txt
└── README.md
```

### Main Components
- **Data processing:** ETL, data cleaning, interaction matrices, and feature engineering.
- **Collaborative filtering:** ALS-based user-item recommendations.
- **Content-based recommendations:** Sentence embeddings and ChromaDB similarity search.
- **Hybrid recommendation:** Collaborative filtering, content similarity, and LTR ranking.
- **Conversational AI:** RAG retrieval, context management, prompt templates, and chatbot interfaces.
- **Evaluation:** Embedding, recommendation, LTR, holdout, and RAG evaluation modules.
- **Optimization:** Optuna-based LTR hyperparameter tuning.
- **Application:** FastAPI service and Streamlit dashboard.
- **Experimentation:** Simulated A/B testing and saved experiment results.
The listed files represent the main project components; additional modules and generated artifacts may exist.


### Embeddings and Vector Database
The project uses Sentence Transformers to generate semantic
representations of furniture products.
Embedding metadata and evaluation results are included
in the repository.
Generated NumPy embedding arrays and the ChromaDB
database are excluded from GitHub because they can
be regenerated using the project scripts.
The vector database must be initialized before using
semantic retrieval and conversational RAG features.

## 6. Installation and Setup
### Prerequisites
- Python installed
- Git installed (optional if downloading ZIP)
- Ollama installed for local conversational AI functionality

### Step 1: Download the project
Download the repository ZIP from GitHub and extract it.
### Step 2: Open the project folder
cd HyperPersonalizedAIRecommendationSystem
### Step 3: Create a virtual environment
python -m venv .venv
### Step 4: Activate the environment
.venv\Scripts\activate
### Step 5: Install dependencies
pip install -r requirements.txt
### Step 6: Configure the required datasets,
database, embeddings and ChromaDB vector store.
Follow the data preparation and regeneration instructions
provided in the project.

## 7. Running the Application
Activate the virtual environment and run:
python run_dashboard.py
Alternatively, on Windows:
.\.venv\Scripts\python.exe run_dashboard.py

## 8. Recommendation Methods
### Popularity-Based Recommendation
Uses product interaction frequency as a baseline.
### Collaborative Filtering
Uses an ALS model trained on user-product interactions.
### Content-Based Recommendation
Uses semantic embeddings and ChromaDB similarity search.
### Hybrid Recommendation
Combines collaborative, content-based, and ranking signals.
### Conversational Recommendation
Processes natural-language requests and retrieves relevant furniture products through the configured RAG pipeline. Dining-table requests map to the catalog's broader `table` category because the catalog does not distinguish dining tables as a separate category. New general-furniture requests clear stale category and product filters; a previously specified budget is retained when the user refers to "my budget." Follow-up turns retain applicable preferences, support brand exclusions such as "not IKEA" or "other than IKEA," and normalize common wardrobe misspellings such as "waardroop."
After a no-match response, repeating only the same category (for example, "sofa") retries that category without inherited filters.
If requested category, color, material, or brand filters produce no exact
matches, the chatbot first reports that no exact matches were found, then may
show products in a separate **Alternative suggestions** section and disclose
the constraint each product does not meet. Alternatives relax one of those
filters at a time while preserving other filters, including excluded brands
and price limits. If none satisfy those remaining constraints, the chatbot
returns no products and explains that no suitable alternatives are available.

## 9. API Endpoints

The API includes the following recommendation endpoints:

| Method | Endpoint | Purpose |
|---|---|---|
| POST | `/recommend/popularity` | Popularity-based recommendations |
| POST | `/recommend/collaborative` | Collaborative filtering |
| POST | `/recommend/embedding` | Embedding-based recommendations |
| POST | `/recommend/hybrid` | Hybrid LTR recommendations |
| POST | `/recommend/conversational` | Conversational recommendations |
| GET | `/health` | Service health check |

Open `/docs` to inspect the actual request schemas and test endpoints interactively.
The conversational endpoint keeps exact matches in `products` and returns any
constraint-disclosed fallbacks separately in `alternative_products`.

## 10. Evaluation Results

### LTR Holdout Evaluation

The full-catalog holdout evaluation completed successfully.

| Metric | Observed result |
|---|---:|
| Precision@5 | 1.0000 |
| Precision@10 | 0.9800 |
| NDCG@5 | 1.0000 |
| NDCG@10 | 1.0000 |
| Users evaluated | 5 |
| Held-out positive interactions | 73 |

Results are saved in `models/ltr_holdout_results.csv`.

These are preliminary offline results. Only five users were evaluated, and unobserved products were treated as non-relevant. The metrics should not be interpreted as a production-quality guarantee.

### A/B Testing

The application includes a simulated comparison between a baseline recommender and a hybrid/RAG treatment.

The simulation tracks click-through-rate proxies and add-to-cart likelihood. These are simulated metrics, not measurements from real customer traffic.

Record the actual generated experiment results and output filenames when preparing the final report.

### Performance

Previously tested API calls returned non-RAG responses below 500 ms and a conversational response below 5 seconds in the tested local environment. Recheck performance before final submission.

### RAG Quality

Document response relevance, retrieval quality, personalization, hallucination checks, and latency after running the RAG evaluation. Do not report unmeasured values as verified results.

## 11. Personalization Prompts

The system should support specialized prompts for:

- Style-based recommendations
- Budget-based recommendations
- Category-based searches
- Color preferences
- Material preferences
- Room-based recommendations
- Brand preferences
- Comfort requirements
- Cross-selling complementary furniture
- Up-selling alternatives
- User-history personalization
- Multi-turn preference refinement

Verify that at least 10 prompt templates are implemented and tested in the existing project.

## 12. Testing

Recommended final checks:

1. Verify all recommendation methods.
2. Test category, material, color, and price filters.
3. Test multi-turn conversational queries.
4. Verify API status codes and response schemas.
5. Run the LTR holdout evaluation.
6. Run the RAG quality evaluation.
7. Verify A/B testing outputs.
8. Confirm the dashboard and API start successfully.

## 13. Limitations

- The holdout evaluation currently covers only five users.
- A/B testing is simulated and does not establish real-world business impact.
- Unobserved interactions may not represent true negative feedback.
- RAG quality and hallucination rates must be established using documented evaluation results.
- Localhost URLs are for development and are not public deployment links.

## 14. Future Improvements

- Evaluate on a larger, more representative dataset.
- Improve retrieval and conversational relevance.
- Validate recommendations with real user feedback.
- Add production monitoring, authentication, and rate limiting where required.
- Deploy the application to a suitable hosting platform.

## 15. Conclusion

The project combines collaborative filtering, semantic retrieval, hybrid ranking, conversational RAG, a REST API, and an interactive dashboard to support personalized furniture discovery. Further evaluation and documentation help establish the system's reliability and readiness for deployment.

## 16. Screenshots

Screenshots of the running application will be added here.

- Main dashboard: `screenshots/dashboard.png`
- Product recommendations: `screenshots/recommendations.png`
- Conversational RAG: `screenshots/conversational-rag.png`
- API documentation: `screenshots/api-docs.png`
