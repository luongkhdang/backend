Calculates a weighted hotness score for each cluster and determines which
clusters are "hot" based on ranking the top N.

    Factors considered:
    1. Size: Logarithmic count of articles in the cluster.
    2. Recency: Proportion of articles published within RECENCY_DAYS.
    3. Entity Influence: Average influence score of entities linked to cluster articles.
    4. Topic Relevance: Whether cluster keywords match CORE_TOPIC_KEYWORDS.
    5. Tiered Persistence: Bonus if similar clusters existed 1, 7, or 30 days ago.
    6. Fading Penalty: Negative adjustment if a similar cluster was hot 7 days ago
                       but has significantly shrunk in size.

recency_days = int(os.getenv("RECENCY_DAYS", "3"))
calc_relevance = os.getenv("CALCULATE_TOPIC_RELEVANCE", "true").lower() == "true"
core_keywords_str = os.getenv("CORE_TOPIC_KEYWORDS", "")
core_keywords = {kw.strip().lower() for kw in core_keywords_str.split(',') if kw.strip()} if core_keywords_str else set()
cluster_sample_size = int(os.getenv("CLUSTER_SAMPLE_SIZE", "10")) # For keyword extraction

        w_size = float(os.getenv("W_SIZE", "0.15"))
        w_recency = float(os.getenv("W_RECENCY", "0.30"))
        w_influence = float(os.getenv("W_INFLUENCE", "0.30"))
        w_relevance = float(os.getenv("W_RELEVANCE", "0.15")) # Adjusted for persistence/penalty
        w_daily_p = float(os.getenv("W_DAILY_P", "0.05"))
        w_weekly_p = float(os.getenv("W_WEEKLY_P", "0.03"))
        w_monthly_p = float(os.getenv("W_MONTHLY_P", "0.02"))
        w_fading_penalty = float(os.getenv("W_FADING_PENALTY", "-0.05")) # Note: Negative weight

        persistence_threshold = float(os.getenv("PERSISTENCE_SIMILARITY_THRESHOLD", "0.90"))
        downward_trend_factor = float(os.getenv("DOWNWARD_TREND_FACTOR", "0.75"))
        target_hot_clusters = int(os.getenv("TARGET_HOT_CLUSTERS", "8"))

Conceptual Logic Flow:

Configuration Loading: Read configuration parameters from environment variables:

Weights for each factor: Size (W_SIZE), Recency (W_RECENCY), Influence (W_INFLUENCE), Relevance (W_RELEVANCE), Daily Persistence (W_DAILY_P), Weekly Persistence (W_WEEKLY_P), Monthly Persistence (W_MONTHLY_P), Fading Penalty (W_FADING_PENALTY - negative value).
Thresholds/Settings: RECENCY_DAYS (for Recency factor), PERSISTENCE_SIMILARITY_THRESHOLD (for historical matching), DOWNWARD_TREND_FACTOR (for Fading Penalty), TARGET_HOT_CLUSTERS (number of top clusters to mark hot).
Relevance settings: CALCULATE_TOPIC_RELEVANCE (boolean flag), CORE_TOPIC_KEYWORDS list, CLUSTER_SAMPLE_SIZE (for keyword extraction).
Historical Data Retrieval: Use the database client to fetch relevant information about clusters from previous runs:

Centroids, is_hot status, and article_count from the most recent run on the previous day.
Centroids, is_hot status, and article_count from the most recent run ~7 days ago.
Centroids from the most recent run ~30 days ago.
Entity Influence Pre-computation: Use the database client to fetch the average entity influence score for all relevant articles identified in the current clustering run (bulk query for efficiency).

Per-Cluster Score Calculation (Iterate through current non-noise clusters):

For each cluster label:
Identify the list of database article IDs belonging to this cluster.
Calculate Raw Scores:
Size: Calculate score based on the number of articles (e.g., log(count + 1)).
Recency: Calculate the proportion of articles within the cluster published in the last RECENCY_DAYS.
Influence: Calculate the average entity influence score for articles in this cluster (using the pre-computed scores).
Relevance (if enabled): Extract keywords from a sample of cluster articles (using the NLP model and database client); assign a score (e.g., 1 or 0) based on matches with CORE_TOPIC_KEYWORDS.
Persistence (Daily, Weekly, Monthly): Compare the current cluster's centroid to the respective historical centroids. Assign a score (e.g., 1 or 0) for each period if similarity exceeds PERSISTENCE_SIMILARITY_THRESHOLD.
Fading Penalty: Check if a similar cluster existed 7 days ago, if it was_hot then, and if the current article_count is significantly lower (based on DOWNWARD_TREND_FACTOR) than the historical count. If all true, assign a penalty score (e.g., 1), otherwise 0.
Store all these raw scores associated with the cluster label.
Score Normalization: Across all clusters calculated in step 4, apply min-max normalization to the raw scores for Size, Recency, and Influence to scale them between 0 and 1. (Binary scores like Relevance and Persistence generally don't need normalization here).

Final Weighted Score Calculation: For each cluster label, calculate the final hotness_score by summing the normalized (or binary) scores multiplied by their respective weights (including the negative weight for the fading penalty). Ensure the final score is not less than zero.

Ranking and Top N Selection:

Rank all clusters based on their final hotness_score in descending order.
Identify the top TARGET_HOT_CLUSTERS from the ranking.
Output Generation: Create the output dictionary, mapping each cluster label to True if it's in the top N, and False otherwise.

Logging: Throughout the process, log key configuration values, progress updates, potential errors (e.g., missing historical data, errors fetching influence), score distributions (min/max/avg for raw and normalized factors), and the final ranking/selection for monitoring and tuning purposes.
