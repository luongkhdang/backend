App Database Structure
The app’s database will support:
Processed Articles: Cleaned-up content from the scraper, ready for analysis.

Entities: Power players (people, orgs) extracted and ranked by influence.

Embeddings: Gemini-generated vectors for clustering and correlations.

Essays: News Feed paragraphs and Rabbit Hole layers, linked to articles and entities.

Metadata: Timestamps and tags for tracking and retrieval.

Here’s the schema, built in PostgreSQL for its robustness and full-text search capabilities.

1. articles Table
   Stores processed articles from the scraper, stripped to essentials and enriched with app-specific fields.
   sql

CREATE TABLE articles (
id SERIAL PRIMARY KEY,
scraper_id INTEGER UNIQUE, -- Links to scraper’s `id` for traceability
title TEXT NOT NULL,
content TEXT NOT NULL,
pub_date TIMESTAMP, -- From scraper’s `pub_date`
domain TEXT, -- From scraper’s `domain`
processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
is_hot BOOLEAN DEFAULT FALSE, -- Flags top 20% “hottest” articles
cluster_id INTEGER -- References `clusters` table (below)
);

Why: Keeps the raw content (title, content) but adds is_hot (30% loud news) and cluster_id for topic grouping. scraper_id ties back to the source without duplicating messy fields like url or error_message.

2. entities Table

CREATE TABLE entities (
id SERIAL PRIMARY KEY,
name TEXT NOT NULL, -- e.g., “X Corp”, “Jack Ma”
type TEXT NOT NULL, -- e.g., “corporation”, “person”, “government”
influence_score FLOAT DEFAULT 0.0, -- Weighted score (mentions + context)
mentions INTEGER DEFAULT 0, -- Count across articles
first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
last_seen TIMESTAMP,
UNIQUE (name, type)
);

Why: Central hub for power players. influence_score combines frequency (mentions) and context (e.g., “owns $50B”)—your lens on who matters. first_seen/last_seen track their timeline for long-term patterns.

1. article_entities Table (Junction)
   Links articles to entities for many-to-many relationships.
   sql

CREATE TABLE article_entities (
article_id INTEGER REFERENCES articles(id),
entity_id INTEGER REFERENCES entities(id),
mention_count INTEGER DEFAULT 1, -- How often entity appears in article
PRIMARY KEY (article_id, entity_id)
);

Why: An article might name “X Corp” and “Z Fund” multiple times—mention_count weights their relevance. This fuels correlations (e.g., “Z Fund + X Corp in 50 articles”).

4. embeddings Table
   Stores Gemini-generated embeddings for articles, enabling clustering and similarity searches.
   sql

CREATE TABLE embeddings (
id SERIAL PRIMARY KEY,
article_id INTEGER UNIQUE REFERENCES articles(id),
embedding VECTOR(768), -- Gemini embedding (e.g., 768 dimensions)
created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

Why: Gemini embeddings (assume 768 dimensions, adjustable) turn article content into vectors for clustering (hot vs. background) and finding patterns (e.g., “Vietnam deal” links to “China dip”). VECTOR type (via PostgreSQL’s pgvector extension) supports fast similarity queries.

5. clusters Table
   Groups articles by topic for hot/background split and essay allocation.
   sql

CREATE TABLE clusters (
id SERIAL PRIMARY KEY,
centroid VECTOR(768), -- Average embedding of cluster
article_count INTEGER DEFAULT 0, -- Number of articles in cluster
created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
is_hot BOOLEAN DEFAULT FALSE -- Top 10 clusters (20% of articles)
);

Why: After clustering 2,000 articles (e.g., KMeans on embeddings), centroid defines the topic, article_count sizes it, and is_hot flags the loud 30%. Links to articles.cluster_id.

6. essays Table
   Stores News Feed paragraphs and Rabbit Hole layers as distinct entries.
   sql

CREATE TABLE essays (
id SERIAL PRIMARY KEY,
type TEXT NOT NULL, -- “news_feed” or “rabbit_hole”
article_id INTEGER REFERENCES articles(id), -- Source article (nullable for analytical)
content TEXT NOT NULL, -- Paragraph or layer text
layer_depth INTEGER, -- For Rabbit Hole: 1 (recap) to 5 (opinion)
created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
tags TEXT[] -- e.g., [“China-US”, “X Corp”]
);

Why: type splits News Feed (20-30/day) from Rabbit Hole (740/day). layer_depth orders Rabbit Hole stages (1-5). tags aid retrieval and user filtering (e.g., “Show me Jack Ma”).

7. essay_entities Table (Junction)
   Links essays to entities for tracking power players in content.
   sql

CREATE TABLE essay_entities (
essay_id INTEGER REFERENCES essays(id),
entity_id INTEGER REFERENCES entities(id),
PRIMARY KEY (essay_id, entity_id)
);

Why: Ties “Z Fund” to an essay about “X Corp’s outage”—key for your mission to spotlight influence.
