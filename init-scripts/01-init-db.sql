-- Enable the pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create articles table
CREATE TABLE articles (
  id SERIAL PRIMARY KEY,
  scraper_id INTEGER UNIQUE, -- Links to scraper's `id` for traceability
  title TEXT NOT NULL,
  content TEXT NOT NULL,
  pub_date TIMESTAMP, -- From scraper's `pub_date`
  domain TEXT, -- From scraper's `domain`
  processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  is_hot BOOLEAN DEFAULT FALSE, -- Flags top 20% "hottest" articles
  cluster_id INTEGER -- References `clusters` table (below)
);

-- Create entities table
CREATE TABLE entities (
  id SERIAL PRIMARY KEY,
  name TEXT NOT NULL, -- e.g., "X Corp", "Jack Ma"
  type TEXT NOT NULL, -- e.g., "corporation", "person", "government"
  influence_score FLOAT DEFAULT 0.0, -- Weighted score (mentions + context)
  mentions INTEGER DEFAULT 0, -- Count across articles
  first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  last_seen TIMESTAMP,
  UNIQUE (name, type)
);

-- Create article_entities junction table
CREATE TABLE article_entities (
  article_id INTEGER REFERENCES articles(id),
  entity_id INTEGER REFERENCES entities(id),
  mention_count INTEGER DEFAULT 1, -- How often entity appears in article
  PRIMARY KEY (article_id, entity_id)
);

-- Create clusters table
CREATE TABLE clusters (
  id SERIAL PRIMARY KEY,
  centroid VECTOR(768), -- Average embedding of cluster
  article_count INTEGER DEFAULT 0, -- Number of articles in cluster
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  is_hot BOOLEAN DEFAULT FALSE -- Top 10 clusters (20% of articles)
);

-- Create embeddings table
CREATE TABLE embeddings (
  id SERIAL PRIMARY KEY,
  article_id INTEGER UNIQUE REFERENCES articles(id),
  embedding VECTOR(768), -- Gemini embedding (e.g., 768 dimensions)
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create essays table
CREATE TABLE essays (
  id SERIAL PRIMARY KEY,
  type TEXT NOT NULL, -- "news_feed" or "rabbit_hole"
  article_id INTEGER REFERENCES articles(id), -- Source article (nullable for analytical)
  content TEXT NOT NULL, -- Paragraph or layer text
  layer_depth INTEGER, -- For Rabbit Hole: 1 (recap) to 5 (opinion)
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  tags TEXT[] -- e.g., ["China-US", "X Corp"]
);

-- Create essay_entities junction table
CREATE TABLE essay_entities (
  essay_id INTEGER REFERENCES essays(id),
  entity_id INTEGER REFERENCES entities(id),
  PRIMARY KEY (essay_id, entity_id)
);

-- Add foreign key constraint to articles table after clusters table is created
ALTER TABLE articles ADD CONSTRAINT fk_cluster FOREIGN KEY (cluster_id) REFERENCES clusters(id); 