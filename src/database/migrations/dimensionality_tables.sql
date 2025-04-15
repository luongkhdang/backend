-- dimensionality_tables.sql
-- Creates tables for storing dimensionality reduction data (UMAP, t-SNE, PCA)

-- Table for storing coordinates for each article
CREATE TABLE IF NOT EXISTS article_coordinates (
    id SERIAL PRIMARY KEY,
    article_id INTEGER NOT NULL,
    method VARCHAR(50) NOT NULL,  -- 'umap', 'tsne', 'pca', etc.
    coordinates JSONB NOT NULL,   -- Stores 2D or 3D coordinates as JSON array
    metadata JSONB,               -- Additional metadata about the reduction process
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
    
    -- Each article can have only one set of coordinates per method
    CONSTRAINT article_method_unique UNIQUE (article_id, method),
    
    -- Foreign key to articles table
    CONSTRAINT fk_article_coordinates_article
        FOREIGN KEY (article_id)
        REFERENCES articles (id)
        ON DELETE CASCADE
);

-- Create indexes for faster lookups
CREATE INDEX IF NOT EXISTS idx_article_coordinates_article_id ON article_coordinates (article_id);
CREATE INDEX IF NOT EXISTS idx_article_coordinates_method ON article_coordinates (method);

-- Table for storing global configurations for each dimensionality reduction method
CREATE TABLE IF NOT EXISTS dim_reduction_config (
    id SERIAL PRIMARY KEY,
    method VARCHAR(50) NOT NULL,  -- 'umap', 'tsne', 'pca', etc.
    config JSONB NOT NULL,        -- Configuration parameters for the method
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
    
    -- Each method can have only one global configuration
    CONSTRAINT method_unique UNIQUE (method)
);

-- Add comment to describe tables
COMMENT ON TABLE article_coordinates IS 'Stores coordinates from dimensionality reduction (UMAP, t-SNE, PCA) for visualization';
COMMENT ON TABLE dim_reduction_config IS 'Stores global configurations for dimensionality reduction methods';

-- Add timestamp trigger for automatic update of updated_at
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create trigger for article_coordinates
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_trigger WHERE tgname = 'trigger_update_article_coordinates_updated_at'
    ) THEN
        CREATE TRIGGER trigger_update_article_coordinates_updated_at
        BEFORE UPDATE ON article_coordinates
        FOR EACH ROW
        EXECUTE FUNCTION update_updated_at_column();
    END IF;
END
$$;

-- Create trigger for dim_reduction_config
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_trigger WHERE tgname = 'trigger_update_dim_reduction_config_updated_at'
    ) THEN
        CREATE TRIGGER trigger_update_dim_reduction_config_updated_at
        BEFORE UPDATE ON dim_reduction_config
        FOR EACH ROW
        EXECUTE FUNCTION update_updated_at_column();
    END IF;
END
$$; 