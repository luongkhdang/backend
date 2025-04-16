article_entities table:
article_id
entity_id
mention_count

articles:
id
scraper_id
title
content
pub_date
domain
processed_at
is_hot
cluster_id

calculated_domain_goodness table:
domain
domain_goodness_score
calculated_at

clusters:
id
centroid
article_count
created_at
is_hot
metadata
hotness_score

embeddings:
id
article_id
embedding
created_at

entities:
id
name
type
influence_score
mentions
first_seen
last_seen
entity_type

essay_entities:
essay_id
entity_id

essays:
id
type
article_id
content
layer_depth
created_at
tags
