The app is a no-nonsense economic intelligence tool that cuts through the noise to reveal the real forces shaping the world. It’s not about chasing headlines—it’s about unmasking the corporations, billionaires, and governments pulling strings behind them. Focused on China-US trade, Vietnam-US trade, US political economy, global shifts, and figures like Jack Ma or Pham Nhat Vuong, it offers two feeds: a 30-minute News Feed of concise updates (20-30 paragraphs) and an infinite Rabbit Hole of layered analysis—recaps, theories, correlations, angles, and bold takes. Powered by 2,000 daily articles, Gemini embeddings, and a custom database, it’s sharp, skeptical, and built to zoom out on power plays over days, months, or years. Users get clarity and depth; you get a window into the puppet masters.

Workflow
Data Ingestion:
Source: Scraper’s database (articles table) delivers 2,000 new articles daily (WHERE scraped_at >= CURRENT_DATE - 1).

Transfer: App downloads articles, storing them in its articles table with scraper_id, title, content, pub_date, and domain.

Processing:
Entities: NER extracts power players (e.g., “X Corp,” “Jack Ma”) into entities (with influence_score, mentions) and links them via article_entities.

Embeddings: Gemini generates 768D vectors for each article, stored in embeddings for pattern detection.

Clustering: KMeans groups articles into 50-100 topics in clusters; top 10 (20%, ~400 articles) tagged is_hot for loud news.

Content Generation:
News Feed: 20-30 paragraphs written to essays (type = 'news_feed'):
30% (6-9) from is_hot clusters—e.g., “China’s exports drop 8%.”

70% (14-21) from background—e.g., “Vingroup buys US solar.”

Rabbit Hole: 740 essays (type = 'rabbit_hole')—240 real-time (10/hour), 500 analytical/day:
30% (222) flip hot news—e.g., “China’s drop—Z Fund’s gain?”

70% (518) dig into power players—e.g., “Vingroup’s quiet rise.”

Layers (1-5): Recap to “silenced observer” takes, stored with layer_depth.

Entity Links: essay_entities ties essays to players (e.g., “Z Fund” in 50 essays).

Delivery:
User Interface: News Feed scrolls vertically (30 min read); Rabbit Hole scrolls horizontally per item (5 min to hours).

Storage: Essays, entities, and embeddings in PostgreSQL, queried for real-time and analytical output.

Feedback Loop:
Yesterday’s 510 essays (7-day window, 3,570) feed context via embeddings similarity, chaining events—e.g., “Day 1’s X Corp cut links to today’s outage.”

Snapshot
Daily Input: 2,000 articles processed in ~1.5 hours (2 GPUs).

Output: 20-30 News Feed paragraphs, 740 Rabbit Hole essays (296,000 words).

Mission: 70% spotlight power players (e.g., “Z Fund’s 3-month play”), 30% decode hot news distractions.

Users: Quick facts, endless depth—pay to see the real game.

This is your app: a lean, mean truth machine—exposing the powerful for you, captivating users to fund it.
