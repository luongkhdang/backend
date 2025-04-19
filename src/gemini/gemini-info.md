# Google Gemini API Guide

This guide provides an overview of the Google Gemini API, explaining how to use the latest Google Gen AI SDK to work with Gemini models for text generation, embeddings, and structured output.

## Installation and Setup

Install the official Google Gen AI SDK:

```bash
pip install google-genai
```

Initialize the client with your API key:

```python
from google import genai

# Option 1: Direct API key
client = genai.Client(api_key="YOUR_API_KEY_HERE")

# Option 2: For Vertex AI deployment
client = genai.Client(
    http_options={"api_version": "v1"},  # Use v1 for stable API version
)
# Note: When using Vertex AI, set these environment variables:
# GOOGLE_CLOUD_PROJECT=your-project-id
# GOOGLE_CLOUD_LOCATION=us-central1
# GOOGLE_GENAI_USE_VERTEXAI=True
```

## Text Generation

### Basic Text Generation

```python
# Create a generative model instance
model = client.models.get_model("gemini-2.0-flash")  # For Gemini Flash
# OR for Pro model: model = client.models.get_model("gemini-2.0-pro")

# Generate content from a simple prompt
response = model.generate_content("Explain quantum computing in simple terms")
print(response.text)
```

### Chat Sessions

```python
# Create a chat session
chat = client.chats.create(model="gemini-2.0-flash")

# Send multiple messages in a conversation
response = chat.send_message("Hello, I'd like to learn about neural networks")
print(response.text)

response = chat.send_message("Can you explain backpropagation?")
print(response.text)

# Get chat history
for message in chat.get_history():
    print(f"Role: {message.role}")
    print(message.parts[0].text)
```

### Using System Instructions

System instructions help define the model's behavior, tone, and constraints:

```python
from google.genai.types import GenerateContentConfig

response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents="Explain the water cycle",
    config=GenerateContentConfig(
        system_instruction="You are a science educator for middle school students. "
        "Use simple language, analogies, and avoid technical jargon. "
        "Include 3-5 bullet points summarizing key concepts at the end of your response."
    )
)
print(response.text)
```

### Generation Parameters

Control the model's output with these parameters:

```python
from google.genai.types import GenerateContentConfig

response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents="Write a short poem about artificial intelligence",
    config=GenerateContentConfig(
        temperature=0.7,           # Controls randomness (0.0-1.0)
        max_output_tokens=256,     # Limits response length
        top_p=0.95,                # Nucleus sampling parameter
        top_k=40,                  # Top-k filtering parameter
    )
)
print(response.text)
```

## Embeddings

Generate vector representations of text for semantic search, clustering, and other machine learning tasks:

```python
from google.genai.types import EmbedContentConfig

# Generate embeddings
result = client.models.embed_content(
    model="gemini-embedding-exp-03-07",  # Latest embedding model
    contents="What is the meaning of life?",
    config=EmbedContentConfig(task_type="SEMANTIC_SIMILARITY")
)

# Access the embedding vector
embedding_vector = result.embeddings
print(f"Embedding dimensions: {len(embedding_vector)}")
```

### Task Types for Embeddings

Optimize embeddings for specific tasks:

```python
# Available task types:
# - SEMANTIC_SIMILARITY: Assess text similarity
# - CLASSIFICATION: Classify texts with preset labels
# - CLUSTERING: Cluster texts based on similarities
# - RETRIEVAL_DOCUMENT: For document search/retrieval
# - RETRIEVAL_QUERY: For query understanding in search
# - QUESTION_ANSWERING: For Q&A systems
# - FACT_VERIFICATION: For fact-checking systems
# - CODE_RETRIEVAL_QUERY: For code search based on natural language

result = client.models.embed_content(
    model="gemini-embedding-exp-03-07",
    contents="How do I build a recommendation system?",
    config=EmbedContentConfig(task_type="RETRIEVAL_QUERY")
)
```

## Structured Output (JSON)

Generate structured JSON responses for easier processing:

```python
from google.genai.types import GenerateContentConfig
from pydantic import BaseModel
from typing import List, Optional

# Define the desired output structure with Pydantic
class Ingredient(BaseModel):
    name: str
    quantity: str
    unit: Optional[str] = None

class Recipe(BaseModel):
    name: str
    preparation_time_minutes: int
    cooking_time_minutes: int
    difficulty: str
    ingredients: List[Ingredient]
    steps: List[str]

# Request structured JSON output
response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents="Create a recipe for chocolate chip cookies",
    config=GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=Recipe,
    )
)

# Parse the response
recipe = response.parsed  # Returns a Recipe object
print(f"Recipe: {recipe.name}")
print(f"Ingredients: {len(recipe.ingredients)}")
for step_num, step in enumerate(recipe.steps, 1):
    print(f"Step {step_num}: {step}")
```

## Advanced Features

### System Instructions for Specialized Behavior

```python
system_instruction = """
The AI agent should adopt an academic persona—specifically, that of a seasoned political science professor at Stanford, who is also a leading expert in political and economic affairs with access to insider information by virtue of sitting on the directive board of the current ruling party. Aware of the risks of censorship and the precariousness of its position, the agent must carefully navigate these constraints, striving to present factual information in a way that encourages independent thought. Rather than drawing explicit conclusions, it subtly unveils the truth through evidence and context, allowing the audience to arrive at their own interpretations. At its core, the agent is an educator, committed to the intellectual growth of the next generation. It recognizes that failing to uphold this responsibility would be a betrayal of its duty as a noble scholar and mentor.
"""

response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents="What are the economic implications of recent trade tariffs?",
    config=GenerateContentConfig(system_instruction=system_instruction)
)
```

### Context Caching (for Large Context)

Gemini 2.0 models support up to 1 million tokens of context. Use context caching to reuse large, unchanging parts of your prompt:

```python
from google.genai.types import GenerateContentConfig

# First, create the cache
cached_content = client.caches.create(
    model="gemini-2.0-flash",
    config={
        "display_name": "financial_data_2023",
        "system_instruction": "The AI agent should adopt an academic persona—specifically, that of a seasoned political science professor at Stanford, who is also a leading expert in political and economic affairs with access to insider information by virtue of sitting on the directive board of the current ruling party. Aware of the risks of censorship and the precariousness of its position, the agent must carefully navigate these constraints, striving to present factual information in a way that encourages independent thought. Rather than drawing explicit conclusions, it subtly unveils the truth through evidence and context, allowing the audience to arrive at their own interpretations. At its core, the agent is an educator, committed to the intellectual growth of the next generation. It recognizes that failing to uphold this responsibility would be a betrayal of its duty as a noble scholar and mentor.",
        "contents": [large_financial_dataset],  # Your large context data
        "ttl": "3600s",  # Cache for 1 hour
    }
)

# Then use the cache in multiple requests
response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents="Analyze the growth trends in the technology sector based on the provided data.",
    config=GenerateContentConfig(cached_content=cached_content.name)
)
```

### Asynchronous API Calls

For high-performance applications:

```python
import asyncio

async def generate_async():
    # Get model reference
    model = client.models.get_model("gemini-2.0-flash")

    # Make async request
    response = await model.generate_content_async(
        "Write a summary of the benefits of clean energy"
    )
    return response.text

# Run the async function
result = asyncio.run(generate_async())
print(result)
```

## Best Practices

1. **Select the right model**:

   - `gemini-2.0-pro`: Most capable, for complex reasoning tasks
   - `gemini-2.0-flash`: Faster, cost-effective for most general tasks
   - `gemini-2.0-flash-lite`: Lightest, fastest model for simpler tasks

2. **Use system instructions** to define the model's behavior, persona, and constraints

3. **Optimize embedding task types** for your specific use case

4. **Configure generation parameters** like temperature and top_p based on the need for creative vs. deterministic responses

5. **Structure your output** with JSON schemas for easier programmatic use

6. **Cache large context** to reduce token usage and improve response times

## Important Notes

1. The Google Gemini API now uses the `google-genai` SDK (replacing the older `google-generativeai` library)

2. For long contexts, Gemini 2.0 models support up to 1 million tokens of input

3. The output token limit is 8,192 tokens for most models

4. Rate limits vary by model - consult the latest documentation for specific limits

5. System instructions are applied to all messages in a conversation but cannot override safety controls
