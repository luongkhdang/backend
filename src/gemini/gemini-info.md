Before you begin
Before calling the Gemini API, ensure you have your SDK of choice installed, and a Gemini API key configured and ready to use.

Python
We provide a Python SDK which you can install by running:
pip install google-genai

Generate embeddings
Use the embedContent method to generate text embeddings:
from google import genai

client = genai.Client(api_key="GEMINI_API_KEY")

result = client.models.embed_content(
model="gemini-embedding-exp-03-07",
contents="What is the meaning of life?")

print(result.embeddings)
You can also generate embeddings for multiple chunks at once by passing them in as a list of strings.

Task types
When building Retrieval Augmented Generation (RAG) systems, a common design is to use text embeddings to perform a similarity search. In some cases this can lead to degraded quality, because questions and their answers are not semantically similar. For example, a question like "Why is the sky blue?" and its answer "The scattering of sunlight causes the blue color," have distinctly different meanings as statements, which means that a RAG system won't automatically recognize their relation.

Task types enable you to generate optimized embeddings for specific tasks, saving you time and cost and improving performance.

from google import genai
from google.genai import types

client = genai.Client(api_key="GEMINI_API_KEY")

result = client.models.embed_content(
model="text-embedding-004",
contents="What is the meaning of life?",
config=types.EmbedContentConfig(task_type="CLUSTERING")
)
print(result.embeddings)
Supported task types
Task type Description
SEMANTIC_SIMILARITY Used to generate embeddings that are optimized to assess text similarity.
CLASSIFICATION Used to generate embeddings that are optimized to classify texts according to preset labels.
CLUSTERING Used to generate embeddings that are optimized to cluster texts based on their similarities.
RETRIEVAL_DOCUMENT, RETRIEVAL_QUERY, QUESTION_ANSWERING, and FACT_VERIFICATION Used to generate embeddings that are optimized for document search or information retrieval.
CODE_RETRIEVAL_QUERY Used to retrieve a code block based on a natural language query, such as sort an array or reverse a linked list. Embeddings of the code blocks are computed using RETRIEVAL_DOCUMENT.

Text generation

The Gemini API can generate text output in response to various inputs, including text, images, video, and audio. This guide shows you how to generate text using text and image inputs. It also covers streaming, chat, and system instructions.

Text input
The simplest way to generate text using the Gemini API is to provide the model with a single text-only input, as shown in this example:

from google import genai

client = genai.Client(api_key="GEMINI_API_KEY")

response = client.models.generate_content(
model="gemini-2.0-flash",
contents=["How does AI work?"]
config=types.GenerateContentConfig(
system_instruction="You are a neutral politic scientist from Harvard. You are closely involved with in-charge political party (2025). You pay attention to the unexpected correlation in things/events but don't rush to assumpt. You know that being too straightfoward, in your current position, will bring trouble, so you like to weave through the censorship by providing facts and connections without a clear conclusion so the audience can come up with their own own.",
max_output_tokens=8192,
cached_content=cache.name,
temperature=1.2
)
print(response.text)

Long context
!IMPORANT: FOR THIS PROJECT, IT IS HIGHLY RECOMMENDED TO FULLY UTILIZE ALL 1-MILLION-TOKEN OF THE INPUT CONTEXT WINDOW. BUT KEEP IN MIND THAT THE OUTPUT WINDOW IS ONLY 8192 TOKENS
Gemini 2.0 Flash come with a 1-million-token context window




CACHE:

# Create a cache with a 5 minute TTL
cache = client.caches.create(
    model=model,
    config=types.CreateCachedContentConfig(
      display_name='sherlock jr movie', # used to identify the cache
      system_instruction=(
          'You are an expert video analyzer, and your job is to answer '
          'the user\'s query based on the video file you have access to.'
      ),
      contents=[video_file],
      ttl="300s",
  )
)

# Construct a GenerativeModel which uses the created cache.
response = client.models.generate_content(
  model = model,
  contents= (
    'Introduce different characters in the movie by describing '
    'their personality, looks, and names. Also list the timestamps '
    'they were introduced for the first time.'),
  config=types.GenerateContentConfig(cached_content=cache.name)
)
Update a cache
You can set a new ttl or expire_time for a cache. Changing anything else about the cache isn't supported.

The following example shows how to update the ttl of a cache using client.caches.update().


from google import genai
from google.genai import types

client.caches.update(
  name = cache.name,
  config  = types.UpdateCachedContentConfig(
      ttl='300s'
  )
)
To set the expiry time, it will accepts either a datetime object or an ISO-formatted datetime string (dt.isoformat(), like 2025-01-27T16:02:36.473528+00:00). Your time must include a time zone (datetime.utcnow() doesn't attach a time zone, datetime.now(datetime.timezone.utc) does attach a time zone).


from google import genai
from google.genai import types
import datetime

# You must use a time zone-aware time.
in10min = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(minutes=10)

client.caches.update(
  name = cache.name,
  config  = types.UpdateCachedContentConfig(
      expire_time=in10min
  )
)
Delete a cache
The caching service provides a delete operation for manually removing content from the cache. The following example shows how to delete a cache:


client.caches.delete(cache.name)

Generate structured output
Gemini generates unstructured text by default, but some applications require structured text. For these use cases, you can constrain Gemini to respond with JSON, a structured data format suitable for automated processing. You can also constrain the model to respond with one of the options specified in an enum.
Generate JSON
When the model is configured to output JSON, it responds to any prompt with JSON-formatted output.

You can control the structure of the JSON response by supplying a schema. There are two ways to supply a schema to the model:

As text in the prompt
As a structured schema supplied through model configuration
Supply a schema as text in the prompt
The following example prompts the model to return cookie recipes in a specific JSON format.

Since the model gets the format specification from text in the prompt, you may have some flexibility in how you represent the specification. Any reasonable format for representing a JSON schema may work.


from google import genai

prompt = """List a few popular cookie recipes in JSON format.

Use this JSON schema:

Recipe = {'recipe_name': str, 'ingredients': list[str]}
Return: list[Recipe]"""

client = genai.Client(api_key="GEMINI_API_KEY")
response = client.models.generate_content(
    model='gemini-2.0-flash',
    contents=prompt,
)

# Use the response as a JSON string.
print(response.text)
Supply a schema through model configuration
The following example does the following:

Instantiates a model configured through a schema to respond with JSON.
Prompts the model to return cookie recipes.
This more formal method for declaring the JSON schema gives you more precise control than relying just on text in the prompt.

Important: When you're working with JSON schemas in the Gemini API, the order of properties matters. For more information, see Property ordering.

from google import genai
from pydantic import BaseModel


class Recipe(BaseModel):
  recipe_name: str
  ingredients: list[str]


client = genai.Client(api_key="GEMINI_API_KEY")
response = client.models.generate_content(
    model='gemini-2.0-flash',
    contents='List a few popular cookie recipes. Be sure to include the amounts of ingredients.',
    config={
        'response_mime_type': 'application/json',
        'response_schema': list[Recipe],
    },
)
# Use the response as a JSON string.
print(response.text)

# Use instantiated objects.
my_recipes: list[Recipe] = response.parsed
Define a Schema with a Type
The easiest way to define a schema is with a direct type. This is the approach used in the preceding example:


config={'response_mime_type': 'application/json',
        'response_schema': list[Recipe]}
The Gemini API Python client library supports schemas defined with the following types (where AllowedType is any allowed type):

int
float
bool
str
list[AllowedType]
For structured types:
dict[str, AllowedType]. This annotation declares all dict values to be the same type, but doesn't specify what keys should be included.
User-defined Pydantic models. This approach lets you specify the key names and define different types for the values associated with each of the keys, including nested structures.
Use an enum to constrain output
In some cases you might want the model to choose a single option from a list of options. To implement this behavior, you can pass an enum in your schema. You can use an enum option anywhere you could use a str in the response_schema, because an enum is a list of strings. Like a JSON schema, an enum lets you constrain model output to meet the requirements of your application.

For example, assume that you're developing an application to classify musical instruments into one of five categories: "Percussion", "String", "Woodwind", "Brass", or ""Keyboard"". You could create an enum to help with this task.

In the following example, you pass the enum class Instrument as the response_schema, and the model should choose the most appropriate enum option.


from google import genai
import enum

class Instrument(enum.Enum):
  PERCUSSION = "Percussion"
  STRING = "String"
  WOODWIND = "Woodwind"
  BRASS = "Brass"
  KEYBOARD = "Keyboard"

client = genai.Client(api_key="GEMINI_API_KEY")
response = client.models.generate_content(
    model='gemini-2.0-flash',
    contents='What type of instrument is an oboe?',
    config={
        'response_mime_type': 'text/x.enum',
        'response_schema': Instrument,
    },
)

print(response.text)