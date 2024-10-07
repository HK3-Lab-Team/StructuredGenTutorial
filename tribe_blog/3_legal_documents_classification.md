# Blog 2: Structured Generation with Anthropic LLMs

## Building Reproducible LLM Applications

**Enrica Troiano¹ and Tommaso Furlanello¹**

¹ HK3Lab

**Correspondence:** {name}.{surname}@hk3lab.ai

**Abstract:** This blog post is a follow-up to our introduction to structured generation. Previously, we covered Pydantic as our foundational toolkit to define schemas for the LLMs' outputs. Now, we streamline the use of Pydantic in LLM-based workflows, demonstrating how structured generation works in practice.

Previously, we covered Pydantic as our foundational toolkit to define schemas for the LLMs’ outputs. Now, we streamline the use of Pydantic in LLM-based workflows, demonstrating how structured generation works in practice. We'll also demonstrate how combining Pydantic with Anthropic's Claude API can further enhance LLM reliability through output preconditioning. 

If the task for the LLM is complex, we can break it down into smaller, manageable operations. For each step, we establish the expected structure and validate the outputs, ensuring that when we combine them, each component adheres to the desired format. This way, we can confidently handle complex tasks while maintaining control over the structure at every stage.

In the following tutorial, we will implement structured generation in Python with Claude 3.5, a LLM released by Anthropic.

You'll see how these tools can be applied to LLM outputs, adding a layer of safety before using them in critical processes. 

## Hands-on Exercise: Text Classification via Structured Generation

Let's dive into a real-world classification task. As a use case, we will implement a LLM-based classifier for topic extraction. The model's goal is to map texts that ask for advice on legal issues to a specific area of law. Based on the data released by [Li et al. (2022)](https://huggingface.co/datasets/jonathanli/legal-advice-reddit), these areas include 11 topics: employment, housing, contract, driving, business, wills, criminal, family, digital, school, and insurance.

```python
# Install all required packages for this tutorial

!pip install dataset
!pip install anthropic
!pip install scikit-learn
!pip install polars
!pip install matplotlib
!pip install seaborn
```

### Step 1: Load and Preprocess the Data

We load the dataset of legal advice using the Hugging Face's `load_dataset` function. This step returns a `DatasetDict` object, `raw_data`, which contains three splits: train, validation, and test. Each split has multiple features (i.e., the dataset columns), including `title` (the title of the legal advice request), `body` (the actual text of a Reddit post asking for advice), `text_label` (the name of the associated topic, one of the 11 legal areas), and some additional meta information. 

```python
from datasets import load_dataset
import html

raw_data = load_dataset("jonathanli/legal-advice-reddit")

print(raw_data["train"].features.keys())
dict_keys(['created_utc', 'full_link', 'id', 'body', 'title', 
           'text_label', 'flair_label'])
```

Since the Reddit posts in this dataset contain special HTML entities, zero-width spaces, and newline characters, we do a quick preprocessing to prepare them for analysis. With the `.map()` method, we apply the cleaning function on all three splits of the dataset.

```python
def clean_body_field(mydict):
    
    # Unescape twice to handle doubly encoded entities
    s_clean = html.unescape(mydict['body'])
    s_clean = html.unescape(s_clean)  
    # Remove zero-width spaces and newlines
    s_clean = s_clean.replace("\u200B", "") 
    s_clean = s_clean.replace("\n", "")  
    
    mydict['body'] = s_clean
    return mydict

# Apply cleaning on entire dataset
data = raw_data.map(clean_body_field)

# An example
example_row = data["test"][0]

print(example_row["body"])
'Hi all, curious to get some opinions on visitation rights in regards to vacations. You can look at my post history to get some more background [...]'

print(example_row["title"])
'Reasonable to deny non-custodial parent vacation week? [PA]'

print(example_row["text_label"])
'family'
```

### Step 2: Define Pydantic Schemas

We can now write down a very simple schema for our task. Assume that we just want the LLM to identify the topic of a text, and to motivate that prediction. Accordingly, our Pydantic class `LegalAdviceSimple` can comprise only two fields. We define `topic` as a literal type to restrict its possible values to the 11 topics of the dataset, and `reason` as a string, with a description that provides guidance to the model about the intended purpose of this field.

```python
from pydantic import BaseModel, Field, field_validator
from typing import Literal

# A first version of the schema
class LegalAdviceSimple(BaseModel):
    topic: Literal['employment','housing','contract','driving','business','wills',
                   'criminal','family','digital','school','insurance']
    reason: str = Field(description="The reason for the classification decision.")
```

### Step 3: Integrate the Schema with Claude

We use the Anthropic library and the model `claude-3-5-sonnet-20240620` to create a function for topic extraction.

```python
import anthropic

def topic_extraction(input_title, input_request, chosen_schema):
    response = client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=1024,
        system="You are a helpful assistant that extracts topics of requests for legal advice and returns a json object",
        messages=[{
            "role": "user", 
            "content": f"Extract the legal advice topic. TITLE: {input_title}. BODY {input_body}"
        }],
        tools=[chosen_schema],
    )
    return(response)
```

This function sends a request to the Claude API, specifying the LLM model to call, the maximum number of output tokens, a system message describing the assistant's role, and a user message with the title and body of the legal advice. Importantly, `topic_extraction` also indicates some additional tools to use: `tools` serve to allow Claude to use external functions when performing a task. In our case, that function is a `chosen_schema`, that is derived from the Pydantic class `LegalAdviceSimple` like so:

```python
from anthropic.types import ToolParam

# Note that natural language descriptions can be used 
# to emphasize key elements of the tool's JSON schema.
schema = ToolParam(
    name="topic_extraction", 
    description="A schema to extract information from a legal advice",
    input_schema=LegalAdviceSimple.model_json_schema()
)
```

`ToolParam` tells Claude:
- what the function is called ("topic_extraction");
- what it does (extracts information from legal advice);
- what structure the input (and by extension, the LLM output) should have (defined by LegalAdviceSimple.model_json_schema()).

The more details the model has regarding the tool and its usage, the better its performance will be. 

In other words, by including `tools` in the API call, we're enabling function calling. This is what informs Claude on what parameters the function expects. As a result, the model structures its response according to the schema we've defined, ensuring that the output will be a JSON object containing the `topic` extracted from the legal advice text, and the `reason` corresponding to this classification decision.

### Step 4: Perform Classification

At this point, we are ready to perform topic classification on (a random sample of) the test data. For each entry, we call the `topic_extraction` function to get a prediction from the model. We store the model's response along with the true label in `topics_simple` for later evaluation.

```python
anthropic_api_key = [YOUR KEY GOES HERE]
client = anthropic.Anthropic(api_key=anthropic_api_key)

sampled_test_set = data['test'].shuffle(seed=42).select(range(100)) 
topics_simple = []

for ind, entry in enumerate(sampled_test_set):
    print(f"Processing entry: {ind}")
    body = entry['body']
    title = entry['title']
    gold_label = entry['text_label']
    llm_answer = topic_extraction(title, body, schema)
    prediction = llm_answer.content[1].input['topic']
    topics_simple.append({
        'gold_label': gold_label, 
        'predicted_label': prediction,
        'title': title, 
        'body': body
    })
```

___
Note that Pydantic is integraetd with various libraries to call LLMs: Instructor,  Libraries like Marvin, Langchain, and Llamaindex.