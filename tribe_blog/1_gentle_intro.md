# A Gentle Introduction to Structured Generation with Anthropic and Pydantic

## Building Reproducible LLM Applications

This is the first article in a series that introduces structured generation, a paradigm aimed at reducing the unpredictability of LLMs by ensuring that their outputs adhere to predefined formats. As part of this framework, the tutorial will cover the general idea of structured generation, and how it can be implemented in Python using Anthropic's Large Language Models (LLMs) inference API and the Pydantic library. We hope this short series will help you clear your doubts about LLM unreliability and set the ground for building more reliable LLM applications.

**Enrica Troiano¹ and Tommaso Furlanello¹²**

¹ HK3Lab
² Tribe AI

**Correspondence:** {name}.{surname}@hk3lab.ai

## Abstract

This post introduces structured generation, an approach aimed at improving the reliability of Large Language Models (LLMs) by guiding their outputs to follow predefined formats. We'll explore practical techniques using the Pydantic Python library to represent and validate data structures. You'll see how these tools can be applied to LLM outputs, adding a layer of safety before using them in critical processes. We'll also demonstrate how combining Pydantic with Anthropic's Claude API can further enhance LLM reliability through output preconditioning. Our goal is to equip you with concrete strategies, grounded in theoretical understanding of LLMs, to build more dependable LLM-powered applications.

Since December 2022, a new generation of large language models (LLMs) has demonstrated remarkable capabilities, sparking excitement about their potential to revolutionize various industries. These models exhibit human-like language skills and possess extensive knowledge across many domains. As a result, LLMs are now being integrated into numerous enterprise workflows, assisting with tasks such as content generation, data analysis, and decision-making.

However, the very feature that makes LLMs so powerful - their ability to generate human-like text - also presents challenges. While this capability is invaluable for processing unstructured data, it often produces verbose outputs that require human interpretation. Consequently, many early LLM applications have focused primarily on generating content for human readers rather than machine consumption.

Whoever has used Claude, the Anthropic's powered LLM-powered virtual assistant, knows that it can give different answers when asked the same question repeatedly. Even the format of the answers doesn't always align with the users' expectations: often, a direct yes-or-no question is met with an elaborate explanation, questions expecting a digit (e.g., `6`) elicit responses in string form ("six"). Of course this is not a problem for a human reader, whose mental representation of a number is unfazed by typing details. But when Chatbots are connected to other systems, a classical example being the integration between ChatGPT and DALL-E 2, the lack of consistency in the output format can be a dealbreaker. 
Virtually every user has been baffled at least once when asking for a creative generation and instead of receiving the image we asked for, we get a textual description of what that image could look like. 

Considering the growing need for more predictable LLMs, this blog post discusses *structured generation*, a paradigm that enables users to guide a LLM's output generation process. First, we'll introduce the intuition behind this paradigm and the concept of JSON schema. Then, we show two methods to implement structured generation with Anthropic's Claude API: assistant response prefilling and function calling. Finally, we'll see how the Python library Pydantic offers powerful tools to convert Python objects into JSON schemas and how to validate LLMs' outputs against these schemas before transforming them into Python objects.

## A GENTLE INTRODUCTION TO STRUCTURED GENERATION



The rationale behind the use of output constraints is to narrow the possible LLM's answers down to the answers that have certain characteristics. That way, we maintain control over complex applications using LLMs, because we force the responses to have a format that is consistent with the expectations of our systems. For example, in a customer support chatbot, we may require a response composed of two parts, one that is a direct answer for the customer's question, and the other that offers a detailed log and explanation for the company's database. Let's look at two examples to illustrate this:

#### Without enforcing syntax:

```
User: "What's the status of my order #12345?"

LLM: "I've checked your order #12345, and it's currently in transit. It was shipped yesterday and is expected to arrive within 3-5 business days. Is there anything else I can help you with?"
```

Processing this response would require complex parsing:

```python
def process_unstructured_response(response: str):
    # Complex parsing logic here
    pass

process_unstructured_response(LLM_response)
```

#### With structured generation:

```
User: "What's the status of my order #12345?"

LLM:
{
  "customer_response": "Your order #12345 is currently in transit. It was shipped yesterday and is expected to arrive within 3-5 business days.",
  "internal_log": {
    "order_number": "12345",
    "status": "in_transit",
    "ship_date": "2023-04-15",
    "estimated_delivery": "2023-04-18 to 2023-04-20",
    "last_update": "2023-04-16T09:30:00Z",
    "notes": "Package scanned at distribution center in Atlanta, GA"
  }
}
```

Processing this response is straightforward:

```python
import json

def process_structured_response(response: str):
    data = json.loads(response)
    return_to_user(data['customer_response'])
    save_to_database(data['internal_log'])

def return_to_user(message: str):
    print(f"Sending to user: {message}")

def save_to_database(log: dict):
    print(f"Saving to database: {log}")

process_structured_response(LLM_response)
```

In the first example, the LLM provides a human-friendly response that answers the question but doesn't follow any particular structure. This is fine for direct human interaction but can be challenging for automated systems to parse and process.

In the second example, using structured generation, the LLM's response is formatted in a specific JSON structure. This format clearly separates the customer-facing response from the internal log information. The structured output makes it easy for the system to:

1. Extract the customer response for display
2. Store detailed order information in a database
3. Update internal tracking systems
4. Trigger automated processes based on the order status

As demonstrated in the code snippets, processing the structured response is much simpler and less error-prone than trying to parse an unstructured response.

Somewhat implicitly, we already exert that control as soon as we prompt a model with our queries: a prompt is the input condition that determines what tokens the model returns, i.e., what answer (from the space all possible answers) is appropriate to the user's input. But finding a prompt that elicits the desired output format can be a long trial-and-error process. We must experiment with prompt variations and see if the model's responses change accordingly. We may eventually find the prompt that works the best for a specific task, but we have no guarantee it will lead to the desired output format systematically. 

Structured generation complements and enhances prompt engineering, offering a more systematic approach to controlling LLM outputs. By combining these techniques, we can achieve more precise and reliable results. Structured generation can be implemented explicitly through careful prompting of the model or implicitly with an algorithm called guided decoding, which manipulates the model generation at each token to enforce the desired structure.


### KEY COMPONENTS OF STRUCTURED GENERATION

To implement structured generation effectively, we focus on two main aspects:

1. Defining output schemas: We specify the exact format and structure we expect from the model's output. This helps the model understand our requirements and constrains its responses accordingly.

2. Validation and processing: We implement mechanisms to validate the model's output against our defined schema, ensuring it meets our criteria before further processing or merging into other systems.

These components work together to create a robust framework for generating structured outputs from LLMs. By leveraging both traditional prompt engineering and structured generation techniques, we can exert fine-grained control over the model's responses while maintaining flexibility. Let's explore how to implement these concepts in Python in the Anthropic API.

## Obtaining Structured Outputs with Anthropic's Claude API

We start with setting up traditional text generation with Anthropic's Claude API. We will think about what makes a good schema and exploit prompt engineering to guide the model towards the desired output format. After a few experiments with prompt-engineering we will explore two more advanced techniques to enforce structured generation: assistant response prefilling and function calling.

Let's start by setting up our environment and installing the anthropic library. First in the terminal we install the anthropic library and set up our API key.

```bash
pip install anthropic
```
If you do not have an API key, you can obtain one by registering for an account on the [Anthropic API](https://console.anthropic.com/settings/api-keys).
To set up our API key in a secure way have two alternatives either register to the global environment variable or store the key in a .env file and load it using the python library `python-dotenv`. Which can be installed with the following command

```bash
pip install python-dotenv
```
If you are following the tutorial from the cloned github repository, you can find the .env.copy file in the root of the repository, rename it to .env after editing it with your API key. Otherwise, you can create the file yourself and add the following line

```bash
ANTHROPIC_API_KEY='YOUR_API_KEY'
```

In case we choose to register to the global environment variable we run the following command in the terminal

```bash
export ANTHROPIC_API_KEY='YOUR_API_KEY'
```
Now in python to check that everything is working we can run the following code using dotenv

```python
from dotenv import load_dotenv
import os
load_dotenv()
key = os.environ["ANTHROPIC_API_KEY"]
if key is None:
    raise ValueError("Error: ANTHROPIC_API_KEY not found")
```

or if using directly the global environment simply:

```python
import os
key = os.environ["ANTHROPIC_API_KEY"]
if key is None:
    raise ValueError("Error: ANTHROPIC_API_KEY not found")
```

Finally we are ready to import the anthropic library and initialize the client with our API key.
```python
import anthropic
from dotenv import load_dotenv
import os
load_dotenv()
key = os.environ["ANTHROPIC_API_KEY"]
if key is None:
    raise ValueError("Error: ANTHROPIC_API_KEY not found")
client = anthropic.Anthropic(
    api_key=key,
)
```

For all our examples we will use Claude 3.5 Sonnet model which has been extensively trained on structured generation tasks and is highly versatile across different structures like JSON, YAML, XML and CSV. The latest model version at the moment of writing is `claude-3-5-sonnet-20240620`. 

Let's start with a simple example where we ask the model what is a JSON schema, to test that everything is properly installed and that the API key is working.

```python
response = client.messages.create(
    model="claude-3-5-sonnet-20240620",
    max_tokens=200,
    messages=[
        {"role": "user", "content": "What is a JSON schema in a sentence?"}
    ],
)
```
will lead to a `Message` object with the following structure:
```
Message = {
    "id": str,
    "content": [
        {
            "text": str,
            "type": str
        }
    ],
    "model": str,
    "role": str,
    "stop_reason": str,
    "stop_sequence": None,  # or potentially str
    "type": str,
    "usage": {
        "input_tokens": int,
        "output_tokens": int
    }
}
```
In order to extract the text content of the message we can index into the content key treating it as a python dictionary.

```python
text_response = response.content[0].text
print(f"Claude response: {text_response}")
```
```
Claude response: A JSON schema is a declarative format for describing the structure, content, and validation rules of JSON data.
```

Now let's see how we can improve our prompt to make the model response more structured by simply adding examples of the desired output format. Anthropic's models are trained to receive instructions through the system prompt, which is a message that is prepended to the user's message and sent along with it, which helps guide the model's response. In order to access the system prompt we use the `system` argument in the `messages.create` function. And we will specify the format using different examples for the desired output format: a Python dictionary, a JSON schema and a YAML schema. 

[Previous content remains the same up to the schema examples]

Let's start by defining our three examples schemas. We want our response to contain a topic, citations and a short answer. Let's insist on the JSON schema definition and write down three output examples describing what .zip format is, using different formats: Python dictionary, JSON, and YAML.

1. Python Dictionary:
```python
example_dictionary = {
    "topic": "zip format",
    "citations": [{"citation_number": 1, "source": "https://example.com"}],
    "answer": "The .zip format is a compressed file format that groups multiple files into a single archive, with the files inside the archive appearing as if they were not compressed."
}
```
This is a native Python data structure. It's easy to work with in Python code but isn't as easily interchangeable with other programming languages.

2. JSON (JavaScript Object Notation):
```python
example_json_string = '{"topic": "zip format", "citations": [{"citation_number": 1, "source": "https://example.com"}], "answer": "The .zip format is a compressed file format that groups multiple files into a single archive, with the files inside the archive appearing as if they were not compressed."}'
```
JSON is a lightweight data interchange format that is easy for humans to read and write and easy for machines to parse and generate. It's language-independent and widely used for API responses and configuration files.

3. YAML (YAML Ain't Markup Language):
```python
example_yaml_string = """topic: zip format
citations:
  - citation_number: 1
    source: https://example.com
answer: The .zip format is a compressed file format that groups multiple files into a single archive, with the files inside the archive appearing as if they were not compressed.
"""
```
YAML is a human-friendly data serialization standard. It's often used for configuration files and in applications where data is being stored or transmitted. YAML is more readable than JSON for complex structures but can be more prone to errors due to its reliance on indentation.

Now, let's use these examples in our prompts:

```python
response_list = []
for example in [example_dictionary, example_json_string, example_yaml_string]:
    response = client.messages.create(
        model="claude-3-5-sonnet-20240620",
        system=f"You are a helpful assistant that responds in the same format as the following example: {example}",
        messages=[
            {"role": "user", "content": "What is a JSON schema in a sentence?"}
        ],
        max_tokens=200,
    )
    response_list.append(response.content[0].text)
    print(f"Claude response: {response.content[0].text}")
```
Which will give us the following output following a python dictionary format:

```python
{
  "topic": "JSON schema",
  "citations": [
    {
      "citation_number": 1,
      "source": "https://json-schema.org/understanding-json-schema/"
    }
  ],
  "answer": "A JSON schema is a declarative language that allows you to annotate and validate JSON documents, defining the structure, constraints, and documentation of JSON data."
}
```
From the json string format:

```json
{
  "topic": "JSON schema",
  "citations": [
    {
      "citation_number": 1,
      "source": "https://json-schema.org/understanding-json-schema/"
    }
  ],
  "answer": "A JSON schema is a declarative language that allows you to annotate and validate JSON documents, defining the structure, constraints, and documentation of JSON data."
}
```

And from the yaml string format:

```yaml
topic: JSON schema

citations:
  - citation_number: 1
    source: https://json-schema.org/understanding-json-schema/

answer: A JSON schema is a declarative language that allows you to annotate and validate JSON documents, defining the structure, constraints, and documentation of JSON data.
```

Now, let's parse each of these responses to demonstrate how we can work with different formats. It's important to note that executing generated Python code can be dangerous in general. We're only doing this for a simple example and with a safe and reliable model like Claude 3.5 Sonnet. In real-world applications, always validate and sanitize any data before processing.

```python
import json
import yaml
import ast

def parse_response(response, format_type):
    try:
        if format_type == 'dict':
            # WARNING: ast.literal_eval is safer than eval, but still use caution
            return ast.literal_eval(response)
        elif format_type == 'json':
            return json.loads(response)
        elif format_type == 'yaml':
            return yaml.safe_load(response)
    except Exception as e:
        print(f"Error parsing {format_type} response: {e}")
        return None

# Parse and print each response
for response, format_type in zip(response_list, ['dict', 'json', 'yaml']):
    parsed = parse_response(response, format_type)
    if parsed:
        print(f"\nParsed {format_type.upper()} response:")
        print(f"Topic: {parsed.get('topic')}")
        print(f"Citation: {parsed.get('citations')[0] if parsed.get('citations') else 'No citation'}")
        print(f"Answer: {parsed.get('answer')}")
    else:
        print(f"\nFailed to parse {format_type.upper()} response")
```

This code demonstrates how to safely parse each type of response. For the Python dictionary, we use `ast.literal_eval()` which is safer than `eval()` but still should be used cautiously. For JSON, we use the built-in `json` module, and for YAML, we use the `pyyaml` library's `safe_load()` function.

By using these different formats and parsing methods, we can see how structured generation can produce outputs that are not only human-readable but also easily processable by machines. This flexibility allows us to integrate LLM outputs into various systems and workflows, enhancing the utility of AI-generated content in practical applications.





## Learning More

Link to anthropic docs and cookbooks.
Link to instructror library
Link to outlines library and state machines paper
Link to Pydantic docs