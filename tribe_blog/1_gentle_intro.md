# A Gentle Introduction to Structured Generation with Anthropic API

## Building Reproducible LLM Applications


Welcome to our short series of articles on structured generation, a paradigm designed to reduce the unpredictability of large language models (LLMs) by ensuring that their outputs adhere to predefined formats. 

The goal of the series is to show how structured generation can be implemented in Python using Anthropic's LLMs inference API. 
As a first step, this post introduces the basic concepts behind structured generation. In addition, it provides a practical example that guides Anthropic's Claude 3.5 Sonnet model to produce structured data in many formats. 

In follow-up tutorials, we will explore assistant response prefilling and function calling, as more advanced techniques for structured generation. We’ll also demonstrate how to generate and validate schemas efficiently with the Python Pydantic library. We will wrap up the series with a complete example of legal text classification. 

By the end of it, you will have gained concrete strategies for building more robust LLM-powered applications, and a theoretical understanding of LLMs, with a particular focus on their (un)reliability. 

**Enrica Troiano¹ and Tommaso Furlanello¹²**

¹ HK3Lab
² Tribe AI

**Correspondence:** {name}.{surname}@hk3lab.ai
___
You can find the raw markdown file for the post and the complete code for the examples in our [github repository](https://github.com/HK3-Lab-Team/StructuredGenTutorial). To run the code yourself, simply clone the repository and execute the Jupyter notebook [gentle-intro.ipynb](https://github.com/HK3-Lab-Team/StructuredGenTutorial/blob/main/notebooks/gentle-intro.ipynb).
___


## Motivation


In December 2022, we hailed a new generation of large language models (LLMs) as a godsend for far-reaching technological revolutions. The linguistic skills of these models had become so astoundingly human-like, and their knowledge so exceptional in breadth, that they could 
feature at the same time buddies to chat with and powerful tools to lead industrial progress. 

As a matter of fact, the first LLM-powered applications focused on content generation for human consumption, and today they represent a key workforce in many enterprises, aiding activities like data analysis and decision-making. However, the very feature that makes LLMs so powerful - their ability to generate human-like text - also presents challenges when integrating them into enterprise systems. The inconsistency and unpredictability of their outputs can be particularly problematic.

Anyone who has used Claude, Anthropic's LLM-powered virtual assistant, knows that it can provide different answers when asked the same question multiple times. Moreover, the format of the answers doesn't always align with users' expectations: a direct yes-or-no question might be met with an elaborate explanation, and questions expecting a digit (e.g., `6`) may elicit responses in string form ("six").
While this inconsistency isn't typically problematic for human readers, it can become a significant issue when chatbots are integrated with other systems that require consistent output formats.


These scenarios highlight the need for more structured and predictable outputs from LLMs, especially in applications where consistent formatting is crucial for seamless integration with other systems or processes. This is where the concept of structured generation comes into play, offering a solution to guide LLM outputs into more predictable and usable formats.



## A GENTLE INTRODUCTION TO STRUCTURED GENERATION



The rationale behind the use of output constraints is to narrow the possible LLM's answers down to the answers that have certain characteristics. That way, we maintain control over complex applications based on LLMs, because we force the responses to have a format that is consistent with our expectations. For example, in a customer support chatbot, we may require a response composed of two parts, one that is a direct answer for the customer's question, and the other that offers a detailed log and explanation for the company's database. 


Somewhat implicitly, we already exert that control as soon as we prompt a model with our queries: a prompt is the input condition that determines what tokens the model returns, i.e., what answer (from the space all possible answers) is appropriate to the user's input. But finding a prompt that elicits the desired output format can be a long trial-and-error process. We must experiment with prompt variations and see if the model's responses change accordingly. We may eventually find the prompt that works the best for a specific task, but we have no guarantee it will lead to the desired output format systematically. 

Structured generation complements and enhances prompt engineering, offering a more systematic approach to controlling LLM outputs. 
Let's look at two examples to illustrate its potential.

#### Example 1: Without enforcing syntax.

```
User: "What's the status of my order #12345?"

LLM: "I've checked your order #12345, and it's currently in transit. It was shipped yesterday and is expected to arrive within 3-5 business days. Is there anything else I can help you with?"
```

Processing this response would require complex parsing with custom regex for each attribute of the response:

```python
import re

def process_unstructured_response(response: str):
    """Only working for the specific example"""
    # Extract order number
    order_number_match = re.search(r'order #(\d+)', response)
    order_number = order_number_match.group(1) if order_number_match else None

    # Extract status
    status_match = re.search(r"it's currently ([\w\s]+)", response)
    status = status_match.group(1) if status_match else None

    # Extract shipping information
    shipped_match = re.search(r'It was shipped (\w+)', response)
    shipped_date = shipped_match.group(1) if shipped_match else None

    # Extract estimated delivery time
    delivery_match = re.search(r'expected to arrive within (\d+-\d+) (\w+)', response)
    if delivery_match:
        delivery_time = delivery_match.group(1)
        delivery_unit = delivery_match.group(2)
    else:
        delivery_time = None
        delivery_unit = None

    return {
        "order_number": order_number,
        "status": status,
        "shipped_date": shipped_date,
        "estimated_delivery": f"{delivery_time} {delivery_unit}" if delivery_time and delivery_unit else None
    }

# Test the function
response = "I've checked your order #12345, and it's currently in transit. It was shipped yesterday and is expected to arrive within 3-5 business days. Is there anything else I can help you with?"
result = process_unstructured_response(response)
print(result)

```
```
{'order_number': '12345', 'status': 'in transit', 'shipped_date': 'yesterday', 'estimated_delivery': '3-5 business'}
```

In this example, the LLM provides a human-friendly response that answers the question but doesn't follow any particular structure. This is fine for direct human interaction but can be challenging for automated systems to parse and process.

#### Example 2: With structured generation.

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


In this example, which uses structured generation, the LLM's response is formatted in a specific JSON structure. The format clearly separates the customer-facing response from the internal log information, which makes it easy for the system to:

1. Extract the customer response for display
2. Store detailed order information in a database
3. Update internal tracking systems
4. Trigger automated processes based on the order status

Overall, processing a structured response is much simpler and less error-prone than trying to parse an unstructured response.


### KEY COMPONENTS OF STRUCTURED GENERATION

To implement structured generation effectively, we focus on two main aspects:

1. Defining output schemas: We specify the exact format and structure we expect from the model's output. This helps the model understand our requirements and constrains its responses accordingly.

2. Validation and processing: We implement mechanisms to validate the model's output against our defined schema, ensuring it meets our criteria before further processing or merging into other systems.

These components work together to create a robust framework for generating structured outputs from LLMs. Let's explore how to implement them with Python in the Anthropic API.

## Obtaining Structured Outputs via Prompt Engineering with Anthropic's Claude API

Structured generation can be implemented explicitly through careful prompting of the model or implicitly with an algorithm called guided decoding, which manipulates the model generation at each token to enforce the desired structure. By leveraging both, we can exert fine-grained control over the model's responses while maintaining flexibility. 

We will now exploit prompt engineering to guide the model towards the desired output format. After a few experiments, we will use two more advanced techniques to enforce structured generation: assistant response prefilling and function calling.

### Setting Up Our Environment
Let's start by setting up our environment and installing the anthropic library. First in the terminal we install the anthropic library and set up our API key.

```bash
pip install anthropic
```
If you do not have an API key, you can obtain one by registering for an account on the [Anthropic API](https://console.anthropic.com/settings/api-keys).
To set up our API key in a secure way, either register to the global environment variable or store the key in a .env file and load it using the python library `python-dotenv`, which can be installed with the following command:

```bash
pip install python-dotenv
```
If you are following the tutorial from the cloned github repository, you can find the .env.copy file in the root of the repository (please rename it to .env after editing it with your API key). Otherwise, you can create the file yourself and add the following line:

```bash
ANTHROPIC_API_KEY='YOUR_API_KEY'
```

In case we choose to register to the global environment variable we run the following command in the terminal:

```bash
export ANTHROPIC_API_KEY='YOUR_API_KEY'
```
Now, back in Python, we check that everything is working we can run the following code using `dotenv`:

```python
from dotenv import load_dotenv
import os
load_dotenv()
key = os.environ["ANTHROPIC_API_KEY"]
if key is None:
    raise ValueError("Error: ANTHROPIC_API_KEY not found")
```

Alternatively, if you're using directly the global environment, simply do:

```python
import os
key = os.environ["ANTHROPIC_API_KEY"]
if key is None:
    raise ValueError("Error: ANTHROPIC_API_KEY not found")
```

We are finally ready to import the anthropic library and initialize the client with our API key.
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
### Testing Anthropic API with Claude 3.5 Sonnet
For all our examples, we will use Claude 3.5 Sonnet model which has been extensively trained on structured generation tasks and is highly versatile across different structures like JSON, YAML, XML and CSV. The latest model version at the moment of writing is `claude-3-5-sonnet-20240620`. 

Let's start with a simple example where we ask the model what a JSON schema is, to test that everything is properly installed and that the API key is working.

```python
response = client.messages.create(
    model="claude-3-5-sonnet-20240620",
    max_tokens=200,
    messages=[
        {"role": "user", "content": "What is a JSON schema in a sentence?"}
    ],
)
```
This code snippet will lead to a `Message` object with the following structure:
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
In order to extract the text content of the message, we can index into the content key treating it as a python dictionary.

```python
text_response = response.content[0].text
print(f"Claude response: {text_response}")
```
```
Claude response: A JSON schema is a declarative format for describing the structure, content, and validation rules of JSON data.
```
### Using the System Prompt to Guide the Output Format
Now let's try to make the model response more structured by simply adding examples of the desired output format. We will repeat the same example about the JSON schema but this time we will provide the model with a single example of the desired output format, varying between a Python dictionary, a JSON string and a YAML string.

Anthropic's models are trained to receive instructions through the system prompt, which is a message that is prepended to the user's message and sent along with it to help guide the model's response. In order to access the system prompt, we use the `system` argument in the `messages.create` function. 


We want our structured response to contain a topic, citations and a short answer. 

When we want the model to output a Python dictionary we provide the following example:
1. Python Dictionary:
```python
example_dictionary = {
    "topic": "zip format",
    "citations": [{"citation_number": 1, "source": "https://example.com"}],
    "answer": "The .zip format is a compressed file format that groups multiple files into a single archive, with the files inside the archive appearing as if they were not compressed."
}
```
This is a native Python data structure. It's easy to work with in Python code but isn't as easily interchangeable with other programming languages. A more exchangeable format is:

2. JSON (JavaScript Object Notation):
```python
example_json_string = '{"topic": "zip format", "citations": [{"citation_number": 1, "source": "https://example.com"}], "answer": "The .zip format is a compressed file format that groups multiple files into a single archive, with the files inside the archive appearing as if they were not compressed."}'
```
JSON easy for humans to read and write, and easy for machines to parse and generate. It's language-independent and widely used for API responses and configuration files. Finally, a more human-friendly format is:

3. YAML (YAML Ain't Markup Language):
```python
example_yaml_string = """topic: zip format
citations:
  - citation_number: 1
    source: https://example.com
answer: The .zip format is a compressed file format that groups multiple files into a single archive, with the files inside the archive appearing as if they were not compressed.
"""
```
YAML is data serialization standard often used for configuration files and in applications where data is being stored or transmitted. It is more readable than JSON for complex structures but can be more prone to errors due to its reliance on indentation.

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
This will give us the following output in a python dictionary format:

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
Here's the answer from the json string format:

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

And this is the answer from the yaml string format:

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

As a final example we will process a larger list of file formats, create a dictionary of their responses indexed by the topic with answers as values and save it to disk as a JSON file.

```python
file_formats = [
    "zip", "tar", "rar", "7z", "iso", "gz", "bz2", "xz", "pdf", "docx"
]

format_info = {}

for format in file_formats:
    response = client.messages.create(
        model="claude-3-5-sonnet-20240620",
        system=f"You are a helpful assistant that responds in the same format as the following example: {example_json_string}",
        messages=[
            {"role": "user", "content": f"What is the {format} file format in one sentence?"}
        ],
        max_tokens=200,
    )
    
    try:
        parsed_response = json.loads(response.content[0].text)
        format_info[parsed_response['topic']] = parsed_response['answer']
    except json.JSONDecodeError:
        print(f"Error parsing response for {format} format")

# Save the dictionary to a JSON file
with open('file_formats_info.json', 'w') as f:
    json.dump(format_info, f, indent=2)

print("File format information has been saved to file_formats_info.json")
```

We have seen how for simple tasks like the one above we can use prompt engineering to guide the model towards the desired output format. 

## Conclusion

In this tutorial, we've introduced the concept of structured generation and demonstrated its practical implementation using Anthropic's Claude API. We explored how to guide LLM outputs into predefined formats such as Python dictionaries, JSON, and YAML, enhancing the reliability and usability of AI-generated content in various applications.

Key takeaways:
1. Structured generation addresses the challenge of inconsistent LLM outputs, crucial for integrating AI into enterprise systems.
2. Combining prompt engineering with structured generation techniques offers fine-grained control over model responses.
3. Different output formats (Python dict, JSON, YAML) cater to various use cases and integration needs.
4. Parsing and processing structured outputs enables seamless incorporation into existing workflows.

As we advance in this series, we'll look deeper into more sophisticated techniques like assistant response prefilling, function calling, and leveraging Pydantic for schema generation and validation. These advanced methods will further refine our ability to create robust, predictable LLM applications.

By mastering structured generation, developers can harness the full potential of LLMs while maintaining the consistency and reliability required for production-grade systems. This approach paves the way for more innovative and dependable AI-powered solutions across various industries.


We hope this tutorial has provided a solid introduction to structured generation using Anthropic's Claude API. By implementing these techniques, you can create more reliable and predictable LLM applications that seamlessly integrate with existing systems.

If you found this tutorial helpful, please consider showing your support:

1. Star our GitHub repository: [StructuredGenTutorial](https://github.com/HK3-Lab-Team/StructuredGenTutorial)
2. Stay tuned for our next blog post on [tribe.ai/blog](https://www.tribe.ai/blog)
3. Follow us on Twitter:
   - [@hyp_enri](https://twitter.com/hyp_enri)
   - [@cyndesama](https://twitter.com/cyndesama)

We're excited to continue the next chapter of this series on Schema Engineering, explaining advanced techniques like assistant response prefilling, function calling, and integrating Pydantic for more complex structured generation tasks. 





## Learning More

Here are some valuable resources to deepen your understanding of structured generation and related topics:

1. **Anthropic Cookbook**: Explore practical examples of using Claude for structured JSON data extraction. The cookbook covers tasks like summarization, entity extraction, and sentiment analysis:
   [Extracting Structured JSON with Claude](https://github.com/anthropics/anthropic-cookbook/blob/main/tool_use/extracting_structured_json.ipynb)

2. **Instructor Library**: A powerful tool for generating structured outputs with LLMs, built on top of Pydantic:
   [Instructor Documentation](https://jxnl.github.io/instructor/)
   
   Key features:
   - Support for multiple LLM models (GPT-3.5, GPT-4, GPT-4-Vision, Mistral/Mixtral, Anyscale, Ollama, llama-cpp-python)
   - Simplifies management of validation context and retries
   - Enables streaming of Lists and Partial responses

3. **[Outlines Library](https://github.com/dottxt-ai/outlines)**: An open-source implementation for structured generation with notable features:
   - Multiple model integrations (OpenAI, transformers, llama.cpp, exllama2, mamba)
   - Prompting primitives based on Jinja templating
   - Support for multiple choices, type constraints, and dynamic stopping
   - Fast regex and JSON generation
   - Grammar-structured generation
   - Python function interleaving with completions
   - Caching and batch inference capabilities

4. **[Pydantic Documentation](https://docs.pydantic.dev/)**: For in-depth information on schema generation and validation.

These resources provide an overview of structured generation techniques, from basic implementations to advanced use case.