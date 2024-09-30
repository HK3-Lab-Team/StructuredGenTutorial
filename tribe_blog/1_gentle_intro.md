# A Gentle Introduction to Structured Generation with Anthropic and Pydantic

## Building Reproducible LLM Applications

This is the first article in a series that introduces structured generation, a paradigm aimed at reducing the unpredictability of LLMs by ensuring that their outputs adhere to predefined formats. As part of this framework, the tutorial will cover the general idea of structured generation, and how it can be implemented in Python using Anthropic's Large Language Models (LLMs) inference API and the Pydantic library. We hope this short series will help you clear your doubts about LLM unreliability and set the ground for building more reliable LLM applications.

**Enrica Troiano¹ and Tommaso Furlanello¹²**

¹ HK3Lab
² Tribe AI

**Correspondence:** {name}.{surname}@hk3lab.ai

## Abstract

This post introduces structured generation, an approach aimed at improving the reliability of Large Language Models (LLMs) by guiding their outputs to follow predefined formats. We'll explore practical techniques using the Pydantic Python library to represent and validate data structures. You'll see how these tools can be applied to LLM outputs, adding a layer of safety before using them in critical processes. We'll also demonstrate how combining Pydantic with Anthropic's Claude API can further enhance LLM reliability through output preconditioning. Our goal is to equip you with concrete strategies, grounded in theoretical understanding of LLMs, to build more dependable LLM-powered applications.

Since December 2022, a new generation of large language models (LLMs) showed unprecedented capabilities that crystallized the market expectations for far-reaching technological revolutions. The linguistic skills of these models had become so astoundingly human-like, and their knowledge so exceptional in breadth, that they could feature at the same time buddies to chat with and powerful tools to lead industrial progress. Unsurprisingly, LLMs are now a key workforce of many enterprises (aiding activities like content generation, data analysis and decision-making), but now without novel issues. Interestingly their biggest advantage is also their biggest weakness: their ability to generate text that closely mimics human-like responses. While this skill is amazing and useful for implicitly processing unstructured data, it also requires a human-like reader to understand all the information packed in verbose answers. It is no surprise that the first wave of LLM-powered applications focused on content generation for human consumption.

Whoever has used Claude, the Anthropic's powered LLM-powered virtual assistant, knows that it can give different answers when asked the same question repeatedly. Even the format of the answers doesn't always align with the users' expectations: often, a direct yes-or-no question is met with an elaborate explanation, questions expecting a digit (e.g., `6`) elicit responses in string form ("six"). Of course this is not a problem for a human reader, whose mental representation of a number is unfazed by typing details. But when Chatbots are connected to other systems, a classical example being the integration between ChatGPT and DALL-E 2, the lack of consistency in the output format can be a dealbreaker. 
Virtually every user has been baffled at least once when asking for a creative generation and instead of receiving the image we asked for, we get a textual description of what that image could look like. 

Considering the growing need for more predictable LLMs, this blog post discusses *structured generation*, a paradigm that enables users to guide a LLM's output generation process. First, we'll introduce the intuition behind this paradigm and the concept of JSON schema. Then, we show two methods to implement structured generation with Anthropic's Claude API: assistant response prefilling and function calling. Finally, we'll see how the Python library Pydantic offers powerful tools to convert Python objects into JSON schemas and how to validate LLMs' outputs against these schemas before transforming them into Python objects.

## A GENTLE INTRODUCTION TO STRUCTURED GENERATION

Structured generation is an approach to specify our expectations for LLMs. This is achieved with a class of coding techniques that serve to put constraints on the structure of the LLMs' outputs.  

### CORE IDEAS

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

