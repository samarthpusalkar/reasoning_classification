import csv
import random
import openai
import re

openai_client = openai.OpenAI(api_key="hi", base_url="http://127.0.0.1:1234/v1")  # Initialize openai_client
MODEL_NAME = 'llama-3.2-3b-instruct'
# Configure openai_client to use the local server
# You might need to set an API key, even if it's a dummy value, depending on your local server setup.
# openai_client.api_key = "dummy_key" # Uncomment and set a dummy key if needed


def generate_reasoning_openai_client_text():
    reasoning_topics = [
        "benefits of learning a new language",
        "importance of art education in schools",
        "value of critical thinking in everyday life",
        "dangers of misinformation in the digital age",
        "need for sustainable transportation in cities",
        "advantages of learning to cook",
        "why community involvement is important",
        "importance of empathy in social interactions",
        "value of preserving indigenous cultures",
        "benefits of spending time in nature",
        "why access to education should be a universal right",
        "importance of protecting wildlife habitats",
        "value of continuous self-reflection",
        "benefits of flexible work arrangements",
        "why ethical leadership is crucial for organizations",
        "importance of mental health awareness",
        "value of philosophical inquiry",
        "dangers of excessive consumerism",
        "need for responsible resource management",
        "advantages of cross-cultural communication",
        "benefits of learning about different religions",
        "importance of media literacy",
        "value of scientific research",
        "dangers of political polarization",
        "need for global cooperation on pandemics",
        "advantages of gardening",
        "why physical fitness is important for all ages",
        "importance of ethical AI development",
        "value of historical fiction",
        "benefits of gratitude practices",
        "why renewable energy is the future"
    ]
    topic = random.choice(reasoning_topics)

    prompts = [
        f"Write a short paragraph demonstrating strong logical reasoning to explain why {topic}. Conclude with a strong statement.",
        f"Explain with logical arguments why {topic} is important.  Use evidence-based reasoning to support your points.",
        f"Construct a reasoned paragraph arguing for the significance of {topic}.  Structure your argument logically.",
        f"Develop a short text that uses logical thought to justify the importance of {topic}.",
        f"Demonstrate clear reasoning in a paragraph explaining the value of {topic}."
    ]
    prompt = random.choice(prompts)

    try:
        response = openai_client.chat.completions.create( # Using chat completion for better models
            model=MODEL_NAME,  # Replace with the name of your locally served model (e.g., "llama2", "mistral", etc.)
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=150,
            n=1,
            temperature=0.8,
        )
        generated_text = response.choices[0].message.content.strip()

    except Exception as e:
        print(f"Error generating reasoning text: {e}")
        return "" # Return empty string in case of error

    return generated_text


def generate_not_reasoning_openai_client_text():
    not_reasoning_types = ["facts", "news", "summary", "questions", "general", "abstract"] # Added "abstract" type
    type = random.choice(not_reasoning_types)

    if type == "facts":
        prompts = [
            "Write a short, generic fact.",
            "State a random fact.",
            "Generate a factual statement.",
            "Give me a simple fact.",
            "Just the fact:"
        ]
        prompt = random.choice(prompts)
    elif type == "news":
        prompts = [
            "Write a short, generic news headline.",
            "Create a brief news snippet.",
            "Generate a news headline.",
            "Write a news update.",
            "News headline:"
        ]
        prompt = random.choice(prompts)
    elif type == "summary":
        prompts = [
            "Write a very short abstract summary.",
            "Create a brief summary.",
            "Generate an abstract.",
            "Write a short summary of something.",
            "Abstract summary:"
        ]
        prompt = random.choice(prompts)
    elif type == "questions":
        prompts = [
            "Write a generic question.",
            "Create a simple question.",
            "Generate a question.",
            "Ask a random question.",
            "Question:"
        ]
        prompt = random.choice(prompts)
    elif type == "general":
        prompts = [
            "Write a short, generic sensible sentence.",
            "Create a random sensible sentence.",
            "Generate a general sentence.",
            "Write a typical sentence.",
            "Sentence:"
        ]
        prompt = random.choice(prompts)
    elif type == "abstract": # New "abstract" type for not-reasoning
        prompts = [
            "Write a short abstract of a non-person book.", # Explicitly ask for "non-person book" abstract
            "Generate a brief book abstract (non-fiction, impersonal).",
            "Create an abstract for a generic, factual book.",
            "Write an abstract in the style of a dry, informational book.",
            "Book abstract (impersonal):"
        ]
        prompt = random.choice(prompts)


    prompt_final =  prompt #+ " " + random.choice(["", " related to general knowledge.", " suitable for a generic text.", " without analysis.", " impersonal."]) #Trying to guide towards generic/impersonal

    try:
        response = openai_client.chat.completions.create( # Using chat completion for better models
            model="your-local-model-name", # Replace with the name of your locally served model (e.g., "llama2", "mistral", etc.)
            messages=[
                {"role": "user", "content": prompt_final}
            ],
            max_tokens=1024,
            temperature=1.0,
        )
        generated_text = response.choices[0].message.content.strip()
        generated_text = re.sub(r'<think>.*</think>', '', generated_text)  # Remove extra newlines

    except Exception as e:
        print(f"Error generating non-reasoning text: {e}")
        return "" # Return empty string in case of error


    return generated_text


# Generate CSV data
num_data_points = 50  # Generate 9900 data points
data = []
for i in range(num_data_points):
    if i % 2 == 0:  # Alternate between reasoning and not reasoning (approximately balanced)
        text = generate_reasoning_openai_client_text()
        label = 1
    else:
        text = generate_not_reasoning_openai_client_text()
        label = 0

    if text: # Only add if text generation was successful (not empty string due to error)
        data.append({"text": text, "label": label})
    else:
        print(f"Skipping data point {i} due to generation error.") # Inform about skipped data points


# Write to CSV file
csv_file_path = "reasoning_dataset_9900_openai_client_local.csv"
with open(csv_file_path, mode='w', newline='', encoding='utf-8') as csvfile:
    fieldnames = ['text', 'label']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    writer.writerows(data)

print(f"CSV file '{csv_file_path}' with {len(data)} data points generated using local openai_client API successfully.")
