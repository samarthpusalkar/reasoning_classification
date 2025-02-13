import csv
import random

def generate_reasoning_text():
    reasoning_topics = [
        "benefits of reading fiction",
        "importance of learning history",
        "value of travel and cultural exchange",
        "dangers of social media overuse",
        "need for urban green spaces",
        "advantages of learning a musical instrument",
        "why volunteering is beneficial",
        "importance of creativity in problem solving",
        "value of preserving historical landmarks",
        "benefits of mindfulness and meditation",
        "why access to clean water is essential",
        "importance of protecting endangered species",
        "value of lifelong learning",
        "benefits of a diverse and inclusive workplace",
        "why ethical considerations are important in technology",
        "importance of sleep for cognitive function",
        "value of art in society",
        "dangers of processed foods",
        "need for recycling and waste reduction",
        "advantages of teamwork in projects"
    ]
    topic = random.choice(reasoning_topics)

    structures = [
        [
            "Firstly, [point1].",
            "Secondly, [point2].",
            "Finally, [point3].",
            "Therefore, [conclusion]."
        ],
        [
            "[Claim].",
            "This is supported by [evidence1].",
            "Furthermore, [evidence2].",
            "Thus, [conclusion]."
        ],
        [
            "It is argued that [claim].",
            "One key reason is [reason1].",
            "Another compelling point is [reason2].",
            "Consequently, [conclusion]."
        ]
    ]
    structure = random.choice(structures)

    if topic == "benefits of reading fiction":
        points = ["fiction enhances empathy by allowing us to experience different perspectives",
                  "it expands vocabulary and improves language skills",
                  "reading fiction stimulates imagination and creativity",
                  "reading fiction cultivates critical thinking and analytical abilities"]
        conclusion = "reading fiction is undeniably beneficial for personal growth and intellectual development"
    elif topic == "importance of learning history":
        points = ["history provides context for understanding current events",
                  "it teaches us about past mistakes and successes, informing present decisions",
                  "history fosters critical thinking and analytical skills by examining evidence",
                  "understanding history promotes cultural awareness and empathy"]
        conclusion = "learning history is crucial for informed citizenship and societal progress"
    elif topic == "value of travel and cultural exchange":
        points = ["travel broadens perspectives by exposing us to different cultures",
                  "it fosters tolerance and understanding by breaking down stereotypes",
                  "travel enhances problem-solving skills and adaptability",
                  "cultural exchange promotes global cooperation and diplomacy"]
        conclusion = "travel and cultural exchange are invaluable for personal growth and global harmony"
    elif topic == "dangers of social media overuse":
        points = ["excessive social media use can lead to social isolation and loneliness",
                  "it contributes to mental health issues like anxiety and depression",
                  "social media often promotes unrealistic comparisons and body image concerns",
                  "it can be a source of misinformation and cyberbullying"]
        conclusion = "overuse of social media poses significant risks to mental and social well-being"
    elif topic == "need for urban green spaces":
        points = ["green spaces in cities improve air quality and reduce pollution",
                  "they provide habitats for wildlife and enhance biodiversity",
                  "urban green spaces offer recreational opportunities and promote physical activity",
                  "they reduce stress and improve mental well-being for city dwellers"]
        conclusion = "urban green spaces are essential for creating healthier and more livable cities"
    elif topic == "advantages of learning a musical instrument":
        points = ["learning music enhances cognitive skills, including memory and attention",
                  "it fosters discipline and perseverance through practice",
                  "playing music improves coordination and motor skills",
                  "it provides a creative outlet and a source of personal enjoyment"]
        conclusion = "learning a musical instrument offers numerous cognitive, personal, and creative benefits"
    elif topic == "why volunteering is beneficial":
        points = ["volunteering makes a positive impact on the community and helps those in need",
                  "it provides a sense of purpose and fulfillment",
                  "volunteering can lead to new skills and experiences",
                  "it strengthens social connections and combats isolation"]
        conclusion = "volunteering is highly beneficial both for the individual and for society as a whole"
    elif topic == "importance of creativity in problem solving":
        points = ["creative thinking allows for innovative solutions to complex problems",
                  "it encourages thinking outside the box and exploring unconventional approaches",
                  "creativity enhances adaptability and resilience in the face of challenges",
                  "it fosters innovation and progress in various fields"]
        conclusion = "creativity is vital for effective problem-solving and driving innovation"
    elif topic == "value of preserving historical landmarks":
        points = ["historical landmarks connect us to the past and provide a sense of continuity",
                  "they offer insights into previous cultures and ways of life",
                  "preserving landmarks contributes to cultural identity and heritage",
                  "they can be valuable for education and tourism"]
        conclusion = "preserving historical landmarks is important for cultural understanding and historical appreciation"
    elif topic == "benefits of mindfulness and meditation":
        points = ["mindfulness reduces stress and anxiety by focusing on the present moment",
                  "it improves emotional regulation and self-awareness",
                  "meditation can enhance concentration and focus",
                  "it promotes overall mental well-being and inner peace"]
        conclusion = "mindfulness and meditation are highly beneficial for mental and emotional health"
    elif topic == "why access to clean water is essential":
        points = ["clean water is fundamental for basic human health and survival",
                  "it prevents waterborne diseases and improves sanitation",
                  "access to clean water supports agriculture and food production",
                  "it is crucial for economic development and societal well-being"]
        conclusion = "access to clean water is not a luxury but a fundamental human right and necessity"
    elif topic == "importance of protecting endangered species":
        points = ["endangered species are vital components of ecosystems and biodiversity",
                  "their extinction can have cascading effects on the food chain",
                  "many endangered species have potential medicinal or scientific value",
                  "protecting them is an ethical responsibility to preserve life on Earth"]
        conclusion = "protecting endangered species is crucial for ecological balance and ethical stewardship"
    elif topic == "value of lifelong learning":
        points = ["lifelong learning keeps minds sharp and adaptable throughout life",
                  "it allows individuals to adapt to changing job markets and technologies",
                  "continuous learning enhances personal growth and fulfillment",
                  "it contributes to a more informed and engaged citizenry"]
        conclusion = "lifelong learning is essential for personal and professional development in the modern world"
    elif topic == "benefits of a diverse and inclusive workplace":
        points = ["diverse teams bring a wider range of perspectives and ideas",
                  "inclusive workplaces foster creativity and innovation",
                  "diversity improves employee satisfaction and retention",
                  "it reflects the diversity of the customer base and society"]
        conclusion = "diversity and inclusion in the workplace are key drivers of success and ethical practice"
    elif topic == "why ethical considerations are important in technology":
        points = ["technology can have unintended consequences and ethical dilemmas",
                  "algorithms can perpetuate biases and inequalities",
                  "data privacy and security are crucial ethical concerns in the digital age",
                  "ethical frameworks are needed to guide technological development and use"]
        conclusion = "ethical considerations are paramount in technology to ensure responsible innovation"
    elif topic == "importance of sleep for cognitive function":
        points = ["sleep is essential for memory consolidation and learning",
                  "it allows the brain to clear toxins and repair itself",
                  "adequate sleep improves attention and concentration",
                  "sleep deprivation impairs cognitive performance and decision-making"]
        conclusion = "sufficient sleep is crucial for optimal cognitive function and overall health"
    elif topic == "value of art in society":
        points = ["art expresses emotions and ideas in powerful and unique ways",
                  "it enriches culture and provides aesthetic enjoyment",
                  "art can challenge perspectives and promote social commentary",
                  "it fosters creativity and imagination in both creators and observers"]
        conclusion = "art plays a vital role in society by enriching culture, fostering creativity, and promoting expression"
    elif topic == "dangers of processed foods":
        points = ["processed foods are often high in unhealthy fats, sugars, and sodium",
                  "they contribute to obesity and related health problems",
                  "processed foods may lack essential nutrients and fiber",
                  "excessive consumption of processed foods is linked to chronic diseases"]
        conclusion = "processed foods pose significant health risks and should be consumed in moderation"
    elif topic == "need for recycling and waste reduction":
        points = ["recycling conserves natural resources and reduces landfill waste",
                  "it saves energy and reduces pollution compared to raw material production",
                  "waste reduction minimizes environmental impact and promotes sustainability",
                  "recycling and waste reduction are crucial for a circular economy"]
        conclusion = "recycling and waste reduction are essential for environmental protection and resource management"
    elif topic == "advantages of teamwork in projects":
        points = ["teamwork allows for diverse skill sets and perspectives to be combined",
                  "it increases efficiency and productivity by sharing workload",
                  "teamwork fosters collaboration and communication skills",
                  "it can lead to more creative and robust solutions"]
        conclusion = "teamwork offers significant advantages in project completion and achieving complex goals"


    text = ""
    for i, line_template in enumerate(structure):
        point_placeholder = f"point{i+1}"
        if point_placeholder in line_template:
            text += line_template.replace(point_placeholder, points[i]) + " "
        else:
            text += line_template.replace("[claim]", topic.capitalize()).replace("[evidence1]", points[0]).replace("[evidence2]", points[1]).replace("[reason1]", points[0]).replace("[reason2]", points[1])  + " " #Fallback for claim structures
    text += structure[-1].split()[-1].replace("[conclusion]", conclusion) #Ensure conclusion is correctly added in case of structure variation
    return text.strip()

def generate_not_reasoning_text():
    not_reasoning_types = ["facts", "news", "summary", "questions", "general"]
    type = random.choice(not_reasoning_types)

    if type == "facts":
        facts = [
            "The sky is often blue.",
            "Water boils at 100 degrees Celsius.",
            "Cats are mammals.",
            "Trees have leaves.",
            "The Earth is round.",
            "Dogs bark.",
            "Fish live in water.",
            "Birds fly in the sky.",
            "Mountains are tall.",
            "Rivers flow to the sea."
        ]
        return random.choice(facts)
    elif type == "news":
        news_headlines = [
            "Stock market shows mixed results today.",
            "Local park to host community event.",
            "City council discusses new traffic regulations.",
            "Weather forecast predicts sunny skies for the weekend.",
            "New study reveals interesting findings about sleep patterns.",
            "Art exhibit opens at the downtown gallery.",
            "Popular band announces concert tour dates.",
            "Construction project to begin next month.",
            "Library offers free workshops on digital literacy.",
            "Community garden seeks volunteers for planting season."
        ]
        return random.choice(news_headlines)
    elif type == "summary":
        summary_topics = ["book", "movie", "documentary", "article"]
        topic = random.choice(summary_topics)
        if topic == "book":
            summaries = [
                "Chapter one introduces the main characters. Chapter two develops the central conflict. The book explores themes of identity and belonging. It is set in a fictional town.",
                "This novel tells the story of a young woman's journey. It is a coming-of-age tale. The setting is a small village. The writing style is descriptive and evocative."
            ]
            return random.choice(summaries)
        elif topic == "movie":
            summaries = [
                "The film opens with a dramatic scene. The plot revolves around a mystery. The acting is generally praised. Special effects are used extensively. It received mixed reviews.",
                "This movie is a comedy-drama. It follows the lives of several characters. The dialogue is witty and engaging. The soundtrack is memorable. It won several awards."
            ]
            return random.choice(summaries)
        elif topic == "documentary":
            summaries = [
                "The documentary investigates a historical event. It includes interviews with experts. Archival footage is shown. The narration is informative. It raises important questions.",
                "This film explores the natural world. It focuses on animal behavior. Stunning visuals are presented. The music is calming. It aims to educate and inspire."
            ]
            return random.choice(summaries)
        elif topic == "article":
            summaries = [
                "The article discusses recent scientific discoveries. It presents data and analysis. Graphs and charts are included. The writing is technical and detailed. It is published in a journal.",
                "This article is about current social trends. It offers opinions and perspectives. Anecdotes are used to illustrate points. The tone is informal. It is available online."
            ]
            return random.choice(summaries)

    elif type == "questions":
        questions = [
            "What time is it?",
            "How are you today?",
            "Is it going to rain?",
            "Where is the nearest store?",
            "What is your favorite color?",
            "Do you like coffee?",
            "Have you seen this movie?",
            "Can you help me?",
            "What are you reading?",
            "Is this seat taken?"
        ]
        return random.choice(questions)
    elif type == "general":
        general_sentences = [
            "The car is parked outside.",
            "The coffee is hot.",
            "The book is on the table.",
            "The flowers are blooming.",
            "The music is playing softly.",
            "The door is open.",
            "The window is closed.",
            "The lights are on.",
            "The phone is ringing.",
            "The clock is ticking."
        ]
        return random.choice(general_sentences)


# Generate CSV data
num_data_points = 9900  # Generate 9900 data points
data = []
for i in range(num_data_points):
    if i % 2 == 0:  # Alternate between reasoning and not reasoning (approximately balanced)
        text = generate_reasoning_text()
        label = 1
    else:
        text = generate_not_reasoning_text()
        label = 0
    data.append({"text": text, "label": label})

# Write to CSV file
csv_file_path = "reasoning_dataset_9900.csv"
with open(csv_file_path, mode='w', newline='', encoding='utf-8') as csvfile:
    fieldnames = ['text', 'label']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    writer.writerows(data)

print(f"CSV file '{csv_file_path}' with {num_data_points} data points generated successfully.")