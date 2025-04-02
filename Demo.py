import ao_pyth as ao
from config import ao_apikey
from config import openai_key
from openai import OpenAI

import ast


def llm_call(input_message): #llm call method 
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",  
        messages=[
            {"role": "user", "content": input_message}
        ],
        temperature=0.1
    )
    local_response = response.choices[0].message.content
    return local_response

client = OpenAI(api_key = openai_key)
# Initialize architecture with 8 input neurons, 4 hidden neurons, 5 output neurons. 
# The 5 output neurons correspond to the likelihood of fraud (scale 1-5)
arch = ao.Arch(arch_i="[1, 1, 1, 1, 1, 1, 1, 1]", arch_z="[1, 1, 1, 1, 1]", api_key=ao_apikey, kennel_id="Speak_demo") 
print(arch.api_status)

# Create an agent with the given architecture
agent = ao.Agent(arch, uid="Test11")

# Training examples:
# Format: [Deactivated or No LinkedIn, Zero GitHub or Personal Projects Listed, Buzzword Soup for Skills, 
#          Generic Role Descriptions, Inconsistent or Shady Company Info, Job Titles Don’t Match Timeline, 
#          Too Many Freelance Projects with No Clients Named, Resume Format Looks AI-Generated or Translated]
# -> Likelihood of fraud (scale 1-5)
training_data = [
    # Highest fraud likelihood
    ([1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1]),
    # Very high fraud likelihood with one minor flag missing
    ([1, 1, 1, 1, 1, 1, 0, 1], [1, 1, 1, 1, 0]),
    # High fraud likelihood with one less flag
    ([1, 1, 1, 0, 1, 1, 1, 1], [1, 1, 1, 1, 0]),
    # High fraud likelihood with a couple of flags off
    ([1, 1, 0, 1, 1, 1, 1, 1], [1, 1, 1, 0, 1]),
    # Moderately high fraud likelihood
    ([1, 0, 1, 1, 1, 1, 1, 0], [1, 1, 1, 0, 0]),
    # Medium fraud likelihood
    ([1, 1, 0, 0, 1, 1, 0, 1], [1, 1, 0, 0, 1]),
    # Medium fraud likelihood with balanced flags
    ([1, 0, 1, 0, 1, 0, 1, 1], [1, 0, 1, 0, 1]),
    # Medium-low fraud likelihood
    ([0, 1, 1, 0, 1, 0, 0, 1], [0, 1, 1, 0, 0]),
    # Medium-low fraud likelihood with few red flags
    ([0, 1, 0, 1, 0, 1, 0, 0], [0, 1, 0, 1, 0]),
    # Lower fraud likelihood with more zeros
    ([1, 0, 0, 1, 0, 0, 0, 1], [1, 0, 0, 1, 0]),
    # Lower fraud likelihood with minimal flags
    ([0, 0, 1, 0, 0, 1, 0, 0], [0, 0, 1, 0, 0]),
    # Low fraud likelihood
    ([1, 0, 0, 0, 1, 0, 0, 0], [0, 1, 0, 0, 0]),
    # Very low fraud likelihood with only one flag on
    ([0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 1]),
    # Nearly no fraud indicators
    ([0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0]),
    # Lowest fraud likelihood
    ([0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0])
]

# Train the agent with the examples
###Uncomment to train the agent
# for inp, label in training_data:
#     agent.next_state(INPUT=inp, LABEL=label, unsequenced=True)  # Reset states and unsequenced True


resume = """
John Doe
1234 Example Lane
Anytown, USA 12345
(555) 123-4567
john.doe@example.com

Objective
Innovative and dynamic professional with a passion for delivering value through synergy and holistic solutions. Seeking opportunities to leverage expertise in diverse roles and drive transformative results.

Experience

Freelance Consultant
Various Projects
June 2018 – Present

Delivered end-to-end consulting services across multiple industries using agile frameworks.

Developed cutting-edge solutions and implemented next-generation technologies.

Managed several freelance projects without publicly listed client names.

Senior Project Manager
XYZ Corporation
January 2017 – May 2018

Led cross-functional teams in the delivery of enterprise-scale projects.

Focused on innovative strategies and dynamic process optimization.

Oversaw global initiatives with a mix of inconsistent role descriptions and timelines.

Software Engineer
ABC Innovations
March 2015 – December 2016

Designed and implemented robust software solutions with buzzword-heavy technical jargon.

Engaged in iterative development practices and integrated scalable architectures.

Project roles and job titles did not consistently align with the provided timelines.

Education

Bachelor of Science in Computer Science
University of Nowhere, 2011 – 2015

Skills

Proficient in Python, Java, and C++

Expertise in agile methodologies, cloud computing, and data analytics

Strong ability to drive innovation and optimize system performance

Certifications

Certified Agile Professional

Additional Information

LinkedIn: Profile is deactivated.

GitHub/Projects: No personal GitHub account or project repositories available.

Resume Format: Appears to be auto-generated and translated, with generic role descriptions and inconsistent information.

"""
input_to_agent = [0, 0 ,0 ,0 , 0, 0, 0, 0]
response= ast.literal_eval(llm_call(f"""I am attching a resume to this chat.Fill out this list with 1 OR 0 of length 8 Then return the list only .Format: [Deactivated or No LinkedIn, Zero GitHub or Personal Projects Listed, Buzzword Soup for Skills, 
#          Generic Role Descriptions, Inconsistent or Shady Company Info, Job Titles Don’t Match Timeline, 
#          Too Many Freelance Projects with No Clients Named, Resume Format Looks AI-Generated or Translated] {resume} 
                       """))
print("chatgpt response: ", response)
print(type(response))

agent_response = agent.next_state(response)
print("agent response: ", agent_response)

ones = sum(agent_response)

print("Predicted likelihood of fraud: ", ones / len(agent_response) * 100, "%")


