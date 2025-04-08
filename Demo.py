import ast
import ao_core as ao
from config import ao_apikey
from config import openai_apikey
from openai import OpenAI


def llm_call(input_message): #llm call method 
    client = OpenAI(api_key = openai_apikey)
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",  
        messages=[
            {"role": "user", "content": input_message}
        ],
        temperature=0.1
    )
    local_response = response.choices[0].message.content
    return local_response


# Initialize AO agent architecture with 8 input neurons, 8 hidden neurons (by default), 5 output neurons. 
# The 5 output neurons correspond to the likelihood of fraud (scale 1-5).
arch = ao.Arch(arch_i="[1, 1, 1, 1, 1, 1, 1, 1]", arch_z="[1, 1, 1, 1, 1]", api_key=ao_apikey, kennel_id="Speak_demo") 
print(arch.api_status)


# Create an agent with the given architecture
agent = ao.Agent(arch, uid="Test12", save_meta=True)
agent.api_reset_compatibility = True # to enable similar behavior in local core with reset states as ao_python when running cross-compatible scripts


# Setting a baseline by pre-training example patterns that are known to be fraudulent. 
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
###Uncomment to train the agent
for inp, label in training_data:
    agent.next_state(INPUT=inp, LABEL=label, unsequenced=True)  # Reset states and unsequenced True


# Input to the agent
linkedin = [0]    # linkedin not active, extract from api
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
# Extracting features for input (using an LLM here - we can use other APIs)
response= ast.literal_eval(llm_call(f"""I am attaching a resume to this chat.Fill out this list with 1 OR 0 of length 7 Then return the list only .Format: [Zero GitHub or Personal Projects Listed, Buzzword Soup for Skills, 
#          Generic Role Descriptions, Inconsistent or Shady Company Info, Job Titles Don’t Match Timeline, 
#          Too Many Freelance Projects with No Clients Named, Resume Format Looks AI-Generated or Translated] {resume} 
                       """))
print("LLM response: ", response)
print(type(response))


# Changing input to binary for our system (we don't need the raw data; you keep the encoding)
input_to_agent = []
input_to_agent.append(linkedin[0])
input_to_agent.extend(bit for i, bit in enumerate(response))
print("input to agent: ", input_to_agent)


# Predicting the likelihood of fraud based on the input
agent_response = agent.next_state(input_to_agent)
print("agent response: ", agent_response)
ones = sum(agent_response)
print("Predicted likelihood of fraud: ", ones / len(agent_response) * 100, "%")


# Closing the Learning Loop - passing feedback to the system to drive learning
res = input("Closing the Learning Loop-- was this input-pattern actually fraud (Y or N)?  ")
if res == "Y":
    agent.next_state(input_to_agent, [1, 1, 1, 1, 1])
else:
    agent.next_state(input_to_agent, [0, 0, 0, 0, 0])