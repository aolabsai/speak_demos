# Python standard libraries
import ast

# Third-party libraries
import requests
import numpy as np

# AO library
# import ao_pyth as ao # $ pip install ao_pyth - https://pypi.org/project/ao-pyth/
import ao_core as ao # private package, to run our code locally, useful for advanced debugging; ao_pyth is enough for most use cases


# Importing API keys
from config import ao_apikey, openai_apikey




                                    # ----------- Helper Functions -----------#


def convert_to_binary(input_to_agent_scaled, scale=10):
    input_to_agent = []
    if type(scale) == list:
        s=0
        for i in input_to_agent_scaled:
            likelihood = np.zeros(scale[s], dtype=int)
            s+=1
            likelihood[0:i] = 1
            input_to_agent += likelihood.tolist()
    else:
        for i in input_to_agent_scaled:
            likelihood = np.zeros(scale, dtype=int)
            likelihood[0:i] = 1
    return input_to_agent




                                    # ----------- Initialize AO Agent -----------#

testing_data = [1, 0, 1, 4, 0, 0]

# Name - is there any pattern here to be learned? Might not be relevant.

# Email - 6 neurons - Speak TODO  if/then that validations to categorize if it looks like a fraudulent email or not this should be easy
    # if not pingable/reachable: 1
    # if has numbers: 1
    # if has more than one special characters: 1
    # if domain not in set(.com, .org, .net, .edu): 3
    # if domain not in set(<defined by speak>): 

# Title - 6 neurons
    # if level not in set(intern, junior, senior, vice president ... <up to 8 levels>): 3  - each level is a binary id, eg: none"000", intern="001", junior="010", senior="100", vice president="111"
    # if function not in set(software engineer, data scientist, project manager ... <up to 8 functions>): 3  - each function is a binary id, eg: none="000", software engineer="001", data scientist="010", project manager="100", vice president="111"

# LinkedIn presence - 2 neurons
    # if LinkedIn not is included: 1
    # if LinkedIn not is valid: 1   Speak TODO - LinkedIn <> is LinkedIn valid? E.g. 200 vs a 404 HTTP -network responses, you’ll need to ping LinkedIn and store the yes or no response 200=yes, 404=no

# Phone number - 4 neurons
    # if phone number country code does not matches candidate country location: 1
    # if phone number is not valid: 1   Speak TODO setup some if/then that validations, like with the email
    # if phone number is not reachable: 1
    # if phone number is Google voice or other internet number: 1

# Company they are applying to - 1 neuron
    # if name in application is not actual company name: 1

# Job by they are applying to - 2 neurons
    # if level not in job applied to: 1
    # if function not in job applied to: 1


# Maybe include school?
# Maybe 



# Initialize AO agent architecture, here with 30 input neurons and 5 output neurons. 
# Input consists of 3 features, each given on a intensity (or other) scale of 0-10 (10 neurons for each feature):
# Output consists of 5 neurons corresponding to a single scale of 1-5 (or whatever output(s) you want to associate with input).

arch = ao.Arch(arch_i="[6, 6, 2, 4, 1, 2]", arch_z="[10]", api_key=ao_apikey, kennel_id="Speak_demo") # --> architecture setup
agent = ao.Agent(arch, uid="Speak's client company X", save_meta=True)  # --> agent creation

agent.api_reset_compatibility = True



                                    # ----------- Pre-train with Baseline Examples -----------#

# Optional - Use this to train the agent on a baseline (if the agent has no prior training, it would output random;
# if it only has 1 label/training event, it can only ever output that until trained on more examples)

training_data = [
    [[6, 6, 2, 4, 1, 2], [10]],     # Highest fraud likelihood
    [[0, 0, 0, 0, 0, 0], [0]]      # Lowest fraud likelihood
]
for inp, label in training_data:
    inp = convert_to_binary(inp, scale=[6, 6, 2, 4, 1, 2])
    label = convert_to_binary(label, scale=5)
    agent.next_state(INPUT=inp, LABEL=label, unsequenced=True)  # unsequenced is default; you can set it to `False` to run on data that is sequential





                                    # ----------- Inference on Content (using YouTube as an example) -----------#

testing_data = [1, 0, 1, 4, 0, 0]  # new candidate with this set of inputs, AO system will predict % fraud

# converting input to binary
input_to_agent = convert_to_binary(testing_data, scale=[6, 6, 2, 4, 1, 2])

# # Initial prediction, predicting the likelihood of infringement based on the binary input
agent_response = agent.next_state(input_to_agent, unsequenced=True)
print("Agent raw binary response: ", agent_response)
print("Response percentage: ", sum(agent_response) / len(agent_response) * 100, "%")



                                    # ----------- Feedback Loop -----------#

# Closing the Learning Loop - passing feedback to the system to drive learning positively or negatively
res = input("Closing the Learning Loop-- was this input-pattern actually infringement (Y or N)?  ")
if res == "Y":
    agent.next_state(input_to_agent, LABEL=[1,1,1,1,1,1,1,1,1,1], unsequenced=True)
else:
    agent.next_state(input_to_agent, LABEL=[0,0,0,0,0,0,0,0,0,0], unsequenced=True)

# Re-evaluate After Feedback. To verify the learning, predict infringement again on the SAME input-pattern
agent_response = agent.next_state(input_to_agent, unsequenced=True)
print("Agent raw binary response: ", agent_response)
print("AFTER LEARNING LOOP, response percentage: ", sum(agent_response) / len(agent_response) * 100, "%")