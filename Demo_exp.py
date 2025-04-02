import ao_pyth as ao
from config import API_KEY

# Initialize architecture with predefined 4 input neurons, 4 hidden neurons, 5 output neurons. 
# The 5 output neurons correspond to the likelihood of buying (scale 1-5)
arch = ao.Arch(arch_i="[1, 1, 1, 1]", arch_z="[1, 1, 1, 1, 1]", api_key=API_KEY, kennel_id="Parsed_DEMO3") 
print(arch.api_status)

# Create an agent for each user with the given architecture
agent = ao.Agent(arch, uid="ContLearn4")

# Training examples: 
# Format: [Payment setup, Item in basket, User logged in, User new] -> Likelihood of buying (scale 1-5)
training_data = [
    ([1, 1, 1, 0], [1, 1, 1, 1, 1]),  # High likelihood (returning user, logged in, has item)
    ([1, 1, 0, 0], [1, 1, 1, 0, 1]),  # Medium likelihood (not logged in)
    ([0, 1, 1, 0], [1, 1, 0, 1, 0]),  # Medium-low likelihood (no payment setup)
    ([1, 0, 1, 1], [1, 0, 1, 0, 0]),  # Low likelihood (no item in basket)
    ([0, 0, 0, 1], [0, 0, 1, 0, 0]),  # Very low likelihood (new user, no setup, no item)
    ([1, 1, 1, 1], [1, 1, 1, 1, 1]),  # Highest likelihood (everything is optimal)
    ([0, 1, 1, 1], [1, 1, 0, 1, 0]),  # Medium-low likelihood (new user)
    ([1, 0, 0, 0], [0, 1, 0, 0, 0]),  # Very low likelihood (no basket, not logged in)
    ([1, 1, 0, 1], [1, 0, 1, 0, 1]),  # Medium likelihood (new user but has setup)
    ([1, 0, 1, 0], [0, 1, 0, 1, 0])   # Low likelihood (payment setup User logged in)
]

# Train the agent with the examples
###Uncomment to train the agent
# for inp, label in training_data:
#     agent.next_state(INPUT=inp, LABEL=label, unsequenced=True)  # Reset states and unsequenced True


# Experimental -- Incremental labels for more granular continuous learning

response = agent.next_state([1, 0, 1, 0], unsequenced=True) 
ones = sum(response)

print("Predicted likelihood of buying: ", ones / len(response) * 100, "%")

i = input("did the user buy? Y/N: ")

if i == "Y":
    ones = sum(response)
    Label = [1]*(ones+1)
    Label += [0]*(4-ones)
    print("old response: ", response)
    print("new label: ", Label)
    agent.next_state([1, 0, 1, 0], Label, unsequenced=True)  # Retrain the agent with the new label

if i == "N":
    ones = sum(response)
    Label = [1]*(ones-1)
    Label += [0]*(6-ones)
    print("old response: ", response)
    print("new label: ", Label)
    agent.next_state([1, 0, 1, 0], Label, unsequenced=True) 