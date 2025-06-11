import ao_pyth as ao # $ pip install ao_pyth - https://pypi.org/project/ao-pyth/
from config import ao_apikey


# The True/False values returned from your LLM prompt need to be changed to binary
""" + """
{
"has_tools_and_clear_tech_stack": "1 if True or 0 if False",
"has_companies_with_external_footprint": "1 if True or 0 if False",
"has_clear_responsibilities": "1 if True or 0 if False",
"has_metrics_and_outcomes": "1 if True or 0 if False",
"has_valid_email": "1 if True or 0 if False",
"risk_level": "Risk level input here. Only 1, 2, 3 or 4.",
"risk_level_explanation": "Risk level explanation input here. For the ema"
}


# Extracting communication data (email) - using a 3rd party API
# https://apilayer.com/marketplace/email_verification-api
email_apikey = ""

# email = "alebrahim.ali@gmail.com"
# email = "ali@aolabs.ai"

def getCommEmail(applicant_info, email="ali@aolabs.ai"):

    import requests
    
    url = f"http://apilayer.net/api/check?access_key={email_apikey}&email={email}&smtp=1&format=1"
    response = requests.get(url)
    data = response.json()

    # changing data to binary, storing it all in applicant_info dictionary

    if data.get("mx_found", False) and data.get("smtp_check", False):
        applicant_info["email_valid"] = 1
    else:
        applicant_info["email_valid"] = 0

    if data.get("disposable", False):
        applicant_info["email_disposable"] = 1
    else:
        applicant_info["email_disposable"] = 0

    if data.get("free", False):
        applicant_info["email_free"] = 1
    else:
        applicant_info["email_free"] = 0

    score = data.get("score", 0.0)

    if score < 0.3:
        applicant_info["email_score"] = "00"
    elif score < 0.6:
        applicant_info["email_score"] = "01"
    else:
        applicant_info["email_score"] = "11"


    return applicant_info


# # Extracting social data (linkedin) input - using a 3rd party api
# # https://rapidapi.com/freshdata-freshdata-default/api/fresh-linkedin-profile-data/playground

# # li_url = "https://www.linkedin.com/in/alebrahimali/"

# def getSocialLinkedin(applicant_info, li_url="https://www.linkedin.com/in/alebrahimali/"):

#     if li_url == "":
#         li_url = applicant_info["linkedin_url"]

#     url = "https://fresh-linkedin-profile-data.p.rapidapi.com/get-linkedin-profile"

#     querystring = {"linkedin_url":li_url,"include_skills":"false","include_certifications":"false","include_publications":"false","include_honors":"false","include_volunteers":"false","include_projects":"false","include_patents":"false","include_courses":"false","include_organizations":"false","include_profile_status":"false","include_company_public_url":"false"}

#     headers = {
#         "x-rapidapi-key": rapid_apikey,
#         "x-rapidapi-host": "fresh-linkedin-profile-data.p.rapidapi.com"
#     }

#     li_data = requests.get(url, headers=headers, params=querystring)

#     applicant_info["linkedin_exists"] = "0"
#     applicant_info["linkedin_following"] = "0000"
#     applicant_info["linkedin_is_creator"] = "0"
#     applicant_info["linkedin_is_influencer"] = "0"
#     applicant_info["linkedin_is_premium"] = "0"
#     applicant_info["linkedin_is_verified"] = "0"


#     if li_data.status_code != 200:
#         print("LinkedIn profile not reachable, unable to verify.")
        
#     else:
#         li_data = li_data.json()["data"]
    
#         applicant_info["linkedin_exists"] = "1"

#         base_following = 100
#         if li_data["follower_count"] < base_following / 3:
#             applicant_info["linkedin_following"] = "0000"
#         elif li_data["follower_count"] >= base_following / 3 and li_data["follower_count"] < base_following / 3 * 2:
#             applicant_info["linkedin_following"] = "1100"
#         elif li_data["follower_count"] >= base_following / 3 * 2:
#             applicant_info["linkedin_following"] = "1110"
#         elif li_data["follower_count"] > base_following:
#             applicant_info["linkedin_following"] = "1111"

#         if li_data["is_creator"]:
#             applicant_info["linkedin_is_creator"] = "1"

#         if li_data["is_influencer"]:
#             applicant_info["linkedin_is_influencer"] = "1"

#         if li_data["is_premium"]:
#             applicant_info["linkedin_is_premium"] = "1"

#         if li_data["is_verified"]:
#             applicant_info["linkedin_is_verified"] = "1"

#     return applicant_info



# Combing all data (semantic, communication, social) in binary as INPUT to learning loop

def getAOAgentInput(applicant_info):

    agent_input= list(
        str(applicant_info["has_tools_and_clear_tech_stack"])+
        str(applicant_info["has_companies_with_external_footprint"])+
        str(applicant_info["has_clear_responsibilities"])+
        str(applicant_info["has_metrics_and_outcomes"])+
        # str(applicant_info["linkedin_exists"])+
        # str(applicant_info["linkedin_following"])+
        # str(applicant_info["linkedin_is_creator"])+
        # str(applicant_info["linkedin_is_influencer"])+
        # str(applicant_info["linkedin_is_premium"])+
        # str(applicant_info["linkedin_is_verified"])+
        str(applicant_info["email_valid"])+
        str(applicant_info["email_disposable"])+
        str(applicant_info["email_free"])+
        str(applicant_info["email_score"])
    )

    agent_input = list("".join(agent_input))

    return agent_input


                                    # ----------- Initialize AO Agent -----------#

# Initialize AO agent architecture, here with 28 input neurons and 5 output neurons.
# Output consists of 5 neurons corresponding to a single scale of 1-5 (or whatever output(s) you want to associate with input).

# agent_size = [len(agent_input)]
agent_size = [9]

arch = ao.Arch(arch_i=agent_size, arch_z=[10], api_key=api_key, email="gustavo@speak.ai", kennel_id="Speak_demo_01") # --> architecture setup

uid="Speak Dev 0" # change this variable name for each Speak user, to made a separate model for each user
agent = ao.Agent(arch, uid="Speak Dev 0")  # --> agent creation


                                    # ----------- Inference on Candidate (to get fraud %) -----------#

# # Initial prediction, predicting the likelihood of infringement based on the binary input

test_input = "110101010" # test input of 9 binary digits
test_input = [0,0,0,0,0,0,0,0,0] # can also be a list

agent_response = agent.next_state(test_input, unsequenced=True)
agent_response_percentage = sum(agent_response) / len(agent_response) * 100


                                    # ----------- Training on Candidate -----------#

train_input = "110101010" # 0% fraud

train_label = "0000000000" # 0% fraud
train_label = "1111111111" # 100% fraud

# # Initial prediction, predicting the likelihood of infringement based on the binary input
agent.next_state(train_input, LABEL=train_label, unsequenced=True)