import requests
import json
import time

# Define the API endpoint
url = "http://ec2-43-198-104-59.ap-east-1.compute.amazonaws.com:11434/api/generate"
# url = "http://localhost:11434/api/generate"

headers = {"Content-Type": "application/json"}

# prompt = "Translate this sentence in clear English with no extra explanation or repetition. only the output without adding any verbosity like 'Translation:' etc: tum kya khaana chahoge?"
# prompt = "Translate this in English with no extra explanation or repetition. only the output without adding any verbosity like 'Translation:' etc: 我喜欢打乒乓球"
prompt = "I am a language translation agent and need to translate input to English. Respond with just the translated result without adding any verbosity like 'Translation:'. Text: 哈哈，明白你的感受！抱歉給你帶來了困擾。如果有什麼需要幫忙的，隨時可以告訴我，我會隨時準備好協助你！希望你有個愉快的一天"
# prompt = "Respond with just the answer without any extra verbosity \n\n from the data given below provide 3 names to store this conversation by, each of MAXIMUM 2 words and separate the names by === \n\nClick Me 10/26/2024, 12:14:20 AM acc_36dce712-a9eb-4275-96_29de 60173381136@s.whatsapp.net Harvard Chin Yihao 60173381136===Test===Wow amazing===sendgo===Test : sendgo \n10/29/2024, 9:51:22 AM \nacc_36dce712-a9eb-4275-96_29de \n60173381136@s.whatsapp.net \nfalse \n60173381136===10/30/2024, 9:41:09 AM \n60173381136===Hi===Zzz===hhhh===zz===Testing===testing no edit yet 1===testing no edit vv 2===edit later > edit now > edit 2 > Edit 3===send edit > edited\n The cat is, above all things, a dramatist; its life is lived in an endless romance though the drama is played out on quite another stage than our own, and we only enter into it as subordinate characters, as stage managers, or rather stage carpenters."
# prompt = "Extract exactly 3 two-word names/labels to identify/label this text. Format the output as: name1===name2===name3\n\nInput Text:\nClick Me 10/26/2024, 12:14:20 AM acc_36dce712-a9eb-4275-96_29de 60173381136@s.whatsapp.net Harvard Chin Yihao 60173381136===Test===Wow amazing===sendgo===Test : sendgo \n10/29/2024, 9:51:22 AM \nacc_36dce712-a9eb-4275-96_29de \n60173381136@s.whatsapp.net \nfalse \n60173381136===10/30/2024, 9:41:09 AM \n60173381136===Hi===Zzz===hhhh===zz===Testing===testing no edit yet 1===testing no edit vv 2===edit later > edit now > edit 2 > Edit 3===send edit > edited\n The cat is, above all things, a dramatist; its life is lived in an endless romance though the drama is played out on quite another stage than our own, and we only enter into it as subordinate characters, as stage managers, or rather stage carpenters."
# prompt = "i am a customer service agent tasked with extracting labels from a conversation to categorize it. extract exactly 3 two-word labels from the conversation and separate them by ===. Output only the labels.\n\nConversation:\nClick Me 10/26/2024, 12:14:20 AM acc_36dce712-a9eb-4275-96_29de 60173381136@s.whatsapp.net Harvard Chin Yihao 60173381136===Test===Wow amazing===sendgo===Test : sendgo \n10/29/2024, 9:51:22 AM \nacc_36dce712-a9eb-4275-96_29de \n60173381136@s.whatsapp.net \nfalse \n60173381136===10/30/2024, 9:41:09 AM \n60173381136===Hi===Zzz===hhhh===zz===Testing===testing no edit yet 1===testing no edit vv 2===edit later > edit now > edit 2 > Edit 3===send edit > edited\n The cat is, above all things, a dramatist; its life is lived in an endless romance though the drama is played out on quite another stage than our own, and we only enter into it as subordinate characters, as stage managers, or rather stage carpenters."
# prompt = (
#     "[INST] Follow these steps:\n"
#     + "1. Read the text below\n"
#     + "2. Create exactly 3 names (max 2 words each)\n"
#     + "3. Join them with === \n"
#     + "4. Output strictly in this format: name1===name2===name3\n\n"
#     + "Text: \nClick Me 10/26/2024, 12:14:20 AM acc_36dce712-a9eb-4275-96_29de 60173381136@s.whatsapp.net Harvard Chin Yihao 60173381136===Test===Wow amazing===sendgo===Test : sendgo \n10/29/2024, 9:51:22 AM \nacc_36dce712-a9eb-4275-96_29de \n60173381136@s.whatsapp.net \nfalse \n60173381136===10/30/2024, 9:41:09 AM \n60173381136===Hi===Zzz===hhhh===zz===Testing===testing no edit yet 1===testing no edit vv 2===edit later > edit now > edit 2 > Edit 3===send edit > edited\n The cat is, above all things, a dramatist; its life is lived in an endless romance though the drama is played out on quite another stage than our own, and we only enter into it as subordinate characters, as stage managers, or rather stage carpenters. \n[/INST]"
# )

data = {"prompt": prompt}

start_time = time.time()
# Make the POST request
response = requests.post(url, headers=headers, data=json.dumps(data))
end_time = time.time()
latency = end_time - start_time
print(f"Latency: {latency} seconds")

print("LLM response: " + response.text)
