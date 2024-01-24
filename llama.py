import os
import lamini
from dotenv import load_dotenv

load_dotenv()

api_key = os.environ.get("LAMINI_API_KEY")

lamini.api_key = os.environ.get("LAMINI_API_KEY")

llm = lamini.LlamaV2Runner()

system_prompt="You are an assistant."
instruction = "How are you?"

prompt = f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{instruction}[/INST]"

print(llm(prompt))
