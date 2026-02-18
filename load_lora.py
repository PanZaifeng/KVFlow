import requests


url = "http://127.0.0.1:8001"
lora_path = "qingpingwan/Qwen2.5-7B-Lora-Law"
agent_num = 200

for i in range(agent_num + 1):
    lora_name = f"lora{i}"
    response = requests.post(
        url + "/load_lora_adapter",
        json={
            "lora_name": lora_name,
            "lora_path": lora_path,
        },
    )
    if response.status_code == 200:
        print(f"LoRA adapter {lora_name} loaded successfully.")
    else:
        print(f"Failed to load LoRA adapter {lora_name}.", response.json())
