import gradio as gr
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import re

# 1. Setup Model
MODEL_ID = "purvansh01/conflict-env-final"

print("🚀 Loading Model... This may take a few minutes on first run.")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Robust loading for HF Spaces (Handles CPU/GPU automatically)
try:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, 
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        low_cpu_mem_usage=True,
        device_map="auto"
    )
except Exception as e:
    print(f"⚠️ GPU Loading failed, falling back to CPU: {e}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, 
        low_cpu_mem_usage=True,
        device_map="cpu"
    )

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

def resolve_conflict(scenario):
    if not scenario:
        return "Please enter a scenario.", "Waiting for input..."
    
    prompt = f"""<|im_start|>system
You are an Elite Executive Assistant. Resolve conflicts using deep reasoning.
Always look for efficient 3rd-party solutions (like Uber, mechanics, or delegation).
<|im_end|>
<|im_start|>user
{scenario}
<|im_end|>
<|im_start|>assistant
<thought>
"""
    
    # Inference
    outputs = pipe(prompt, max_new_tokens=512, do_sample=False)
    raw_text = outputs[0]["generated_text"]
    
    # Split reasoning and response
    try:
        assistant_part = raw_text.split("<|im_start|>assistant")[-1]
        
        # Extract Thought
        if "</thought>" in assistant_part:
            thought = assistant_part.split("</thought>")[0].replace("<thought>", "").strip()
            final_response = assistant_part.split("</thought>")[-1].strip()
        else:
            thought = assistant_part[:300] + "..."
            final_response = assistant_part
            
        # Logic Bridge
        if "delegate" in final_response.lower():
            status = "💡 STRATEGY: DELEGATION RECOMMENDED"
        elif "reschedule" in final_response.lower():
            status = "📅 STRATEGY: RESCHEDULING RECOMMENDED"
        else:
            status = "✅ STRATEGY: ACTION REQUIRED"

        return thought, f"{status}\n\n{final_response}"
    except:
        return "Parsing reasoning...", raw_text

# 2. Premium UI Design
theme = gr.themes.Soft(primary_hue="indigo")

with gr.Blocks(theme=theme, title="ConflictEnv Agent") as demo:
    gr.Markdown("# 🤖 ConflictEnv: The Strategic Executive Agent")
    
    with gr.Row():
        with gr.Column(scale=1):
            input_text = gr.Textbox(label="Describe your Conflict Scenario", lines=5)
            submit_btn = gr.Button("🧠 Analyze & Resolve", variant="primary")
        
        with gr.Column(scale=2):
            with gr.Accordion("🔓 Step-by-Step Reasoning", open=True):
                thought_output = gr.Markdown("Waiting...")
            final_output = gr.Textbox(label="Action & Output", lines=8, interactive=False)

    submit_btn.click(fn=resolve_conflict, inputs=input_text, outputs=[thought_output, final_output])

if __name__ == "__main__":
    demo.launch()
