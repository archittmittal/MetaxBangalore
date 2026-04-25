import gradio as gr
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import re

# 1. Setup Model
MODEL_ID = "purvansh01/conflict-env-final"

print("🚀 Loading Model... This may take a few minutes on first run.")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, 
    torch_dtype=torch.bfloat16, 
    device_map="auto",
    low_cpu_mem_usage=True
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
    
    print(f"🧠 Processing scenario: {scenario[:50]}...")
    
    # Inference
    outputs = pipe(prompt, max_new_tokens=1024, do_sample=False)
    raw_text = outputs[0]["generated_text"]
    
    # Split reasoning and response
    assistant_part = raw_text.split("<|im_start|>assistant")[-1]
    
    # Extract Thought
    thought_match = re.search(r'<thought>(.*?)</thought>', assistant_part, re.DOTALL)
    thought = thought_match.group(1).strip() if thought_match else "Reasoning not captured."
    
    # Clean up the final response (the part after thought)
    final_response = assistant_part.split("</thought>")[-1].strip()
    
    # Logic Bridge: If model didn't give strict JSON, we format it for the UI
    if "delegate" in final_response.lower() or "assignee" in final_response.lower():
        status = "💡 STRATEGY: DELEGATION RECOMMENDED"
    elif "reschedule" in final_response.lower() or "new_time" in final_response.lower():
        status = "📅 STRATEGY: RESCHEDULING RECOMMENDED"
    elif "cancel" in final_response.lower():
        status = "🚫 STRATEGY: CANCELLATION RECOMMENDED"
    else:
        status = "✅ STRATEGY: PROCEED WITH CAUTION"

    return thought, f"{status}\n\n{final_response}"

# 2. Premium UI Design
theme = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="slate",
    neutral_hue="slate",
).set(
    body_background_fill="*neutral_50",
    block_background_fill="white",
    block_border_width="1px",
    block_label_text_weight="600",
)

with gr.Blocks(theme=theme, title="ConflictEnv Agent") as demo:
    gr.Markdown("""
    # 🤖 ConflictEnv: The Strategic Executive Agent
    ### Deep Reasoning for Complex Life-Work Conflicts
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            input_text = gr.Textbox(
                label="Describe your Conflict Scenario",
                placeholder="Ex: Anniversary dinner vs. Server crash vs. Gym...",
                lines=5
            )
            submit_btn = gr.Button("🧠 Analyze & Resolve", variant="primary")
            
            gr.Examples(
                examples=[
                    ["[SCENARIO]TRAVEL_EMERGENCY: Wife's car broke down at 5 PM. Investor Pitch at 5:15 PM. Team meeting at 4:30 PM."],
                    ["[SCENARIO]SOCIAL_MINEFIELD: 10th Anniversary Dinner at 8 PM (non-refundable). Production server crash at 7:45 PM. Gym at 6:30 PM."]
                ],
                inputs=input_text
            )

        with gr.Column(scale=2):
            with gr.Accordion("🔓 Step-by-Step Reasoning (The Thinking Process)", open=True):
                thought_output = gr.Markdown("Waiting for input...")
            
            with gr.Group():
                gr.Markdown("### 🎯 Final Agentic Decision")
                final_output = gr.Textbox(label="Action & Output", lines=8, interactive=False)

    submit_btn.click(
        fn=resolve_conflict,
        inputs=input_text,
        outputs=[thought_output, final_output]
    )

    gr.Markdown("""
    ---
    **Technical Specs:** Trained on Qwen-1.5B using GRPO (Reinforcement Learning). 
    Optimized for *Strategic Trade-offs* and *Social Intelligence*.
    """)

if __name__ == "__main__":
    demo.launch()
