import torch
import streamlit as st
from PIL import Image, ImageDraw
from transformers import AutoProcessor, AutoModelForCausalLM


st.set_page_config(page_title="Florence-2 Vision AI", layout="centered")

@st.cache_resource
def load_model():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch_dtype = torch.float32
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        torch_dtype = torch.float32
    else:
        device = torch.device("cpu")
        torch_dtype = torch.float32

    model_id = "microsoft/Florence-2-base"
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        trust_remote_code=True, 
        torch_dtype=torch_dtype,
        attn_implementation="eager"
    ).to(device)
    
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    return model, processor, device, torch_dtype

st.title("Florence-2 Vision AI App")

with st.spinner("Loading Florence-2 model..."):
    model, processor, device, torch_dtype = load_model()

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    task_prompt = st.selectbox(
        "Select Vision Task",
        ["<CAPTION>", "<DETAILED_CAPTION>", "<MORE_DETAILED_CAPTION>", "<OD>"]
    )

    if st.button("Run Inference"):
        with st.spinner("Processing image..."):
            # Pass original image directly to processor without manual resizing
            inputs = processor(text=task_prompt, images=image, return_tensors="pt")
            
            # Cast inputs to device and appropriate dtype
            inputs = {
                k: v.to(device=device, dtype=torch_dtype if v.dtype == torch.float32 else v.dtype) 
                for k, v in inputs.items()
            }

            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=512,
                num_beams=3,
                do_sample=False,
                use_cache=False
            )

            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
            
            parsed_answer = processor.post_process_generation(
                generated_text, 
                task=task_prompt, 
                image_size=(image.width, image.height)
            )

            st.subheader("Results")
            
            if task_prompt == "<OD>":
                bbox_image = image.copy()
                draw = ImageDraw.Draw(bbox_image)
                od_results = parsed_answer.get("<OD>", {})
                
                boxes = od_results.get("bboxes", [])
                labels = od_results.get("labels", [])

                for box, label in zip(boxes, labels):
                    x1, y1, x2, y2 = box
                    draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
                    draw.text((x1, y1 - 10 if y1 > 10 else y1), label, fill="red")

                st.image(bbox_image, caption="Detected Objects", use_container_width=True)
            else:
                st.write(parsed_answer.get(task_prompt, generated_text))