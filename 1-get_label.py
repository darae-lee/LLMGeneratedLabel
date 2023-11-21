from transformers import AutoProcessor, AutoModelForVision2Seq
import torch
from PIL import Image
import csv
import pandas as pd
import argparse
import os

def get_description(model, processor, device, context, image):
    prompt = ""
    if context == 2: # short description
        prompt += "Use a few words to illustrate what is in the satellite image."
    elif context == 3: # detail description
        prompt += "Describe the image in detail."
    
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)

    outputs = model.generate(
                    **inputs,
                    num_beams=5,
                    max_length=256,
                    min_length=1,
                    top_p=0.9,
                    repetition_penalty=1.5,
                    length_penalty=1.0,
                    temperature=1,
                    )

    generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()

    return generated_text

def get_label(output):
    # return hard label (urban, rural, uninhabited area)
    if output[1] == "a":
        return [1, 0, 0]
    elif output[1] == "b":
        return [0, 1, 0]
    elif output[1] == "c":
        return [0, 0, 1]
    return

def main(model_root, processor_root, data_dir, output_dir, label_type, context_type):
    processor = AutoProcessor.from_pretrained(processor_root)
    print("processor downloaded")
    model = AutoModelForVision2Seq.from_pretrained(model_root)
    print("model downloaded")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    header = ["image", "type"]
    output_file = open(f"./{output_dir}/output_context{context_type}_type{label_type}.csv", 'w', newline='')
    output_writer = csv.writer(output_file)
    output_writer.writerow(header)

    label_file = open(f"./{output_dir}/label_context{context_type}_type{label_type}.csv", 'w', newline='')
    label_writer = csv.writer(label_file)
    label_writer.writerow(["image","urban","rural","environment"])

    data_path = f"{data_dir}/"
    image_files = [f for f in os.listdir(data_path) if f.endswith('.png')]

    for image_file in image_files:
        image_path = os.path.join(data_path, image_file)
        image_name, _ = os.path.splitext(os.path.basename(image_path))
        image = Image.open(image_path)
        context = ""
        prompt = ""

        if context_type == 0:
            context += ""
        elif context_type == 1:
            context += "Context: This is a satellite image of a certain area of Korea. "
        elif context_type == 2:
            description = get_description(model, processor, device, context_type, image)
            context += f"Context: This is {description}. "
        elif context_type == 3:
            description = get_description(model, processor, device, context_type, image)
            context += f"Context: This is {description}. "
        
        if label_type == 0: # basic
            prompt += f"{context}Question: What type of area is this image? Option: (a) urban (b) rural (c) environment. Answer:"
        elif label_type == 1: # population density
            prompt += f"{context}Question: What kind of place is in the picture? Option: (a) highly populated city (b) populated countryside (c) unpopulated area. Answer:"
        elif label_type == 2: # building
            prompt += f"{context}Question: What kind of place is in the picture? Option: (a) urban area with tall buildings (b) rural area with houses (c) unpopulated area. Answer:"
        elif label_type == 3: # peripheral environment
            prompt += f"{context}Question: What kind of area is in the picture? Option: (a) city with tall apartments (b) farmland (c) green or brown mountain. Answer:"
        elif label_type == 4: # development degree
            prompt += f"{context}Question: How developed is the area? Option: (a) highly developed city (b) developed rural (c) undeveloped environment. Answer:"

        inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)

        outputs = model.generate(
            **inputs,
            num_beams=5,
            max_length=256,
            min_length=1,
            top_p=0.9,
            repetition_penalty=1.5,
            length_penalty=1.0,
            temperature=1,
        )

        generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()

        row = [image_name, generated_text]
        output_writer.writerow(row)

        generated_label = get_label(generated_text)
        label_writer.writerow([image_name] + generated_label)

    output_file.close()
    label_file.close()
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate labels from images.")
    parser.add_argument("--type", type=int, help="Specify the type (0, 1, 2, 3, or 4).")
    parser.add_argument("--context", type=int, help="Specify the context (0, 1, 2 or 3).")
    parser.add_argument("--model", type=str)
    parser.add_argument("--processor", type=str)
    parser.add_argument("--d-dir", type=str)
    parser.add_argument("--o-dir", type=str)
    args = parser.parse_args()

    main(args.model, args.processor, args.d_dir, args.o_dir, args.type, args.context)