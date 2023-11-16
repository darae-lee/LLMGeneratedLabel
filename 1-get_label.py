from transformers import AutoProcessor, AutoModelForVision2Seq
import torch
from PIL import Image
import csv
import pandas as pd
import argparse

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
    print(generated_text)

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

def main(label_type, context_type):
    processor = AutoProcessor.from_pretrained("instructblip_processor")
    print("processor downloaded")
    model = AutoModelForVision2Seq.from_pretrained("instructblip_model", load_in_8bit=True)
    print("model downloaded")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    description_file = pd.read_csv('./description.csv')

    header = ["image_id", "type"]
    output_file = open(f"./output_context{context_type}_type{label_type}.csv", 'w', newline='')
    output_writer = csv.writer(output_file)
    output_writer.writerow(header)

    label_file = open(f"./label_context{context_type}_type{label_type}.csv", 'w', newline='')
    label_writer = csv.writer(label_file)
    label_writer.writerow(["image_id","urban","rural","environment"])

    data_path = "./dataset/"
    data_count = 1

    while data_count <= 1000:
        image_path = data_path + str(data_count) + '.png'
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

        print(prompt)

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

        row = [data_count, generated_text]
        output_writer.writerow(row)

        generated_label = get_label(generated_text)
        label_writer.writerow([data_count] + generated_label)

        data_count += 1

    output_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate labels from images.")
    parser.add_argument("--type", type=int, help="Specify the type (0, 1, 2, 3, or 4).")
    parser.add_argument("--context", type=int, help="Specify the context (0, 1, 2 or 3).")
    args = parser.parse_args()

    main(args.type, args.context)