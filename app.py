import streamlit as st
from PIL import Image
import torch
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer

@st.cache(allow_output_mutation=True)
def load_model():
    model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, feature_extractor, tokenizer, device


def main():
    st.title("Image Captioning with ViT-GPT2")
    st.text("Upload an image and generate a caption.")

    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        model, feature_extractor, tokenizer, device = load_model()

        image = Image.open(uploaded_file)
        if image.mode != "RGB":
            image = image.convert(mode="RGB")

        st.image(image, caption="Uploaded Image", use_column_width=True)

        image_tensor = feature_extractor(images=[image], return_tensors="pt").pixel_values.to(device)

        max_length = 16
        num_beams = 4
        gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

        with torch.no_grad():
            output_ids = model.generate(image_tensor, **gen_kwargs)

        preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        preds = [pred.strip() for pred in preds]

        st.subheader("Generated Caption:")
        for pred in preds:
            st.write(pred)


if __name__ == "__main__":
    main()
