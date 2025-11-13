from spectrumlab.models import Intern_S1
from spectrumlab.utils.image_utils import encode_image_to_base64


def test_intern_s1_text_generation():
    model = Intern_S1()
    prompt = "What is spectroscopy?"
    response = model.generate(prompt)
    assert isinstance(response, str)
    assert len(response) > 0


def test_intern_s1_multimodal_generation():
    model = Intern_S1()
    image_path = "/Users/little1d/Desktop/Code/SpectrumLab/playground/models/test.jpg"
    image_base64 = encode_image_to_base64(image_path)
    prompt = {
        "text": "Please explain this spectroscopy image.",
        "images": [
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{image_base64}"},
            }
        ],
    }
    response = model.generate(prompt)
    assert isinstance(response, str)
    assert len(response) > 0
