from setuptools import setup, find_packages

setup(
    name='image_captioning',
    version='1.0',
    author='Your Name',
    author_email='your_email@example.com',
    description='Image Captioning with ViT-GPT2',
    packages=find_packages(),
    install_requires=[
        'torch',
        'torchvision',
        'transformers',
        'streamlit',
        'Pillow',
    ],
)
