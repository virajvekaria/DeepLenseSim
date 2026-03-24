from setuptools import find_packages, setup

with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
   name='DeepLenseSim',
   version='0.1',
   description='Simulations for DeepLense Project',
   license='MIT',
   long_description=long_description,
   long_description_content_type='text/markdown',
   author='Michael W. Toomey',
   author_email='michael_toomey@brown.edu',
   packages=find_packages(),
   extras_require={
       'agent': [
           'pydantic>=2.10',
           'pydantic-ai>=1.70.0',
           'google-genai>=1.68.0',
           'openai>=2.29.0',
           'pillow>=10.0.0',
       ],
   },
   entry_points={
       'console_scripts': [
           'deeplense-agent=deeplense_agent.cli:main',
       ],
   },
)
