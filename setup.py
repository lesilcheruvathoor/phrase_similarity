from setuptools import setup, find_packages

setup(name='phrase_similarity',
      version='0.1',
      packages=find_packages(),
      install_requires=[
          'gensim'
          ,'pandas'
          ,'numpy'
          ,'sklearn'
          ,'csv'
          ,'sys'
      ],
      entry_points={
          'console_scripts':['phrase_similarity=main:main']
                    }
      )