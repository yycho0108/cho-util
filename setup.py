from setuptools import setup, find_packages

print('pkg -> {}'.format(find_packages()))

setup(name='cho-util',
      version='0.1',
      description='My python utilities',
      url='http://github.com/yycho0108/cho-util.py',
      author='Jamie Cho',
      author_email='jchocholate@gmail.com',
      license='MIT',
      packages=find_packages(),
      zip_safe=False,
      scripts=['bin/pyhelp']
      )
