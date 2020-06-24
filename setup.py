from setuptools import setup

setup(name="fly_eye",
      version='0.3.1',
      description='tools for anayzing images and video (as image stacks),\
      particularly for motion tracking, color keying, and processing of fruit\
      fly eye images.',
      url="https://github.com/jpcurrea/fly_eye.git",
      author='Pablo Currea',
      author_email='johnpaulcurrea@gmail.com',
      license='MIT',
      packages=['fly_eye', 'rolling_window'],
      install_requires=[
          'numpy',
          'scipy',
          'matplotlib',
          'scikit-image',
          'opencv-python'
      ],
      zip_safe=False)
