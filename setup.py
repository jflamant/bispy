from setuptools import setup

setup(name='bispy',
      version='1.0',
      description="BiSPy",
      long_description="",
      author='TODO',
      author_email='todo@example.org',
      license='TODO',
      packages=['bispy'],
      zip_safe=False,
      provides=['bispy'],
      install_requires=['numpy', 'scipy', 'matplotlib', 'quaternion'
          # 'Sphinx',
          # ^^^ Not sure if this is needed on readthedocs.org
          # 'something else?',
          ],
      )