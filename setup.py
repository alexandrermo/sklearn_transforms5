from setuptools import setup
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer


setup(
      name='my_custom_sklearn_transforms',
      version='1.0',
      description='''
            This is a sample python package for encapsulating custom
            tranforms from scikit-learn into Watson Machine Learning
      ''',
      url='https://github.com/vnderlev/sklearn_transforms/',
      author='Vanderlei Munhoz',
      author_email='vnderlev@protonmail.ch',
      license='BSD',
      packages=[
            'my_custom_sklearn_transforms'
      ],
      zip_safe=False
)
