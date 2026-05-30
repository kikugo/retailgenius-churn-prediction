RetailGenius Churn Prediction
=============================

Documentation for the e-commerce customer churn prediction project
(EPITA AI Project Methodology, Parts 2 and 3).

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   api

Pipeline overview
-----------------

The project is split into four stages, each its own module:

#. ``churn.data.make_dataset`` -- load and clean the raw Excel file.
#. ``churn.features.build_features`` -- encode and split into train/test.
#. ``churn.models.train_model`` -- train models and track them in MLflow.
#. ``churn.explain.shap_explain`` -- explain predictions with SHAP.

Indices and tables
-------------------

* :ref:`genindex`
* :ref:`modindex`
