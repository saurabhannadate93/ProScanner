Problem statement
The project involves building a model that recognizes text contained in the images of handwritten text-lines as well as segmented words, extracts the individual characters contained in the image and converts the text into letter codes which are usable within computer and text-processing applications.

Handwritten text recognition has great significance when it comes to the digitisation of pre-existing records and documents in Legal, Banking and HealthCare sectors resulting in documents that are Searchable, Editable, Easily Accessible and can be Stored efficiently. Some specific examples include:
Banking - Handling handwritten cheques through text recognition would result in reduced turnaround time for cheque clearance which is an economic gain for all, from payer to bank to payee.
Customer Service - Digitising handwritten feedback from valued customers would help maintain databases of customer inputs that could then be analyzed to
allow enterprises to enhance customer experience and better serve customers.
Healthcare - Unification of medical records of patients would lead to better health care by improving all aspects of patient care, including safety, effectiveness and patient-centeredness. 
Dataset
Dataset: IAM Handwriting Database
Source: http://www.fki.inf.unibe.ch/databases/iam-handwriting-database

The IAM Handwriting Database contains forms of handwritten English text which can be used to train and test handwritten text recognizers and to perform writer identification and verification Experiments. The IAM Handwriting database is free for non-commercial research purpose for registered users. Different datasets from IAM database has been extensively used for handwritten text recognition and cited in many OCR research papers, hence we believe the dataset is representative of the problem we are trying to address.

The offline handwriting dataset we have access to has 13,000 examples of single line sentences which comprise of ~110k individual words with around 80 characters(lowercase, uppercase, digits, symbols) in total. The database contains forms of unconstrained handwritten text, which were scanned at a resolution of 3000dpi and saved as PNG images with 256 gray levels.  As with any image dataset, we plan to utilize basic transformations (rotate,offset ) to enhance model performance. 
