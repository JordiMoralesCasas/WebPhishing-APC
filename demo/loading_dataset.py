import sys
sys.path.insert(1, '../src/helpers')
sys.path.insert(1, '../src')
from imports import *
from generate_features import *

x, y = load_dataset('..\data\external\dataset_phishing.csv')

## DATA PREPROCESSING

# Replacing uknown values with the mean of the known values. Just "domain_age" and 
# "registration_length" seems
# to have unknown values
known_domain_age = x.loc[x['domain_age'] != -1, 'domain_age']
mean_domain_age = known_domain_age.sum()/len(known_domain_age)

x.loc[x["domain_age"] == -1, 'domain_age'] = mean_domain_age

known_domain_registration_length = x.loc[x['domain_registration_length'] != -1, 'domain_registration_length']
mean_domain_registration_length = known_domain_registration_length.sum()/len(known_domain_registration_length)

x.loc[x["domain_registration_length"] == -1, 'domain_registration_length'] = mean_domain_registration_length


## DATA CONVERSION

# Converting "status" variable to binary variable
y = y.replace({"phishing" : 1, "legitimate" : 0})

# The "url" feature is not useful since all the information that can be extracted from it is already collect in
# the rest of columns
x = x.drop(columns = "url")

# The x columns with float numbers are actually stored as an unknown _object_ variable. A conversion to float
# type is needed
x = object_type_to_float(x)

## DATA STANDARDIZATION

# First we have to determine which features are categorical (Only non-categorical data should be standardized)
categorical_features = ["ip", "http_in_path", "https_token", "punycode", "port", "tld_in_path", "tld_in_subdomain",\
"abnormal_subdomain", "prefix_suffix", "random_domain", "shortening_service", "path_extension", "domain_in_brand",\
"brand_in_subdomain", "brand_in_path", "suspecious_tld", "login_form", "external_favicon", "submit_email", "sfh",\
"iframe", "popup_window", "onmouseover", "right_clic", "empty_title", "domain_in_title", "domain_with_copyright",\
"whois_registered_domain", "dns_record", "google_index"]

x_standard = standardize(x, categorical_features = categorical_features)


## FEATURE SELECTION
# We will do a couple of experiments by selecting the 15 and 30 best features and training some models
best_atributes15 = select_features(x_standard, y, 15)
best_atributes30 = select_features(x_standard, y, 30)


## FINAL DATA

# Dataset with the selected features and standardized
reduced_dataset_standard15 = x_standard[x_standard.columns[best_atributes15]]
reduced_dataset_standard15 = reduced_dataset_standard15.assign(status = y.values)

reduced_dataset_standard30 = x_standard[x_standard.columns[best_atributes30]]
reduced_dataset_standard30 = reduced_dataset_standard30.assign(status = y.values)

# Dataset without feature selection (standardized and non-standardized)
Full_dataset_standard = x_standard.assign(status = y.values) #Full dataset (without feature selection)
Full_dataset = x.assign(status = y.values) #Full dataset (without feature selection)

# PCA performed over the data

# 90% explained variance
pca_dataset90 = perform_pca(Full_dataset_standard, auto=True, obj_variance=0.90)

# 95% explained variance
pca_dataset95 = perform_pca(Full_dataset_standard, auto=True, obj_variance=0.95)