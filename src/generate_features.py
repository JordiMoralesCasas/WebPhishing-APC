from imports import *

np.random.seed(1)
torch.manual_seed(1)

# Loading Dataset
ds_original = pd.read_csv(r'..\data\external\dataset_phishing.csv', header=0, delimiter=',',decimal=',')

# The objective is to predict the status of an url (Dependent variable)
y = ds_original[:]["status"] 
x = ds_original.drop(columns = "status")

## DATA PREPROCESSING

# Replacing uknown values with the mean of the known values. Just "domain_age" and "registration_length" seems
# to have unknown values
known_domain_age = x.loc[x['domain_age'] != -1, 'domain_age']
mean_domain_age = known_domain_age.sum()/len(known_domain_age)
mean_domain_age

x.loc[x["domain_age"] == -1, 'domain_age'] = mean_domain_age

known_domain_registration_length = x.loc[x['domain_registration_length'] != -1, 'domain_registration_length']
mean_domain_registration_length = known_domain_registration_length.sum()/len(known_domain_registration_length)
mean_domain_registration_length

x.loc[x["domain_registration_length"] == -1, 'domain_registration_length'] = mean_domain_registration_length


# The "url" feature is not useful sinse all the information that canbe extracted from it is already collect in
# the rest of columns
x = x.drop(columns = "url")


## DATA CONVERSION

# Converting "status" variable to binary variable
status_labels = y.copy() # "string" labels

y = y.replace({"phishing" : 1, "legitimate" : 0})


# The x columns with float numbers are actually stored as an unknown _object_ variable. A conversion to float
# type is needed
type_of_cols = [x.dtypes == object][0]
for col in range(x.shape[1]):
    if type_of_cols[col]:
        x[x.columns[col]] = x[x.columns[col]].astype('float')



## DATA STANDARDIZATION

# First we have to determine which features are categorical (Only non-categorical data should be standardized)
categorical_features = ["ip", "http_in_path", "https_token", "punycode", "port", "tld_in_path", "tld_in_subdomain",\
"abnormal_subdomain", "prefix_suffix", "random_domain", "shortening_service", "path_extension", "domain_in_brand",\
"brand_in_subdomain", "brand_in_path", "suspecious_tld", "login_form", "external_favicon", "submit_email", "sfh",\
"iframe", "popup_window", "onmouseover", "right_clic", "empty_title", "domain_in_title", "domain_with_copyright",\
"whois_registered_domain", "dns_record", "google_index"]#, "nb_www", "nb_slash", "nb_qm", "nb_hyperlinks", "nb_eq", "nb_dots"]
non_categorical_features = [i for i in x.columns if (i not in categorical_features)]

x_non_categorical = x[non_categorical_features]
x_categorical = x[categorical_features]

scaler = StandardScaler() # Standardize features by removing the mean and scaling to unit variance.
#scaler = MinMaxScaler()

x_non_categorical_standard = scaler.fit_transform(x_non_categorical)

df = pd.DataFrame(x_non_categorical_standard)
df.columns = non_categorical_features

x_standard = pd.concat([df, x_categorical], axis = 1) # Standardized data


## FEATURE SELECTION

# MIGHT DELETE!!
# A good idea for discarding some columns would be to drop those whose mean is very close to 0. We can do this
# only with the features that represent some kind of counter because that means that most of the samples doesn't
# register that feature.

#columns_to_drop = []
#for col in range(x.shape[1]):
#    name_of_feature = x.columns[col]
#    if abs(np.mean(x[name_of_feature])) <= 0.01:
#        columns_to_drop.append(name_of_feature)
# x[columns_to_drop].describe()

# For each atribute we will perform a linear regression so we can compute its r2 score. We will select the ones
# with a higher score
def split_data(x, y, train_ratio=0.8):
    indices = np.arange(x.shape[0])
    np.random.shuffle(indices)
    n_train = int(np.floor(x.shape[0]*train_ratio))
    indices_train = indices[:n_train]
    indices_val = indices[n_train:] 
    x_train = x[indices_train, :]
    y_train = y[indices_train]
    x_val = x[indices_val, :]
    y_val = y[indices_val]
    return x_train, y_train, x_val, y_val

# Split the data
x_train, y_train, x_val, y_val = split_data(x_standard.values, y)

r2_table = np.zeros((x_train.shape[1], 2))

# Linear regression for every attribute
for i in range(x_train.shape[1]):
    x_t = x_train[:,i] # seleccionem atribut i en conjunt de train
    x_v = x_val[:,i] # seleccionem atribut i en conjunt de val.
    x_t = np.reshape(x_t,(x_t.shape[0],1))
    x_v = np.reshape(x_v,(x_v.shape[0],1))

    regr = LinearRegression()
    regr.fit(x_t, y_train)
    
    r2 = r2_score(y_val, regr.predict(x_v))
    
    r2_table[i, 1] = r2
    r2_table[i, 0] = i

# Select the best atributes (15 atributes)
best_atributes = r2_table[r2_table[:, 1].argsort()[::-1]][:,0].astype('int')

# Fully processed data
reduced_dataset_standard = x_standard[x_standard.columns[best_atributes[:15]]]
reduced_dataset_standard = reduced_dataset_standard.assign(status = y.values)


reduced_dataset = x[x_standard.columns[best_atributes[:15]]]
reduced_dataset = reduced_dataset.assign(status = y.values)


X = reduced_dataset.drop(columns='status').values
X_standard = reduced_dataset_standard.drop(columns='status').values

Full_dataset = x_standard.assign(status = y.values) #Full dataset (without feature selection)

Y = reduced_dataset.filter(['status']).values

