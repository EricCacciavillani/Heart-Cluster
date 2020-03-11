
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from collections import Counter
from collections import defaultdict
import copy
import seaborn as sns
import pylab as pl
import scikitplot as skplt
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from scipy.stats import zscore
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, accuracy_score
from subprocess import check_call
from IPython.display import Image
from sklearn.tree import export_graphviz
import pydotplus


# ### Install dependencies

# In[2]:


# !pip install pandas
# !pip install seaborn
# !pip install sklearn
# !pip install scikit-plot


# ### Certificate Of Authenticity
# <b>Author:</b> Eric Cacciavillani
# <br>
# <b>Class:</b> DAT-330-01
# <br>
# <b>Date:</b> January 24, 2019
# <br>
# <b>Certification of Authenticity: </b>
# <br>
#  I certify that this is entirely my own work,
#  except where I have given fully documented
#  references to the work of others.
#  I understand the definition and consequences of
#  plagiarism and acknowledge that the assessor of this assignment may,
#  for the purpose of assessing this assignment reproduce this assignment
#  and provide a copy to another member of academic staff and / or communicate
#  a copy of this assignment to a plagiarism checking service(which may then
#  retain a copy of this assignment on its database for the purpose
#  of future plagiarism checking).

# ### Define functions to use for later use

# In[3]:


def cluster_count(clust):
    """
        Returns back dataframe of clustername to the count
    """
    cluster_count_df = pd.DataFrame(columns=['Cluster_Name', "Cluster_Count"])

    for cluster, count in Counter(clust.labels_).items():
        cluster_count_df = cluster_count_df.append({'Cluster_Name': cluster,
                                                    'Cluster_Count': count},
                                                   ignore_index=True)
    return cluster_count_df.sort_values(by=[
        'Cluster_Name']).reset_index(drop=True)


def find_nearest(numbers, target):
    """
        Find the closest fitting number to the target number
    """
    numbers = np.asarray(numbers)
    idx = (np.abs(numbers - target)).argmin()
    return numbers[idx]

# Uses a hash map to decode dataframe data


def encode_decode_df(passed_df, encoder_decoder_map):

    def encode_decode_col(data, decoder):
        return decoder[data]

    df = copy.deepcopy(passed_df)
    for col in df.columns:
        if col in encoder_decoder_map.keys():
            df[col] = np.vectorize(encode_decode_col)(
                df[col], encoder_decoder_map[col])

    return df


def remove_outliers_df(df, removal_dict):
    df = copy.deepcopy(df)

    for feature_name in df.columns:

        # Replacements needed
        if feature_name in removal_dict.keys():
            if removal_dict[feature_name]["High"]:
                df = df[df[feature_name] < removal_dict[feature_name]["High"]]
            elif removal_dict[feature_name]["Low"]:
                df = df[df[feature_name] > removal_dict[feature_name]["Low"]]

    return df.reset_index(drop=True)


def inspect_feature_matrix(matrix,
                           feature_names):
    scaled_mean_matrix = np.mean(scaled, axis=0)
    scaled_std_matrix = np.std(scaled, axis=0)
    scaled_data_dict = dict()
    for index, feature_name in enumerate(feature_names):
        scaled_data_dict[feature_name] = [scaled_mean_matrix[index],
                                          scaled_std_matrix[index]]

    return pd.DataFrame.from_dict(scaled_data_dict,
                                  orient='index',
                                  columns=['Mean', 'Standard Dev'])


def display_rank_graph(feature_names, metric,
                       title="", y_title="", x_title=""):
    plt.figure(figsize=(7, 7))

    # Init color ranking fo plot
    # Ref: http://tinyurl.com/ydgjtmty
    pal = sns.color_palette("GnBu_d", len(metric))
    rank = np.array(metric).argsort().argsort()
    ax = sns.barplot(y=feature_names, x=metric,
                     palette=np.array(pal[::-1])[rank])
    plt.xticks(rotation=0, fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel(x_title, fontsize=20, labelpad=20)
    plt.ylabel(y_title, fontsize=20, labelpad=20)
    plt.title(title, fontsize=15)
    plt.show()
    plt.close()


# General purpose model optimizer
def optimize_model_grid(model,
                        X_train,
                        y_train,
                        param_grid,
                        cv=10):

    # Instantiate the GridSearchCV object: logreg_cv
    model_cv = GridSearchCV(model, param_grid, cv=cv, n_jobs=-1)

    # Fit it to the data
    model_cv.fit(X_train, y_train)

    # Print the tuned parameters and score
    print("Tuned Parameters: {}".format(model_cv.best_params_))
    print("Best score on trained data was {0:4f}".format(model_cv.best_score_))

    model = type(model)(**model_cv.best_params_)

    return model_cv.best_params_


# Not created by me!
# Author: https://github.com/scikit-learn/scikit-learn/issues/7845
def report_to_dict(cr):
    # Parse rows
    tmp = list()
    for row in cr.split("\n"):
        parsed_row = [x for x in row.split("  ") if len(x) > 0]
        if len(parsed_row) > 0:
            tmp.append(parsed_row)

    # Store in dictionary
    measures = tmp[0]

    D_class_data = defaultdict(dict)
    for row in tmp[1:]:
        class_label = row[0]
        for j, m in enumerate(measures):
            D_class_data[class_label][m.strip()] = float(row[j + 1].strip())
    return D_class_data


# I am this lazy yes
def vertical_spacing(spaces=1):
    for _ in range(0, spaces):
        print()


# ## 1.) Import and analyze dataset

# In[4]:


heart_disease = pd.read_csv('datasets/heart_disease_patients.csv')
heart_disease.head()


# ### Research attributes
# <b>Link</b>:http://archive.ics.uci.edu/ml/datasets/Heart+Disease
# <b>age</b>: <p>&emsp; age in years</p>
# ***
# <b>sex</b>: <p>&emsp; 1 = male; 0 = female</p>
# ***
# <b>cp</b>: <p>&emsp; chest pain type
#     (1 = typical angina; 2 = atypical angina;
#     3 = non-anginal pain; 4 = asymptomatic)</p>
# ***
# <b>trestbps</b>: <p>&emsp; resting blood pressure:
#     (in mm Hg on admission to the hospital)</p>
# ***
# <b>chol</b>: <p>&emsp; Cholesterol: serum cholestoral in mg/dl</p>
# ***
# <b>fbs</b>: <p>&emsp; fasting blood sugar > 1
#     20 mg/dl (1 = true; 0 = false)</p>
# ***
# <b>restecg</b>: <p>&emsp; restin
#     g electrocardiographic results (
#     0 = normal; 1 = having ST-T; 2 = hypertrophy)</p>
# ***
# <b>thalach</b>: <p>&emsp; maximum heart rate achieved</p>
# ***
# <b>exang</b>: <p>&emsp; exercise induced angina(
#     1 = yes; 0 = no)</p>
# ***
# <b>oldpeak</b>: <p>&emsp; ST depression induced b
#     y exercise relative to rest</p>
# ***
# <b>slope</b>: <p>&emsp; the slope of the peak e
#     xercise ST segment (1 = upsloping; 2 = flat; 3 = downsloping)</p>
# ***

# ### Generate master Encoder/Decoder dict by hand

# In[5]:


# http://archive.ics.uci.edu/ml/datasets/Heart+Disease
master_encoder = dict()
master_decoder = dict()

master_encoder["sex"] = {"female": 0, "male": 1}
master_decoder["sex"] = {v: k for k, v in master_encoder["sex"].items()}

master_encoder["cp"] = {"typical angina": 1,
                        "atypical angina": 2, "non-anginal pain": 3,
                        "asymptomatic": 4}
master_decoder["cp"] = {v: k for k, v in master_encoder["cp"].items()}

master_encoder["fbs"] = {"false": 0, "true": 1}
master_decoder["fbs"] = {v: k for k, v in master_encoder["fbs"].items()}

master_encoder["restecg"] = {"normal": 0, "having ST-T": 1, "hypertrophy": 2}
master_decoder["restecg"] = {v: k for k,
                             v in master_encoder["restecg"].items()}

master_encoder["exang"] = {"no": 0, "yes": 1}
master_decoder["exang"] = {v: k for k, v in master_encoder["exang"].items()}

master_encoder["slope"] = {"upsloping": 1, "flat": 2, "downsloping": 3}
master_decoder["slope"] = {v: k for k, v in master_encoder["slope"].items()}


# ### Define features as categorical/numerical by hand

# In[6]:


categorical_features = {"sex", "cp", "fbs", "restecg", "exang", "slope"}
numerical_features = {"age", "trestbps", "chol", "thalach", "oldpeak"}
integer_features = {"age", "trestbps", "chol", "thalach"}
float_features = {"oldpeak"}


# ## 2.) Basic Data Cleaning/Checking

# #### Remove unwanted features

# In[7]:


heart_disease.drop('id', axis=1, inplace=True)


# ### Look at data types of each feature

# In[8]:


heart_disease.dtypes


# In[9]:


heart_disease.isna().any()


# ### Decode the data

# In[10]:


decoded_heart_disease = encode_decode_df(passed_df=heart_disease,
                                         encoder_decoder_map=master_decoder)
decoded_heart_disease.head()


# ## 3.) Analyze data

# ### Inspect at correlate matrix

# In[11]:


corr_metrics = heart_disease.corr()
corr_metrics.style.background_gradient()


# In[12]:


corr_feature_means = []
for feature_name in decoded_heart_disease.columns:
    corr_feature_means.append(corr_metrics[feature_name].mean())


# In[13]:


display_rank_graph(feature_names=decoded_heart_disease.columns,
                   metric=corr_feature_means,
                   title="Average Feature Correlation",
                   y_title="Correlation Average",
                   x_title="Features")


# <p>&emsp; Oldpeak stands out as having heavy correlation with slope.
#     As well as having the most correlation.
#     Might be a good idea to drop the feature... </p>

# ### Plot data

# In[14]:


sns.set(style="darkgrid")
sns.set_palette("muted")
sns.set(rc={'figure.figsize': (10, 7)})

for feature_name in decoded_heart_disease.columns:

    plt.title("Feature: " + feature_name,
              fontsize=15)
    if feature_name not in numerical_features:
        plt.ylabel("count", fontsize=15)
        sns.countplot(decoded_heart_disease[feature_name])
    else:
        sns.distplot(decoded_heart_disease[feature_name])

    plt.xlabel(feature_name, fontsize=15)
    plt.show()
    plt.close()
    vertical_spacing(2)


# ### Quick General Analsysis
# <b>age</b>: <p>&emsp; The dataset showed
#     slightly skewed distribtuion of ages of people.
#     Most common age is between 51 to 60.
#     Np.log might help a tiny bit here; and
#     the removal of outliers will help.</p>
# ***
# <b>sex</b>: <p>&emsp; Around twice
#     as many males as females in this dataset. </p>
# ***
# <b>cp</b>: <p>&emsp; The information showed chest
#     pain type was mainly asymptomatic for people. </p>
# ***
# <b>trestbps</b>: <p>&emsp; Seems to be decent
#     distribution between the values of 110 to 170.
#     Outliers causing a slight problem.
#     Graph is appearing more skewed then it should.
#     Np.log seems like it would help drastically here.
#     Outliers might be able to stay...might.</p>
# ***
# <b>chol</b>: <p>&emsp; Distribution between
#     the values of 150 to 350.
#     Skewed distribution caused from outliers.
#     Np.log required and removal of outliers.</p>
# ***
# <b>fbs</b>: <p>&emsp; More common to find people
#     not having fasting blood sugar levels.</p>
# ***
# <b>restecg</b>: <p>&emsp; The data showed people
#     having a ST-T was the most rare by a large margin.
#     While having a normal and
#     hypertrophic had very similar counts. </p>
# ***
# <b>thalach</b>: <p>&emsp; Distribution between the
#     values of 100 to 200.
#     Slightly skewed data due to outliers.
#     But might be the best numeri distr in the df.</p>
# ***
# <b>exang</b>: <p>&emsp; Statistically there were more
#     people did <b>not</b> have exercise induced angina.</p>
# ***
# <b>oldpeak</b>: <p>&emsp; Horrid distribution. Np.log
#     has no chance of solving this distribution.
#     God can't even help this distribution.
#     Possible suggestion: delete column. </p>
# ***
# <b>slope</b>: <p>&emsp; The slope of the peak
#     exercise ST segment was usually upsloping or flat
#     (which they were around the same). Rarely downsloping.</p>
# ***
# <b>Additionally, the following features adhere
#     to the documentation encoded
#     values by the values being in between
#     the specfied range.</b>

# ### Attempt to center out numerical data

# In[15]:


for feature_name in numerical_features:

    positive_only_vector = np.where(decoded_heart_disease[feature_name] < 1e-5,
                                    1e-8, decoded_heart_disease[feature_name])

    plt.title("Feature: " + feature_name)
    sns.distplot(np.log(positive_only_vector))
    plt.show()
    plt.close()


# ## 4.) Data transformation

# <p>Due to heavy correlation, skewed
#     distributions, and
#     inability to center out with np.log;I
#     have decided to remove the f
#     eature 'oldpeak'.</p>

# ### Feature 'oldpeak' and 'slope' removal

# In[16]:


decoded_heart_disease.drop('oldpeak', axis=1, inplace=True)
numerical_features.remove("oldpeak")
float_features.remove("oldpeak")

decoded_heart_disease.drop('slope', axis=1, inplace=True)
categorical_features.remove("slope")


# ### Remove 'having ST-T' from restecg

# In[17]:


decoded_heart_disease = decoded_heart_disease[
    decoded_heart_disease["restecg"] != "having ST-T"]
heart_disease = encode_decode_df(passed_df=decoded_heart_disease,
                                 encoder_decoder_map=master_encoder)


# In[18]:


heart_disease.head()


# ### Removal of numerical outliers

# #### Init removal dict with specify removal values

# In[19]:


outlier_removal_dict = dict()
outlier_removal_dict["age"] = {"High": None,
                               "Low": 30}
outlier_removal_dict["chol"] = {"High": 410,
                                "Low": None}
outlier_removal_dict["thalach"] = {"High": None,
                                   "Low": 90}
outlier_removal_dict["trestbps"] = {"High": 185,
                                    "Low": None}


# #### Peform Removal and re-init dataframes

# In[20]:


print("Old dataframe size {0}".format(heart_disease.shape))


# In[21]:


heart_disease = remove_outliers_df(
    df=heart_disease, removal_dict=outlier_removal_dict)
heart_disease.head()

display(heart_disease.head())
display(heart_disease.shape)


# In[22]:


encoded_heart_disease = heart_disease
decoded_heart_disease = encode_decode_df(passed_df=encoded_heart_disease,
                                         encoder_decoder_map=master_decoder)
display(decoded_heart_disease.head())
display(decoded_heart_disease.shape)


# #### Np.Log the following the features internally in the dataset

# In[23]:


for feature_name in numerical_features:

    heart_disease[feature_name] = np.log(
        np.where(heart_disease[feature_name] < 1e-5,
                 1e-8, heart_disease[feature_name]))

    plt.title("Feature: " + feature_name)
    sns.distplot(heart_disease[feature_name])
    plt.show()
    plt.close()


# #### One hot encode dataframe

# In[24]:


heart_disease = pd.get_dummies(heart_disease,
                               columns=list(categorical_features),
                               prefix=list(categorical_features))
heart_disease.head()


# ### Scale current data and inspect scaled data

# In[25]:


scaler = StandardScaler()
scaled = scaler.fit_transform(heart_disease)


# In[26]:


inspect_feature_matrix(matrix=scaled,
                       feature_names=heart_disease.columns)


# In[27]:


scaled


# #### Apply PCA to scaled matrix and inspect scaled data

# In[28]:


# Create PCA instance: model
pca_model = PCA(random_state=9814)

# Apply the fit_transform method of model to scaled
scaled = pca_model.fit_transform(scaled)


# In[29]:


inspect_feature_matrix(matrix=scaled,
                       feature_names=heart_disease.columns)


# In[30]:


scaled


# #### Re-apply scaler after PCA applied

# In[31]:


scaled = scaler.fit_transform(scaled)
inspect_feature_matrix(matrix=scaled,
                       feature_names=heart_disease.columns)


# In[32]:


scaled


# ## 5.) Start clustering!!!

# ### K-Means modeling

# #### Store models for dynamic usage for the future.

# In[33]:


kmeans_models = dict()


# #### Generate specified KMeans model

# In[34]:


k = 5
first_clust = KMeans(n_clusters=k, random_state=10).fit(scaled)
kmeans_models["kmeans_cluster_5"] = first_clust


# ### Find the best k value for KMeans

# In[35]:


ks = range(1, 15)
inertias = []

for k in ks:
    # Create a KMeans instance with k clusters: model
    model = KMeans(n_clusters=k,
                   random_state=10).fit(scaled)

    # Append the inertia to the list of inertias
    inertias.append(model.inertia_)

# Plot ks vs inertias
plt.figure(figsize=(13, 6))
plt.plot(ks, inertias, '-o')
plt.xlabel('number of clusters, k')
plt.ylabel('inertia')
plt.xticks(ks)
plt.show()
plt.close()


# #### "Elbow" k value looks to be about 8.

# In[36]:


kmeans_models["kmeans_cluster_8"] = KMeans(n_clusters=8,
                                           random_state=10).fit(scaled)


# ## Peform Hierarchical clustering to confirm 'k' value

# ### Graph cluster to confirm proper k values

# In[37]:


# dendrogram_methods = ["complete",
#                       "single",
#                       "weighted",
#                       "ward",
#                       "average",
#                       "centroid",
#                       "median"]
best_found_methods = ["ward"]

for method in best_found_methods:
    # Calculate the linkage: mergings
    mergings = linkage(scaled, method=method)

    # Plot the dendrogram, using varieties as labels
    dendrogram(mergings,
               labels=list(range(0, len(scaled))),
               leaf_rotation=90,
               leaf_font_size=3,
               )
    plt.title("Hierarchical Clustering Method : " + method)
    plt.show()
    plt.close()


# <p>Hierarchical Clustering Method
#     shows that there should be between 6-7 clusters.</p>

# In[38]:


kmeans_models["kmeans_cluster_7"] = KMeans(n_clusters=8,
                                           random_state=10).fit(scaled)


# ### Compare models on cluster counts and visualization

# In[39]:


markers = ["+", "*", ".", "o", "v", "P", "H", "X"]
colors = ['b', 'g', 'r', 'c', 'm', 'y', '#007BA7', '#ff69b4']
for model_name, kmeans_model in kmeans_models.items():

    # Display ranking on color based on amount data points per cluster
    unique, counts = np.unique(kmeans_model.labels_, return_counts=True)
    cluster_names = ["Cluster:" + str(cluster_label)
                     for cluster_label in unique]
    display_rank_graph(feature_names=cluster_names,
                       metric=counts,
                       title=model_name,
                       y_title="Clusters",
                       x_title="People per cluster")
    vertical_spacing(2)

    # Display clustered graph
    cluster_array = list(range(0, len(cluster_names)))
    scaled_cluster_label = np.hstack(
        (scaled, np.reshape(
            kmeans_model.labels_.astype(int), (scaled.shape[0], 1))))
    for i in range(0, scaled_cluster_label.shape[0]):
        cluster_label = int(scaled_cluster_label[i][-1])
        cluster_array[cluster_label] = pl.scatter(
            scaled_cluster_label[i, 0], scaled_cluster_label[i, 1],
            c=colors[cluster_label], marker=str(markers[cluster_label]))

    pl.legend(cluster_array, cluster_names)
    pl.title(model_name + ' visualized with data', fontsize=15)
    pl.show()
    pl.close()
    plt.close()

    # Spacing for next model
    vertical_spacing(5)


# <p>Our count plots shows k=7
#     for kmeans to have the best
#     of the given distributions.
#     But <b>ALL</b> of our
#     models show the data as not
#     being very clusterable with kmeans.</p>

# ## 6.) Create clustering profiles for best model

# ### Select "best" model to create profiles for each cluster

# #### Generate clustered dataframes

# In[40]:


best_model_name = "kmeans_cluster_7"


# In[41]:


# Re-init dataframes with labels
decoded_heart_disease["Cluster_Name"] = kmeans_models[best_model_name].labels_
encoded_heart_disease = encode_decode_df(passed_df=decoded_heart_disease,
                                         encoder_decoder_map=master_encoder)

# Dataframe to analyze model 'better' choices
model_choices_df = encoded_heart_disease.drop(
    'Cluster_Name', axis=1).drop(encoded_heart_disease.index)

# Store each sub-dataframe based on cluster label
clustered_dataframes = dict()

for cluster_label in set(kmeans_models[best_model_name].labels_):
    cluster_df = encoded_heart_disease[
        encoded_heart_disease["Cluster_Name"] ==
        cluster_label]

    # Ignore cluster with only one patient
    if len(cluster_df) <= 1:
        continue
    # ---
    zscore_cluster_df = cluster_df.drop(
        'Cluster_Name', axis=1).apply(zscore)

    # Check if cluster is only comprised of one data point
    if cluster_df.shape[0] > 1:

        # Iterate through all numerical features
        for numerical_feature in numerical_features:

            # Check for nans
            if not zscore_cluster_df[numerical_feature].isnull().values.any():
                zscore_cluster_df = zscore_cluster_df[
                    zscore_cluster_df[numerical_feature] >= -2]
                zscore_cluster_df = zscore_cluster_df[
                    zscore_cluster_df[numerical_feature] <= 2]

    # Dummy list of -1s alloc at given pos of 'zscore_cluster_df' indexs
    reshaped_index = [-1] * len(encoded_heart_disease.index.values)

    for given_index in list(zscore_cluster_df.index.values):
        reshaped_index[given_index] = given_index

    # Pass back all vectors that passed the zscore test
    bool_array = pd.Series(reshaped_index).astype(int) == pd.Series(
        list(encoded_heart_disease.index.values)).astype(int)

    temp_cluster_df = encoded_heart_disease[bool_array].reset_index(drop=True)

    # Store in proper collection objs
    model_choices_df = model_choices_df.append(temp_cluster_df)

    clustered_dataframes[
        "Cluster:" + str(cluster_label)] = temp_cluster_df.drop(
        'Cluster_Name', axis=1)


# In[42]:


cluster_profiles_df = pd.DataFrame(columns=encoded_heart_disease.columns).drop(
    'Cluster_Name', axis=1)
rows_count = 0
for cluster_identfier, cluster_dataframe in clustered_dataframes.items():
    df = pd.DataFrame(columns=cluster_dataframe.columns)
    df = df.append(cluster_dataframe.mean(), ignore_index=True)
    df.index = [cluster_identfier]

    # Attempt to convert numbers found within the full set of data
    for col in cluster_dataframe.columns:
        if col not in float_features:
            df[col] = find_nearest(numbers=encoded_heart_disease[
                col].value_counts().index.tolist(),
                                   target=df[col].values[0])

    # Evaluate cluster dataframe by dataframe
    eval_df = pd.DataFrame(columns=cluster_dataframe.columns)
    eval_df = eval_df.append(cluster_dataframe.mean(), ignore_index=True)
    eval_df = eval_df.append(cluster_dataframe.min(), ignore_index=True)
    eval_df = eval_df.append(cluster_dataframe.median(), ignore_index=True)
    eval_df = eval_df.append(cluster_dataframe.max(), ignore_index=True)
    eval_df = eval_df.append(cluster_dataframe.std(), ignore_index=True)
    eval_df = eval_df.append(cluster_dataframe.var(), ignore_index=True)
    eval_df.index = ["Mean", "Min", "Median",
                     "Max", "Standard Deviation", "Variance"]

    print("Total found in {0} is {1}".format(
        cluster_identfier, cluster_dataframe.shape[0]))
    display(df)
    display(eval_df)

    cluster_profiles_df = cluster_profiles_df.append(
        encode_decode_df(passed_df=df,
                         encoder_decoder_map=master_decoder))
    vertical_spacing(7)

    rows_count += cluster_dataframe.shape[0]
# End clusters loop

print("Total points in all shrunken clusters: ", rows_count)
print("Removed {0} points to get more concise: ".format(
    heart_disease.shape[0] - rows_count))


# In[43]:


cluster_profiles_df


# <p>&emsp; Very little difference in profiles.
#     Each feature isnt very varying.
#     Data seems to not cluster very well.
#     Will go more in detail in report</p>

# ## 7.) Visualize kmeans cluster choices with decision tree.

# In[44]:


model_choices_df.reset_index(drop=True,
                             inplace=True)
display(model_choices_df.head())
display(model_choices_df.shape)


# ### Train test split on model kmeans model choices

# In[45]:


X = np.array(model_choices_df.drop('Cluster_Name', axis=1, inplace=False))
y = np.array(model_choices_df['Cluster_Name'])

# Split dataframe into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.35,
                                                    random_state=528)


# ### Hyperparamters for dt

# In[46]:


# Find best parameters for model
param_grid = {
    "max_depth": list(range(1, 5)),
    "min_samples_leaf": list(range(10, 35, 5)),
    "criterion": ["gini", "entropy"],
}

best_param = optimize_model_grid(
    model=DecisionTreeClassifier(),
    X_train=X_train,
    y_train=y_train,
    param_grid=param_grid
)

# Train our decision tree with 'best' parameters
tree = DecisionTreeClassifier(**best_param)
tree.fit(X_train, y_train)

train_pred = tree.predict(X_train)
test_pred = tree.predict(X_test)


# ### Look at confusion matrix for both train and test

# In[47]:


skplt.metrics.plot_confusion_matrix(y_train, train_pred)
plt.show()
plt.close()


# In[48]:


skplt.metrics.plot_confusion_matrix(y_test, test_pred)
plt.show()
plt.close()


# ### Evaluate Results

# In[49]:


tree_stats_test = pd.DataFrame(
    report_to_dict(
        classification_report(y_test,
                              test_pred))).T
display(tree_stats_test)
print("Test accuracy is {0:2f}".format(accuracy_score(y_test, test_pred)))


# In[50]:


feature_names = list(model_choices_df.columns)
feature_names.remove('Cluster_Name')

