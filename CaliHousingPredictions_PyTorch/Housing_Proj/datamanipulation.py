# 1) Design model (input, output size, forward pass)
# 2) Construct loss and optimizer
# 3) Training loop
#       - forward pass: compute prediction
#       - backward pass: gradients
#       - update weights
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

#------DATA MANIPULATION-----#

#changing data using pandas (ONE HOT ENCODING and SCALING)
df = pd.DataFrame(pd.read_csv('/Users/calebfernandes/Desktop/CaliHousingPredictions_PyTorch/data/California_Housing_Cities.csv'))
#ONE HOT ENCODING
ohe = OneHotEncoder()
feature_array = ohe.fit_transform(df[["ocean_proximity", "City"]]).toarray() #gets the values of the array turns them into an array
feature = ohe.categories_ #all the unique values in the arrays
feature_labels = np.concatenate((feature[0], feature[1])) #combines the two feature arrays
features = pd.DataFrame(feature_array, columns=feature_labels) #combines the names of the columns with the values
#dropping the columns because they are not needed anymore
df = df.drop(['Latitude', 'Longitude', 'ocean_proximity', 'City'], axis=1)
final_df = pd.concat([df, features.set_index(df.index)], axis=1)

#converting to csv
#creating test and train sets
train, test = train_test_split(final_df, shuffle=True, test_size=0.2)

#scaling data
scaler = MinMaxScaler(feature_range=(0,1))
train = scaler.fit_transform(train)
test = scaler.transform(test)
#NOTE: SCALED VALUES
print("\nNote: total_earnings values were scaled by multiplying by {:.10f} and adding {:.6f}\n".format(scaler.scale_[8], scaler.min_[8]))
#converting to DF
train = pd.DataFrame(train, columns=final_df.columns.values)
test = pd.DataFrame(test, columns=final_df.columns.values)
#creating prediction files
pred1 = pd.DataFrame(train.iloc[[0]])
pred2 = pd.DataFrame(train.iloc[[1]])
pred3 = pd.DataFrame(train.iloc[[2]])
pred4 = pd.DataFrame(train.iloc[[3]])
pred5 = pd.DataFrame(train.iloc[[4]])
# #convert to csv files
train.to_csv("tf_train_df.csv", index=False)
test.to_csv("tf_test_df.csv", index=False)
# # #prediction files
pred1.to_csv("pred1.csv", index=False)
pred2.to_csv("pred2.csv", index=False)
pred3.to_csv("pred3.csv", index=False)
pred4.to_csv("pred4.csv", index=False)
pred5.to_csv("pred5.csv", index=False)