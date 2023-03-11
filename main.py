import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Load the data from the CSV file
df = pd.read_csv('train_data_mod.csv')

# Convert categorical variables into dummy variables
df = pd.get_dummies(df, columns=['MealPlan', 'RoomType', 'MarketSegment'])

# Scale the numerical variables
scaler = StandardScaler()
df[['LeadTime', 'NumWeekendNights', 'NumWeekNights', 'NumAdults', 'NumChildren', 
    'RepeatedGuest', 'NumPrevCancellations', 'NumPreviousNonCancelled', 
    'AvgRoomPrice', 'SpecialRequests']] = scaler.fit_transform(df[['LeadTime', 'NumWeekendNights', 
                                                                     'NumWeekNights', 'NumAdults', 'NumChildren', 
                                                                     'RepeatedGuest', 'NumPrevCancellations', 
                                                                     'NumPreviousNonCancelled', 'AvgRoomPrice', 
                                                                     'SpecialRequests']])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df.drop('BookingStatus', axis=1), df['BookingStatus'], 
                                                    test_size=0.3, random_state=42)

# Train the logistic regression model
lr = LogisticRegression(random_state=42)
lr.fit(X_train, y_train)

# Evaluate the performance of the model
y_pred = lr.predict(X_test)
print('Accuracy:', accuracy_score(y_test, y_pred))
print('Precision:', precision_score(y_test, y_pred))
print('Recall:', recall_score(y_test, y_pred))
print('F1 Score:', f1_score(y_test, y_pred))
print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred))

# Use the model to predict the booking status for new customers
new_customer_data = pd.read_csv('new_customer_data.csv')
new_customer_data = pd.get_dummies(new_customer_data, columns=['MealPlan', 'RoomType', 'MarketSegment'])
new_customer_data[['LeadTime', 'NumWeekendNights', 'NumWeekNights', 'NumAdults', 'NumChildren', 
                   'RepeatedGuest', 'NumPrevCancellations', 'NumPreviousNonCancelled', 
                   'AvgRoomPrice', 'SpecialRequests']] = scaler.transform(new_customer_data[['LeadTime', 'NumWeekendNights', 
                                                                                                 'NumWeekNights', 'NumAdults', 
                                                                                                 'NumChildren', 'RepeatedGuest', 
                                                                                                 'NumPrevCancellations', 
                                                                                                 'NumPreviousNonCancelled', 
                                                                                                 'AvgRoomPrice', 'SpecialRequests']])
new_customer_booking_status = lr.predict(new_customer_data)
print('New customer booking status:', new_customer_booking_status)
