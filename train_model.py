import pandas as pd
import xgboost as xg
import matplotlib.pyplot as plt
import seaborn as sns
import joblib  # Added for saving the model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# --- STEP 1: LOAD DATA ---
# Using the exact path and columns from your dataset
file_path = r'D:\HACKATHON\house_prices_dataset.csv'
df = pd.read_csv(file_path)

# Defining features and target
X = df[['square_feet', 'num_rooms', 'age', 'distance_to_city(km)']]
y = df['price']

# --- STEP 2: TRAIN MODEL ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initializing XGBoost Regressor
model = xg.XGBRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=5,
    objective='reg:squarederror'
)

print("Training model... please wait.")
model.fit(X_train, y_train)

# --- STEP 3: SAVE THE MODEL ---
# Saving immediately after training so it's ready for deployment
save_path = r'D:\HACKATHON\house_model.pkl'
joblib.dump(model, save_path)
print(f"Model saved successfully at: {save_path}")

# --- STEP 4: EVALUATE & VISUALIZE ---
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\n" + "="*40)
print(f"MODEL TRAINING COMPLETE")
print(f"Accuracy (R2 Score): {r2:.4f}")
print(f"Mean Absolute Error: ${mae:,.2f}")
print("="*40 + "\n")

# --- STEP 5: FEATURE IMPORTANCE PLOT ---
def show_analysis():
    print("Generating Feature Importance Plot... (Close the plot window to continue to predictions)")
    importance = model.feature_importances_
    feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importance})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='magma')
    plt.title('Which Features Influence the Price Most?')
    plt.xlabel('Importance Score')
    plt.ylabel('Property Features')
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

# Show the graph once before starting the predictor
show_analysis()

# --- STEP 6: INTERACTIVE PREDICTION ---
def predict_house_price():
    print("\n--- LIVE PRICE PREDICTOR ---")
    print("Please answer the following property questions:")
    
    try:
        # User Inputs
        sq_ft = float(input("1. Total Area (Square Feet): "))
        rooms = float(input("2. Number of Rooms: "))
        age = float(input("3. Age of the House (Years): "))
        dist = float(input("4. Distance to City Center (km): "))

        # Convert input into the format the model expects
        user_data = pd.DataFrame([[sq_ft, rooms, age, dist]], 
                                 columns=['square_feet', 'num_rooms', 'age', 'distance_to_city(km)'])

        # Generate Prediction
        prediction = model.predict(user_data)[0]

        print("\n" + "*"*35)
        print(f"PREDICTED MARKET VALUE: ${prediction:,.2f}")
        print("*"*35 + "\n")

    except ValueError:
        print("\n[Error] Please enter numerical values only.")

def generate_advice(sq_ft, rooms, age, dist, score, price):
    advice = ""

   
# Run the prediction interface
if __name__ == "__main__":
    while True:
        predict_house_price()
        cont = input("Do another prediction? (y/n): ").lower()
        if cont != 'y':
            print("Exiting predictor. Good luck with your hackathon!")
            break