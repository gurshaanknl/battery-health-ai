# %%
import scipy.io

mat = scipy.io.loadmat('data/matfiles/B0005.mat')
print(mat.keys())

# %%
battery = mat['B0005'][0][0]
cycles = battery['cycle'][0]
print("Number of cycles:", len(cycles))
print("First cycle type:", cycles[0]['type'][0])
print("First cycle fields:", cycles[0]['data'].dtype.names)

# %%
# find a discharge cycle and check its fields
for i, cycle in enumerate(cycles):
    if cycle['type'][0] == 'discharge':
        print(f"Found discharge cycle at index {i}")
        print("Discharge cycle fields:", cycle['data'][0][0].dtype.names)
        break

# %%
import pandas as pd

rows = []
for i, cycle in enumerate(cycles):
    if cycle['type'][0] == 'discharge':
        d = cycle['data'][0][0]
        rows.append({
            'cycle_number': i,
            'ambient_temp': cycle['ambient_temperature'][0][0],
            'capacity': d['Capacity'][0][0],
            'voltage_avg': d['Voltage_measured'][0].mean(),
            'current_avg': d['Current_measured'][0].mean(),
            'temperature_avg': d['Temperature_measured'][0].mean(),
        })

df = pd.DataFrame(rows)
print("Total discharge cycles found:", len(df))
print(df.head())

# %%
df.to_csv('data/csvfiles/B0005_clean.csv', index=False)
print("Saved!")

# %%
def process_battery(filename, battery_key):
    mat = scipy.io.loadmat(f'data/matfiles/{filename}.mat')
    battery = mat[battery_key][0][0]
    cycles = battery['cycle'][0]

    rows = []
    for i, cycle in enumerate(cycles):
        if cycle['type'][0] == 'discharge':
            d = cycle['data'][0][0]
            rows.append({
                'cycle_number': i,
                'ambient_temp': cycle['ambient_temperature'][0][0],
                'capacity': d['Capacity'][0][0],
                'voltage_avg': d['Voltage_measured'][0].mean(),
                'current_avg': d['Current_measured'][0].mean(),
                'temperature_avg': d['Temperature_measured'][0].mean(),
            })

    result = pd.DataFrame(rows)
    result.to_csv(f'data/csvfiles/{filename}_clean.csv', index=False)
    print(f"{filename}: saved {len(result)} discharge cycles")
    return result

# %%
df_6 = process_battery('B0006', 'B0006')
df_7 = process_battery('B0007', 'B0007')
df_18 = process_battery('B0018', 'B0018')

# %%
df_5 = pd.read_csv('data/csvfiles/B0005_clean.csv')
df_6 = pd.read_csv('data/csvfiles/B0006_clean.csv')
df_7 = pd.read_csv('data/csvfiles/B0007_clean.csv')
df_18 = pd.read_csv('data/csvfiles/B0018_clean.csv')

all_data = pd.concat([df_5, df_6, df_7, df_18], ignore_index=True)
print("Total rows combined:", len(all_data))
print(all_data.head())

# %%
all_data['label'] = all_data['capacity'].apply(lambda c: 'good' if c >= 1.4 else 'worn_out')
print(all_data['label'].value_counts())

# %%
from sklearn.model_selection import train_test_split

features = all_data[['ambient_temp', 'voltage_avg', 'current_avg', 'temperature_avg']]
labels = all_data['label']

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42, stratify=labels)
print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))

# %%
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
model.fit(X_train, y_train)
print("Model trained!")

# %%
from sklearn.metrics import classification_report, confusion_matrix

predictions = model.predict(X_test)
print(classification_report(y_test, predictions))

# %%
print(confusion_matrix(y_test, predictions))

# %%
mat_25 = scipy.io.loadmat('data/matfiles/B0025.mat')
battery_25 = mat_25['B0025'][0][0]
cycles_25 = battery_25['cycle'][0]

for i, cycle in enumerate(cycles_25):
    if cycle['type'][0] == 'discharge':
        print(f"Found discharge cycle at index {i}")
        print("Discharge cycle fields:", cycle['data'][0][0].dtype.names)
        break


# %%
df_25 = process_battery('B0025', 'B0025')
df_26 = process_battery('B0026', 'B0026')
df_27 = process_battery('B0027', 'B0027')
df_28 = process_battery('B0028', 'B0028')


# %%
all_data = pd.concat([df_5, df_6, df_7, df_18, df_25, df_26, df_27, df_28], ignore_index=True)
print("Total rows combined:", len(all_data))

# %%
all_data['label'] = all_data['capacity'].apply(lambda c: 'good' if c >= 1.4 else 'worn_out')
print(all_data['label'].value_counts())

# %%
features = all_data[['ambient_temp', 'voltage_avg', 'current_avg', 'temperature_avg']]
labels = all_data['label']

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42, stratify=labels)

model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))

# %%
import joblib

joblib.dump(model, 'battery_model.pkl')
print("Model saved!")

# %%
worn_out_examples = all_data[all_data['label'] == 'worn_out']
print(worn_out_examples[['ambient_temp', 'voltage_avg', 'current_avg', 'temperature_avg', 'capacity']].head(10))