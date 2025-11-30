import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

df_clean = pd.read_csv("cleaned_internship_data.csv")

categorical_features = ['Car Condition', 'Weather', 'Traffic Condition']
numerical_features = df_clean.drop(columns=['fare_amount'] + categorical_features).columns

preprocessor = ColumnTransformer(
    transformers=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ],
    remainder='passthrough'
)

X_transformed = preprocessor.fit_transform(df_clean.drop(columns=['fare_amount']))

feature_names = list(preprocessor.named_transformers_['onehot'].get_feature_names_out(categorical_features)) + list(numerical_features)
X = pd.DataFrame(X_transformed, columns=feature_names)
y = df_clean['fare_amount']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lgb_model = lgb.LGBMRegressor(
    objective='regression',
    metric='rmse',
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=31,
    max_depth=-1,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

print("Training LightGBM model...")
lgb_model.fit(X_train, y_train,
              eval_set=[(X_test, y_test)],
              eval_metric='rmse',
              callbacks=[lgb.early_stopping(100, verbose=False)])

y_pred = lgb_model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"\nModel: LightGBM Regressor")
print(f"Root Mean Squared Error (RMSE): ${rmse:.4f}")
print(f"R-squared (R^2): {r2:.4f}")

plt.figure(figsize=(10, 6))
subset_size = 500
plt.scatter(y_test[:subset_size], y_pred[:subset_size], alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.title('Actual vs. Predicted Fare Amounts (LightGBM)')
plt.xlabel('Actual Fare Amount ($)')
plt.ylabel('Predicted Fare Amount ($)')
plt.grid(True)
plt.savefig('actual_vs_predicted_fare_amounts_lgbm.png')

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
import gradio as gr
import numpy as np

# --- 1. إعداد النموذج والمُعالِج الأولي (Preprocessor) ---

# تحميل البيانات
df = pd.read_csv('final_internship_data.csv')
# الأعمدة التي تم إسقاطها
columns_to_drop = ['User ID', 'User Name', 'Driver Name', 'key', 'pickup_datetime']
df_clean = df.drop(columns=columns_to_drop)

# تحديد الميزات الفئوية والرقمية
categorical_features = ['Car Condition', 'Weather', 'Traffic Condition']
numerical_features = df_clean.drop(columns=['fare_amount'] + categorical_features).columns

# إنشاء محول العمود لـ One-Hot Encoding
preprocessor = ColumnTransformer(
    transformers=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ],
    remainder='passthrough'
)

# تجهيز البيانات للتدريب
X = df_clean.drop(columns=['fare_amount'])
y = df_clean['fare_amount']

# تدريب المُعالِج الأولي على جميع البيانات
preprocessor.fit(X)

# تحويل البيانات واستخراج أسماء الميزات النهائية
X_transformed = preprocessor.transform(X)
feature_names = list(preprocessor.named_transformers_['onehot'].get_feature_names_out(categorical_features)) + list(numerical_features)
X_processed = pd.DataFrame(X_transformed, columns=feature_names)

# تقسيم البيانات للتدريب
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

# تدريب نموذج Random Forest Regressor
rf_model = RandomForestRegressor(
    n_estimators=150,
    max_depth=15,
    random_state=42,
    n_jobs=-1,
    min_samples_leaf=5
)
rf_model.fit(X_train, y_train)

# --- 2. دالة التنبؤ التي تستخدمها Gradio ---

def predict_fare(car_condition, weather, traffic_condition,
                 pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude,
                 passenger_count, hour, day, month, weekday, year,
                 jfk_dist, ewr_dist, lga_dist, sol_dist, nyc_dist, distance, bearing):

    # بناء إطار بيانات للمدخلات (يجب أن يتطابق ترتيب الأعمدة مع البيانات الأصلية)
    input_data = pd.DataFrame([{
        'Car Condition': car_condition,
        'Weather': weather,
        'Traffic Condition': traffic_condition,
        'pickup_longitude': pickup_longitude,
        'pickup_latitude': pickup_latitude,
        'dropoff_longitude': dropoff_longitude,
        'dropoff_latitude': dropoff_latitude,
        'passenger_count': passenger_count,
        'hour': hour,
        'day': day,
        'month': month,
        'weekday': weekday,
        'year': year,
        'jfk_dist': jfk_dist,
        'ewr_dist': ewr_dist,
        'lga_dist': lga_dist,
        'sol_dist': sol_dist,
        'nyc_dist': nyc_dist,
        'distance': distance,
        'bearing': bearing
    }])

    # تطبيق المُعالِج الأولي على المدخلات
    processed_input = preprocessor.transform(input_data)
    processed_df = pd.DataFrame(processed_input, columns=feature_names)

    # التنبؤ بالأجرة
    prediction = rf_model.predict(processed_df)[0]
    
    # تنسيق الإخراج
    return f"القيمة المتوقعة للأجرة: ${prediction:.2f}"

# --- 3. إطلاق واجهة Gradio ---

# تحديد مكونات الإدخال
input_components = [
    gr.Dropdown(df_clean['Car Condition'].unique().tolist(), label="حالة السيارة", value='Very Good'),
    gr.Dropdown(df_clean['Weather'].unique().tolist(), label="حالة الطقس", value='sunny'),
    gr.Dropdown(df_clean['Traffic Condition'].unique().tolist(), label="حالة المرور", value='Flow Traffic'),
    
    # الإحداثيات (تم استخدام قيم نموذجية)
    gr.Slider(minimum=-1.30, maximum=-1.28, step=0.000001, value=-1.29132, label="خط الطول (Pickup Longitude)"),
    gr.Slider(minimum=0.70, maximum=0.72, step=0.000001, value=0.71092, label="خط العرض (Pickup Latitude)"),
    gr.Slider(minimum=-1.30, maximum=-1.28, step=0.000001, value=-1.29140, label="خط الطول (Dropoff Longitude)"),
    gr.Slider(minimum=0.70, maximum=0.72, step=0.000001, value=0.71136, label="خط العرض (Dropoff Latitude)"),
    
    # ميزات الوقت والعدد
    gr.Slider(minimum=1, maximum=6, step=1, value=1, label="عدد الركاب"),
    gr.Slider(minimum=0, maximum=23, step=1, value=16, label="الساعة (0-23)"),
    gr.Slider(minimum=1, maximum=31, step=1, value=5, label="اليوم (1-31)"),
    gr.Slider(minimum=1, maximum=12, step=1, value=1, label="الشهر (1-12)"),
    gr.Slider(minimum=0, maximum=6, step=1, value=1, label="اليوم من الأسبوع (0=الاثنين, 6=الأحد)"),
    gr.Slider(minimum=2009, maximum=2015, step=1, value=2010, label="السنة"),
    
    # ميزات المسافة
    gr.Number(value=44.66, label="المسافة لمطار JFK"),
    gr.Number(value=31.83, label="المسافة لمطار EWR"),
    gr.Number(value=23.13, label="المسافة لمطار LGA"),
    gr.Number(value=15.12, label="المسافة لمنطقة Sol"),
    gr.Number(value=8.75, label="المسافة لمركز نيويورك (NYC)"),
    gr.Number(value=8.45, label="المسافة الفعلية (Haversine Distance)"),
    gr.Number(value=-0.37, label="زاوية الاتجاه (Bearing)")
]

# إنشاء وإطلاق الواجهة
iface = gr.Interface(
    fn=predict_fare,
    inputs=input_components,
    outputs=gr.Text(label="نتيجة التنبؤ"),
    title="واجهة تطبيق التنبؤ بأجرة التاكسي (Random Forest)",
    description="استخدم المدخلات أدناه للحصول على تنبؤ دقيق لقيمة الأجرة ($) من نموذج التعلم الآلي المدرب."
)

iface.launch()
