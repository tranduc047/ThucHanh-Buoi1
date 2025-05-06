import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import StackingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
import shap
from sklearn.ensemble import IsolationForest
from statsmodels.stats.outliers_influence import variance_inflation_factor

# 1. Tải và chuẩn bị dữ liệu
boston = fetch_california_housing(as_frame=True)
df = boston.frame
df['PRICE'] = boston.target

# 2. Khám phá dữ liệu
print(df.info())
print(df.describe())
sns.pairplot(df)
plt.show()

# 3. Kiểm tra mối quan hệ Pearson với biến mục tiêu
correlation = df.corr()['PRICE'].sort_values(ascending=False)
print(correlation)

# 4. Xử lý ngoại lai bằng Isolation Forest
iso_forest = IsolationForest(contamination=0.05)
outliers = iso_forest.fit_predict(df.drop(columns='PRICE'))
df_clean = df[outliers == 1]

# 5. Kiểm tra đa cộng tuyến với VIF
X = df_clean.drop(columns='PRICE')
vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
print(vif_data)

# 6. Tạo đặc trưng mới
df_clean['room_per_crime'] = df_clean['AveRooms'] / df_clean['AveOccup']
df_clean['high_tax'] = (df_clean['AveOccup'] > df_clean['AveOccup'].median()).astype(int)
df_clean['RM_LSTAT'] = df_clean['AveRooms'] * df_clean['AveOccup']

# 7. Tạo đặc trưng phi tuyến
poly = PolynomialFeatures(degree=2, include_bias=False)
poly_features = poly.fit_transform(df_clean[['AveRooms', 'AveOccup']])
df_poly = pd.DataFrame(poly_features, columns=poly.get_feature_names_out(['AveRooms', 'AveOccup']))
df_clean = pd.concat([df_clean, df_poly], axis=1)

# 8. Chuẩn bị dữ liệu huấn luyện
X = df_clean.drop(columns='PRICE')
y = df_clean['PRICE']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 9. Xây dựng pipeline tiền xử lý
numeric_features = X.select_dtypes(include=['float64', 'int64']).columns
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features)
    ])

# 10. Huấn luyện mô hình
models = [
    ('lr', LinearRegression()),
    ('gbr', GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3)),
    ('mlp', MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000))
]

for name, model in models:
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('regressor', model)])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    print(f"{name} - MSE: {mean_squared_error(y_test, y_pred):.2f}, "
          f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}, "
          f"R²: {r2_score(y_test, y_pred):.2f}, "
          f"MAPE: {mean_absolute_percentage_error(y_test, y_pred) * 100:.2f}%")

# 11. Kết hợp mô hình với Stacking
stacking_model = StackingRegressor(
    estimators=models,
    final_estimator=LinearRegression()
)
stacking_model.fit(X_train, y_train)
y_pred_stacking = stacking_model.predict(X_test)
print(f"Stacking - MSE: {mean_squared_error(y_test, y_pred_stacking):.2f}, "
      f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_stacking)):.2f}, "
      f"R²: {r2_score(y_test, y_pred_stacking):.2f}, "
      f"MAPE: {mean_absolute_percentage_error(y_test, y_pred_stacking) * 100:.2f}%")

# 12. Phân tích SHAP cho mô hình Gradient Boosting
explainer = shap.Explainer(models[1][1], X_test)
shap_values = explainer(X_test)
shap.summary_plot(shap_values, X_test)
