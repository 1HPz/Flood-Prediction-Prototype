import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# --- 1. SET UP PAGE ---
st.set_page_config(
    page_title="Flood Prediction - Location Selector",
    page_icon="🗺️",
    layout="wide"
)

st.title("🗺️ ระบบทำนายน้ำท่วมรายพื้นที่ (Location-based Prediction)")
st.write("เลือกพิกัดหรือสถานที่จากข้อมูลจริงเพื่อจำลองสถานการณ์และทำนายจำนวนครั้งน้ำท่วม")
st.markdown("---")

# --- 2. LOAD DATA & TRAIN MODEL ---
@st.cache_data
def load_data_and_model():
    try:
        # 🟢 โหลดข้อมูลจริงตามไฟล์ใน Notebook ของคุณ
        df = pd.read_csv('merged_dataset_v2.csv')
    except FileNotFoundError:
        # 🟡 กรณีไม่พบไฟล์ (เช่น รันเดโมสั้นๆ) จะสร้างข้อมูลจำลองที่มีคอลัมน์พิกัดให้แทน
        np.random.seed(42)
        n_samples = 100
        mock_data = {
            'h3_index': [f'852f1ad{i}ffffff' for i in range(n_samples)],
            'center_lat': np.random.uniform(13.5, 14.0, n_samples),
            'center_lon': np.random.uniform(100.4, 100.9, n_samples),
            'elevation': np.random.uniform(1, 50, n_samples),
            'slope': np.random.uniform(0, 15, n_samples),
            'rainfall_intensity': np.random.uniform(100, 250, n_samples),
            'distance_to_river': np.random.uniform(50, 3000, n_samples),
            'urban_percentage': np.random.uniform(10, 90, n_samples),
            'flood_count': np.random.poisson(lam=1.5, size=n_samples)
        }
        df = pd.DataFrame(mock_data)

    # รายชื่อคอลัมน์ที่ต้อง Drop ก่อนเข้าโมเดล (ตามที่กำหนดใน Notebook)
    drop_cols = ['h3_index', 'flood_count', 'is_water', 'total_days', 'center_lat', 'center_lon']
    actual_drop_cols = [col for col in drop_cols if col in df.columns]
    
    X = df.drop(columns=actual_drop_cols)
    y = df['flood_count'] if 'flood_count' in df.columns else np.random.poisson(1, len(df))
    
    # ฝึกสอนโมเดล XGBoost Poisson Regression
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = xgb.XGBRegressor(
        objective='count:poisson',
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    return model, df, X.columns.tolist()

with st.spinner("กำลังเตรียมข้อมูลและโมเดล..."):
    model, main_df, feature_names = load_data_and_model()

# --- 3. SIDEBAR: LOCATION SELECTION ---
st.sidebar.header("📍 เลือกสถานที่ / พิกัด")

# ตรวจสอบรูปแบบการระบุตำแหน่ง (ใช้ h3_index หรือ พิกัด Lat/Lon)
if 'h3_index' in main_df.columns:
    # สร้างเมนูให้เลือกตามรหัสพิกัดพื้นที่ (H3 Index)
    location_options = main_df['h3_index'].tolist()
    selected_loc = st.sidebar.selectbox("เลือกรหัสพื้นที่ (H3 Index):", location_options)
    # ดึงแถวข้อมูลของพื้นที่ที่เลือกมา
    selected_row = main_df[main_df['h3_index'] == selected_loc].iloc[0]
else:
    # หากไม่มี h3_index จะให้เลือกตามลำดับแถวแทน
    location_options = [f"พื้นที่ตำแหน่งที่ {i+1}" for i in range(len(main_df))]
    selected_loc = st.sidebar.selectbox("เลือกพื้นที่:", location_options)
    idx = location_options.index(selected_loc)
    selected_row = main_df.iloc[idx]

st.sidebar.markdown("---")
st.sidebar.subheader("🔄 ปรับแต่งค่าเพิ่มเติม (Simulation)")
st.sidebar.write("คุณสามารถปรับค่าของพื้นที่นี้เพิ่มเติมเพื่อดูการเปลี่ยนแปลงได้:")

# ดึงค่าตั้งต้นของสถานที่ที่เลือกมาใส่ใน Slider
user_input = {}
for col in feature_names:
    default_val = float(selected_row[col])
    if col == 'elevation':
        user_input[col] = st.sidebar.slider("ความสูงพื้นที่ (ม.)", 0.0, 100.0, default_val)
    elif col == 'slope':
        user_input[col] = st.sidebar.slider("ความลาดชัน (องศา)", 0.0, 30.0, default_val)
    elif col == 'rainfall_intensity':
        user_input[col] = st.sidebar.slider("ปริมาณน้ำฝน (มม.)", 50.0, 400.0, default_val)
    elif col == 'distance_to_river':
        user_input[col] = st.sidebar.slider("ระยะห่างจากแม่น้ำ (เมตร)", 10.0, 5000.0, default_val)
    elif col == 'urban_percentage':
        user_input[col] = st.sidebar.slider("ความเป็นเมือง (%)", 0.0, 100.0, default_val)
    else:
        user_input[col] = st.sidebar.slider(f"{col}", 0.0, float(main_df[col].max()), default_val)

input_df = pd.DataFrame([user_input])

# --- 4. MAIN PAGE DISPLAY ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📊 ข้อมูลและพิกัดของพื้นที่ที่เลือก")
    
    # แสดงพิกัดจริงบนแผนที่จำลองของ Streamlit (ถ้ามีข้อมูล Lat/Lon)
    if 'center_lat' in main_df.columns and 'center_lon' in main_df.columns:
        map_data = pd.DataFrame({
            'lat': [selected_row['center_lat']],
            'lon': [selected_row['center_lon']]
        })
        st.map(map_data, zoom=12)
        st.caption(f"📍 พิกัดทางภูมิศาสตร์: ละติจูด {selected_row['center_lat']:.4f}, ลองจิจูด {selected_row['center_lon']:.4f}")
    
    st.write("ค่าฟีเจอร์ปัจจุบันของพื้นที่นี้:")
    st.dataframe(input_df.T.rename(columns={0: "ค่าที่เปิดใช้งาน"}), use_container_width=True)

with col2:
    st.subheader("🚀 ผลการทำนายปริมาณความเสี่ยงน้ำท่วม")
    
    # ทำนายผล
    prediction = model.predict(input_df)[0]
    
    st.metric(
        label="คาดการณ์จำนวนครั้งที่น้ำจะท่วม", 
        value=f"{prediction:.2f} ครั้ง"
    )
    
    # แสดงระดับความเสี่ยง
    if prediction < 1.0:
        st.success("🟢 ความเสี่ยงต่ำ: พื้นที่ค่อนข้างปลอดภัย")
    elif prediction < 3.0:
        st.warning("🟡 ความเสี่ยงปานกลาง: ควรเฝ้าระวังเมื่อมีฝนตกชุก")
    else:
        st.error("🔴 ความเสี่ยงสูง: เป็นพื้นที่น้ำท่วมซ้ำซากหรือที่ลุ่มต่ำ")
        
    # แสดงค่าสถิติจริงที่เคยเกิดขึ้นในอดีตเปรียบเทียบ (ถ้ามีในชุดข้อมูล)
    if 'flood_count' in main_df.columns:
        st.markdown("---")
        st.info(f"📅 สถิติในอดีต: พื้นที่นี้เคยเกิดน้ำท่วมจริงมาแล้ว **{int(selected_row['flood_count'])}** ครั้ง")

# --- 5. FEATURE IMPORTANCE ---
st.markdown("---")
st.subheader("📈 ปัจจัยที่มีผลต่อการตัดสินใจของโมเดล")
fig, ax = plt.subplots(figsize=(10, 3))
xgb.plot_importance(model, max_num_features=5, importance_type='gain', color='dodgerblue', ax=ax)
plt.title('XGBoost Feature Importance (Gain)')
st.pyplot(fig)
