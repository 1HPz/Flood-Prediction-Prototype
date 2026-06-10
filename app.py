import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# --- 1. SET UP PAGE ---
st.set_page_config(
    page_title="Flood Prediction - User Friendly Locations",
    page_icon="🗺️",
    layout="wide"
)

st.title("🗺️ ระบบทำนายน้ำท่วมรายพื้นที่")
st.write("เลือกพื้นที่ที่คุณต้องการตรวจสอบจากเมนูด้านซ้าย (แสดงพิกัดที่เข้าใจง่ายแทนรหัส H3)")
st.markdown("---")

# --- 2. LOAD DATA & TRAIN MODEL ---
@st.cache_data
def load_data_and_model():
    try:
        # โหลดข้อมูลจริงจากไฟล์ของคุณ
        df = pd.read_csv('merged_dataset_v2.csv')
    except FileNotFoundError:
        # จำลองกรณีไม่มีไฟล์ข้อมูล
        np.random.seed(42)
        n_samples = 50
        mock_data = {
            'h3_index': [f'852f1ad{i}ffffff' for i in range(n_samples)],
            'center_lat': np.random.uniform(13.7, 13.9, n_samples),
            'center_lon': np.random.uniform(100.5, 100.7, n_samples),
            'elevation': np.random.uniform(1, 15, n_samples),
            'slope': np.random.uniform(0, 5, n_samples),
            'rainfall_intensity': np.random.uniform(120, 280, n_samples),
            'distance_to_river': np.random.uniform(100, 4000, n_samples),
            'urban_percentage': np.random.uniform(30, 95, n_samples),
            'flood_count': np.random.poisson(lam=2, size=n_samples)
        }
        df = pd.DataFrame(mock_data)

    # แปลงตัวเลือก H3 เป็นข้อความที่เข้าใจง่ายขึ้นให้ User อ่านใน Selectbox
    # ผลลัพธ์จะเป็น เช่น -> "📍 พิกัด 13.756, 100.501 (เคยท่วม 3 ครั้ง)"
    if 'center_lat' in df.columns and 'center_lon' in df.columns:
        df['user_friendly_name'] = df.apply(
            lambda r: f"📍 พิกัด ({r['center_lat']:.3f}, {r['center_lon']:.3f}) " + 
                      (f"[เคยท่วม {int(r['flood_count'])} ครั้ง]" if 'flood_count' in df.columns else ""),
            axis=1
        )
    else:
        df['user_friendly_name'] = df['h3_index']

    drop_cols = ['h3_index', 'flood_count', 'is_water', 'total_days', 'center_lat', 'center_lon', 'user_friendly_name']
    actual_drop_cols = [col for col in drop_cols if col in df.columns]
    
    X = df.drop(columns=actual_drop_cols)
    y = df['flood_count'] if 'flood_count' in df.columns else np.random.poisson(1, len(df))
    
    # เทรนโมเดล
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = xgb.XGBRegressor(
        objective='count:poisson', n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42
    )
    model.fit(X_train, y_train)
    
    return model, df, X.columns.tolist()

with st.spinner("กำลังเตรียมระบบข้อมูลรายพื้นที่..."):
    model, main_df, feature_names = load_data_and_model()

# --- 3. SIDEBAR: USER FRIENDLY LOCATION SELECTOR ---
st.sidebar.header("📍 ค้นหาและเลือกพื้นที่")

# ให้ผู้ใช้เลือกจากชื่อภาษาไทย/พิกัดที่เราจัดฟอร์แมตไว้ แทนรหัส H3 บูดๆ
location_options = main_df['user_friendly_name'].tolist()
selected_display_name = st.sidebar.selectbox(
    "เลือกพื้นที่ที่ต้องการตรวจสอบ:", 
    location_options,
    help="ระบบแปลงรหัสเชิงพื้นที่ (H3 Index) เป็นพิกัดจริงเพื่อความง่ายต่อการใช้งาน"
)

# ดึงแถวข้อมูลจริงที่ตรงกับตัวเลือกของ User
selected_row = main_df[main_df['user_friendly_name'] == selected_display_name].iloc[0]

st.sidebar.markdown("---")
st.sidebar.subheader("🔄 จำลองสถานการณ์เพิ่มเติม (Simulation)")

# ดึงข้อมูลจากจุดที่เลือกมาใส่เป็น Default ในสไลเดอร์
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
    st.subheader("🗺️ แผนที่ระบุตำแหน่งพื้นที่")
    if 'center_lat' in main_df.columns and 'center_lon' in main_df.columns:
        map_data = pd.DataFrame({
            'lat': [selected_row['center_lat']],
            'lon': [selected_row['center_lon']]
        })
        st.map(map_data, zoom=13)
        st.caption(f"🆔 รหัสอ้างอิงระบบเบื้องหลัง (H3 Index): {selected_row['h3_index']}")
    
    st.write("📋 **คุณลักษณะทางกายภาพของพื้นที่นี้:**")
    st.dataframe(input_df.T.rename(columns={0: "ค่าปัจจุบัน"}), use_container_width=True)

with col2:
    st.subheader("🚀 ผลลัพธ์การทำนายความเสี่ยง")
    
    prediction = model.predict(input_df)[0]
    
    st.metric(
        label="คาดการณ์จำนวนครั้งที่น้ำจะท่วม", 
        value=f"{prediction:.2f} ครั้ง"
    )
    
    if prediction < 1.0:
        st.success("🟢 ความเสี่ยงต่ำ: สภาพแวดล้อมค่อนข้างปลอดภัย")
    elif prediction < 3.0:
        st.warning("🟡 ความเสี่ยงปานกลาง: ควรเฝ้าระวังเมื่อมีพายุหรือฝนตกสะสม")
    else:
        st.error("🔴 ความเสี่ยงสูง: โครงสร้างพื้นที่เสี่ยงต่อการเกิดภัยน้ำท่วมสูง")
        
    if 'flood_count' in main_df.columns:
        st.markdown("---")
        st.info(f"📅 **สถิติจากดาวเทียม:** ในช่วงเวลาที่บันทึกข้อมูล พื้นที่นี้เคยน้ำท่วมจริงมาแล้ว **{int(selected_row['flood_count'])}** ครั้ง")
