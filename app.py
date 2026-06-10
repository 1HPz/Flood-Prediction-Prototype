import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# --- 1. SET UP PAGE ---
st.set_page_config(
    page_title="Flood Prediction Prototype (XGBoost)",
    page_icon="🌊",
    layout="wide"
)

st.title("🌊 ระบบจำลองการทำนายจำนวนครั้งน้ำท่วม (พร้อมระบบเลือกพิกัด)")
st.write("โมเดลนี้พัฒนาขึ้นโดยใช้ XGBoost (Poisson Regression) สำหรับข้อมูลประเภทนับจำนวน (Count Data)")
st.markdown("---")

# --- 2. ฟังก์ชันระบุชื่อพื้นที่คร่าวๆ จากพิกัดเชียงใหม่ ---
def get_chiangmai_area(lat, lon):
    if 18.76 <= lat <= 18.82 and 98.96 <= lon <= 100.02: return "อ.เมืองเชียงใหม่"
    elif 18.82 < lat <= 18.95 and 98.90 <= lon <= 100.00: return "อ.แม่ริม"
    elif 18.82 < lat <= 18.95 and 100.00 < lon <= 100.10: return "อ.สันทราย"
    elif 18.65 <= lat < 18.76 and 98.88 <= lon <= 98.96: return "อ.หางดง"
    elif 18.65 <= lat < 18.76 and 98.96 < lon <= 100.05: return "อ.สารภี"
    elif 18.76 <= lat <= 18.90 and 100.05 < lon <= 100.20: return "อ.สันกำแพง - ดอยสะเก็ด"
    else: return "พื้นที่เชียงใหม่ตอนนอก"

# --- 3. MOCK OR LOAD DATA ---
@st.cache_data
def load_and_train_model():
    # --- จำลองข้อมูลพิกัดและฟีเจอร์ตามที่คุณระบุ ---
    np.random.seed(42)
    n_samples = 1000
    mock_data = {
        'h3_index': [f'852f1ad{i}ffffff' for i in range(n_samples)],
        'center_lat': np.random.uniform(18.65, 18.95, n_samples),  # พิกัดเชียงใหม่
        'center_lon': np.random.uniform(98.88, 100.15, n_samples), # พิกัดเชียงใหม่
        'elevation': np.random.uniform(0, 100, n_samples),          # ความสูงพื้นที่
        'slope': np.random.uniform(0, 30, n_samples),              # ความลาดชัน
        'rainfall_intensity': np.random.uniform(50, 300, n_samples), # ปริมาณน้ำฝน
        'distance_to_river': np.random.uniform(10, 5000, n_samples), # ระยะห่างจากแหล่งน้ำ
        'urban_percentage': np.random.uniform(0, 100, n_samples)     # พื้นที่สิ่งปลูกสร้าง
    }
    df = pd.DataFrame(mock_data)
    
    # สร้างชื่อที่เข้าใจง่ายให้ User เลือก
    df['user_friendly_name'] = df.apply(
        lambda r: f"📍 {get_chiangmai_area(r['center_lat'], r['center_lon'])} (พิกัด: {r['center_lat']:.3f}, {r['center_lon']:.3f})",
        axis=1
    )
    
    # จำลอง target (flood_count) ให้สัมพันธ์กับฟีเจอร์
    lam = np.exp(0.5 + 0.01 * df['rainfall_intensity'] - 0.01 * df['elevation'] - 0.0005 * df['distance_to_river'])
    df['flood_count'] = np.random.poisson(lam) # แก้ไขจาก .poison เป็น .poisson แล้ว
    
    # กำหนดคอลัมน์ที่ไม่ใช้ในการเทรน
    drop_cols = ['h3_index', 'center_lat', 'center_lon', 'user_friendly_name', 'flood_count']
    X = df.drop(columns=drop_cols)
    y = df['flood_count']
    
    # แบ่งข้อมูล
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # อิงตาม Hyperparameters จาก Notebook ของคุณ
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

with st.spinner("กำลังโหลดและฝึกสอนโมเดลต้นแบบ..."):
    model, main_df, feature_names = load_and_train_model()

# --- 4. SIDEBAR FOR USER INPUT ---
st.sidebar.header("🎯 เลือกพื้นที่และปรับแต่งฟีเจอร์")

# 1. ให้เลือกพิกัดก่อน
location_options = main_df['user_friendly_name'].tolist()
selected_display_name = st.sidebar.selectbox(
    "เลือกพิกัดพื้นที่:", 
    location_options
)
selected_row = main_df[main_df['user_friendly_name'] == selected_display_name].iloc[0]

st.sidebar.markdown("---")
st.sidebar.write("ปรับค่าจำลองสถานการณ์สำหรับพื้นที่นี้:")

# 2. สร้าง Input Slider โดยดึงค่าตั้งต้นมาจากพิกัดที่เลือก
user_input = {}
for col in feature_names:
    default_val = float(selected_row[col])
    if col == 'elevation':
        user_input[col] = st.sidebar.slider("ความสูงของพื้นที่จากระดับน้ำทะเล (ม.)", 0.0, 100.0, default_val)
    elif col == 'slope':
        user_input[col] = st.sidebar.slider("ความลาดชันของพื้นที่ (องศา)", 0.0, 30.0, default_val)
    elif col == 'rainfall_intensity':
        user_input[col] = st.sidebar.slider("ปริมาณน้ำฝนสะสมเฉลี่ย (มม.)", 50.0, 300.0, default_val)
    elif col == 'distance_to_river':
        user_input[col] = st.sidebar.slider("ระยะห่างจากแม่น้ำ/ลำคลอง (เมตร)", 10.0, 5000.0, default_val)
    elif col == 'urban_percentage':
        user_input[col] = st.sidebar.slider("สัดส่วนความเป็นเมือง/คอนกรีต (%)", 0.0, 100.0, default_val)

# แปลง Input เป็น DataFrame เพื่อส่งเข้าโมเดล
input_df = pd.DataFrame([user_input])

# --- 5. MAIN PAGE DISPLAY ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("🗺️ ตำแหน่งและข้อมูลพื้นที่ที่คุณเลือก")
    
    # แสดงแผนที่ตามพิกัด
    map_data = pd.DataFrame({
        'lat': [selected_row['center_lat']],
        'lon': [selected_row['center_lon']]
    })
    st.map(map_data, zoom=13)
    
    st.dataframe(input_df.T.rename(columns={0: "ค่าที่กำหนด/ค่าปัจจุบัน"}), use_container_width=True)
    
    # ปุ่มกดทำนาย
    if st.button("🚀 คำนวณผลการทำนายน้ำท่วม", type="primary", use_container_width=True):
        prediction = model.predict(input_df)[0]
        
        st.markdown("---")
        st.subheader("📊 ผลลัพธ์การทำนาย")
        
        # แสดงผลลัพธ์แบบเด่นชัดด้วย Metric บ่งบอกจำนวนครั้ง
        st.metric(
            label="คาดการณ์จำนวนครั้งที่น้ำจะท่วม", 
            value=f"{prediction:.2f} ครั้ง"
        )
        
        # แจ้งเตือนระดับความเสี่ยงตามเกณฑ์จำนวนครั้ง
        if prediction < 1.0:
            st.success("🟢 ระดับความเสี่ยงต่ำ: พื้นที่มีโอกาสเกิดน้ำท่วมน้อยมาก")
        elif prediction < 3.0:
            st.warning("🟡 ระดับความเสี่ยงปานกลาง: ควรเฝ้าระวังเมื่อมีฝนตกหนักสะสม")
        else:
            st.error("🔴 ระดับความเสี่ยงสูง: พื้นที่นี้มีแนวโน้มเกิดน้ำท่วมซ้ำซากสูง")

with col2:
    st.subheader("📈 ความสำคัญของปัจจัย (Feature Importance)")
    st.write("กราฟแสดงว่าปัจจัยใดส่งผลต่อการตัดสินใจของโมเดล XGBoost (Gain)")
    
    # พล็อต Feature Importance
    fig, ax = plt.subplots(figsize=(6, 4.5))
    xgb.plot_importance(model, max_num_features=10, importance_type='gain', color='crimson', ax=ax)
    plt.title('XGBoost Feature Importance (Gain)')
    st.pyplot(fig)

# --- 6. MODEL PERFORMANCE FOOTER ---
st.markdown("---")
st.subheader("📉 ประสิทธิภาพของโมเดล (อ้างอิงจากผลการทดลอง)")
perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)
perf_col1.metric("RMSE (ความคลาดเคลื่อนเฉลี่ย)", "0.9531")
perf_col2.metric("MAE (ความคลาดเคลื่อนสมบูรณ์)", "0.6233")
perf_col3.metric("MAPE (เปอร์เซ็นต์ความคลาดเคลื่อน)", "35.33%")
perf_col4.metric("R-squared (การอธิบายความผันแปร)", "56.66%")
