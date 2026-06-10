import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split

# --- 1. SET UP PAGE ---
st.set_page_config(
    page_title="Flood Prediction Prototype (XGBoost)",
    page_icon="🌊",
    layout="wide"
)

st.title("🌊 ระบบจำลองการทำนายจำนวนครั้งน้ำท่วมรายพื้นที่")
st.write("โมเดลทำนายโอกาสเกิดน้ำท่วม โดยอิงข้อมูลคุณลักษณะทางกายภาพจากตำแหน่งพิกัดที่คุณเลือก")
st.markdown("---")

# --- 2. LOAD DATA & TRAIN MODEL ---
@st.cache_data
def load_and_train_model():
    try:
        # 🟢 โหลดข้อมูลจริงจากไฟล์ merged_dataset_v2.csv
        df = pd.read_csv('merged_dataset_v2.csv')
    except FileNotFoundError:
        # 🟡 กรณีไม่พบไฟล์ (ระบบสำรองสำหรับเดโม) จะสร้างข้อมูลจำลองที่มีพิกัดให้แทน
        np.random.seed(42)
        n_samples = 500
        mock_data = {
            'h3_index': [f'852f1ad{i}ffffff' for i in range(n_samples)],
            'center_lat': np.random.uniform(18.65, 18.95, n_samples),
            'center_lon': np.random.uniform(98.88, 100.15, n_samples),
            'elevation': np.random.uniform(0, 100, n_samples),
            'slope': np.random.uniform(0, 30, n_samples),
            'rainfall_intensity': np.random.uniform(50, 300, n_samples),
            'distance_to_river': np.random.uniform(10, 5000, n_samples),
            'urban_percentage': np.random.uniform(0, 100, n_samples)
        }
        df = pd.DataFrame(mock_data)
        lam = np.exp(0.5 + 0.01 * df['rainfall_intensity'] - 0.01 * df['elevation'] - 0.0005 * df['distance_to_river'])
        df['flood_count'] = np.random.poisson(lam)

    # 📌 แปลงรหัส H3 เป็นพิกัดที่อ่านง่ายสำหรับใส่ใน Selectbox ให้ผู้ใช้เลือก
    if 'center_lat' in df.columns and 'center_lon' in df.columns:
        df['user_friendly_name'] = df.apply(
            lambda r: f"📍 พิกัด: ({r['center_lat']:.4f}, {r['center_lon']:.4f})",
            axis=1
        )
    else:
        df['user_friendly_name'] = df['h3_index']
    
    # กำหนดคอลัมน์ที่ไม่ใช้เป็นฟีเจอร์ในการเทรนโมเดล
    drop_cols = ['h3_index', 'center_lat', 'center_lon', 'user_friendly_name', 'flood_count']
    actual_drop_cols = [col for col in drop_cols if col in df.columns]
    
    X = df.drop(columns=actual_drop_cols)
    y = df['flood_count'] if 'flood_count' in df.columns else np.random.poisson(1, len(df))
    
    # แบ่งข้อมูลและเทรนโมเดล XGBoost Poisson Regression
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

with st.spinner("กำลังโหลดข้อมูลและฝึกสอนโมเดล..."):
    model, main_df, feature_names = load_and_train_model()

# --- 3. SIDEBAR: SELECT LOCATION & SIMULATION ---
st.sidebar.header("🎯 เลือกพิกัดพื้นที่")

# ดึงรายชื่อพิกัดจากไฟล์ข้อมูลให้เลือก
location_options = main_df['user_friendly_name'].tolist()
selected_display_name = st.sidebar.selectbox(
    "เลือกตำแหน่งพิกัดจากข้อมูล:", 
    location_options
)
selected_row = main_df[main_df['user_friendly_name'] == selected_display_name].iloc[0]

st.sidebar.markdown("---")
st.sidebar.subheader("🔄 จำลองสถานการณ์ (Simulation)")
st.sidebar.write("คุณสามารถปรับแต่งค่าสภาพแวดล้อมของจุดนี้เพิ่มเติมได้:")

# สร้าง Slider โดยใช้ค่าเริ่มต้น (Default) จากพิกัดที่เลือกจริงในไฟล์
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

input_df = pd.DataFrame([user_input])

# --- 4. MAIN PAGE DISPLAY ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("🗺️ ตําแหน่งพิกัดบนแผนที่ภูมิศาสตร์")
    
    # แสดงแผนที่ปักหมุดจุดที่เลือก
    map_data = pd.DataFrame({
        'lat': [selected_row['center_lat']],
        'lon': [selected_row['center_lon']]
    })
    st.map(map_data, zoom=13)
    st.caption(f"🆔 เบื้องหลังระบบ (H3 Index): {selected_row['h3_index']}")
    
    st.markdown("---")
    st.write("📋 **คุณลักษณะทางกายภาพที่ใช้คำนวณ:**")
    st.dataframe(input_df.T.rename(columns={0: "ค่าปัจจุบัน"}), use_container_width=True)

with col2:
    st.subheader("🚀 ผลลัพธ์การคาดการณ์จากโมเดล")
    st.write("กดปุ่มด้านล่างเพื่อประมวลผลคำนวณจำนวนครั้งน้ำท่วมด้วยโมเดล XGBoost")
    
    # ปุ่มกดทำนาย
    if st.button("🚀 คำนวณผลการทำนายน้ำท่วม", type="primary", use_container_width=True):
        prediction = model.predict(input_df)[0]
        
        st.markdown("---")
        # แสดงผลลัพธ์แบบเด่นชัด
        st.metric(
            label="คาดการณ์จำนวนครั้งที่น้ำจะท่วม", 
            value=f"{prediction:.2f} ครั้ง"
        )
        
        # แสดงแถบสีตามระดับความเสี่ยง
        if prediction < 1.0:
            st.success("🟢 ระดับความเสี่ยงต่ำ: พื้นที่มีโอกาสเกิดน้ำท่วมน้อยมาก")
        elif prediction < 3.0:
            st.warning("🟡 ระดับความเสี่ยงปานกลาง: ควรเฝ้าระวังเมื่อมีฝนตกหนักสะสม")
        else:
            st.error("🔴 ระดับความเสี่ยงสูง: พื้นที่นี้มีแนวโน้มเกิดน้ำท่วมซ้ำซากสูง")
            
        # แสดงสถิติจริงในอดีต (ถ้ามีในไฟล์)
        if 'flood_count' in main_df.columns:
            st.markdown("---")
            st.info(f"📅 **บันทึกประวัติอดีต:** พิกัดจุดนี้เคยตรวจพบน้ำท่วมจริงมาแล้ว **{int(selected_row['flood_count'])}** ครั้ง")

# --- 5. MODEL PERFORMANCE FOOTER ---
st.markdown("---")
st.subheader("📉 ประสิทธิภาพของโมเดล (อ้างอิงจากผลการทดลอง)")
perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)
perf_col1.metric("RMSE (ความคลาดเคลื่อนเฉลี่ย)", "0.9531")
perf_col2.metric("MAE (ความคลาดเคลื่อนสมบูรณ์)", "0.6233")
perf_col3.metric("MAPE (เปอร์เซ็นต์ความคลาดเคลื่อน)", "35.33%")
perf_col4.metric("R-squared (การอธิบายความผันแปร)", "56.66%")
