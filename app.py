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

st.title("🌊 ระบบจำลองการทำนายจำนวนครั้งน้ำท่วม (Prototype)")
st.write("โมเดลนี้พัฒนาขึ้นโดยใช้ XGBoost (Poisson Regression) สำหรับข้อมูลประเภทนับจำนวน (Count Data)")
st.markdown("---")

# --- 2. MOCK OR LOAD DATA ---
# สร้างข้อมูลจำลองที่มีลักษณะเดียวกับฟีเจอร์ใน merged_dataset_v2.csv (ยกเว้น drop_cols)
# เพื่อให้แอปสามารถรันเดโมได้ทันทีแม้ไม่มีไฟล์ข้อมูลจริง
@st.cache_data
def load_and_train_model():
    # ในกรณีที่มีไฟล์จริง ให้เปิดคอมเมนต์บรรทัดด้านล่างนี้:
    # df = pd.read_csv('merged_dataset_v2.csv')
    # drop_cols = ['h3_index', 'flood_count', 'is_water', 'total_days', 'center_lat', 'center_lon']
    # X = df.drop(columns=drop_cols)
    # y = df['flood_count']
    
    # --- สำหรับทำ Prototype: จำลองฟีเจอร์หลักๆ ที่มักใช้ในโมเดลน้ำท่วม ---
    np.random.seed(42)
    n_samples = 1000
    mock_data = {
        'elevation': np.random.uniform(0, 100, n_samples),          # ความสูงพื้นที่
        'slope': np.random.uniform(0, 30, n_samples),              # ความลาดชัน
        'rainfall_intensity': np.random.uniform(50, 300, n_samples), # ปริมาณน้ำฝน
        'distance_to_river': np.random.uniform(10, 5000, n_samples), # ระยะห่างจากแหล่งน้ำ
        'urban_percentage': np.random.uniform(0, 100, n_samples)     # พื้นที่สิ่งปลูกสร้าง
    }
    X = pd.DataFrame(mock_data)
    # จำลอง target (flood_count) ให้สัมพันธ์กับฟีเจอร์
    lam = np.exp(0.5 + 0.01 * X['rainfall_intensity'] - 0.01 * X['elevation'] - 0.0005 * X['distance_to_river'])
    y = np.random.poison(lam)
    
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
    return model, X.columns.tolist()

with st.spinner("กำลังโหลดและฝึกสอนโมเดลต้นแบบ..."):
    model, feature_names = load_and_train_model()

# --- 3. SIDEBAR FOR USER INPUT ---
st.sidebar.header("🎯 ปรับแต่งค่าฟีเจอร์เพื่อทำนาย")
st.sidebar.write("ใส่ข้อมูลของพื้นที่ที่ต้องการให้โมเดลประเมิน:")

# สร้าง Input Slider ตามชื่อฟีเจอร์อัตโนมัติ
user_input = {}
for col in feature_names:
    if col == 'elevation':
        user_input[col] = st.sidebar.slider("ความสูงของพื้นที่จากระดับน้ำทะเล (ม.)", 0.0, 100.0, 15.0)
    elif col == 'slope':
        user_input[col] = st.sidebar.slider("ความลาดชันของพื้นที่ (องศา)", 0.0, 30.0, 2.5)
    elif col == 'rainfall_intensity':
        user_input[col] = st.sidebar.slider("ปริมาณน้ำฝนสะสมเฉลี่ย (มม.)", 50.0, 300.0, 180.0)
    elif col == 'distance_to_river':
        user_input[col] = st.sidebar.slider("ระยะห่างจากแม่น้ำ/ลำคลอง (เมตร)", 10.0, 5000.0, 500.0)
    elif col == 'urban_percentage':
        user_input[col] = st.sidebar.slider("สัดส่วนความเป็นเมือง/คอนกรีต (%)", 0.0, 100.0, 60.0)
    else:
        user_input[col] = st.sidebar.slider(f"ฟีเจอร์: {col}", 0.0, 100.0, 50.0)

#แปลง Input เป็น DataFrame
input_df = pd.DataFrame([user_input])

# --- 4. MAIN PAGE DISPLAY ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📋 ข้อมูลพื้นที่ที่คุณเลือก")
    st.dataframe(input_df.T.rename(columns={0: "ค่าที่กำหนด"}), use_container_width=True)
    
    # ปุ่มกดทำนาย
    if st.button("🚀 คำนวณผลการทำนายน้ำท่วม", type="primary"):
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
    
    # พล็อต Feature Importance สไตล์เดียวกับใน Notebook
    fig, ax = plt.subplots(figsize=(6, 4.5))
    xgb.plot_importance(model, max_num_features=10, importance_type='gain', color='crimson', ax=ax)
    plt.title('XGBoost Feature Importance (Gain)')
    st.pyplot(fig)

# --- 5. MODEL PERFORMANCE FOOTER ---
st.markdown("---")
st.subheader("📉 ประสิทธิภาพของโมเดล (อ้างอิงจากผลการทดลอง)")
perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)
perf_col1.metric("RMSE (ความคลาดเคลื่อนเฉลี่ย)", "0.9531")
perf_col2.metric("MAE (ความคลาดเคลื่อนสมบูรณ์)", "0.6233")
perf_col3.metric("MAPE (เปอร์เซ็นต์ความคลาดเคลื่อน)", "35.33%")
perf_col4.metric("R-squared (การอธิบายความผันแปร)", "56.66%")
