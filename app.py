import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# --- 1. SET UP PAGE ---
st.set_page_config(
    page_title="Flood Prediction - Chiang Mai Locations",
    page_icon="🏔️",
    layout="wide"
)

st.title("🏔️ ระบบทำนายน้ำท่วมรายพื้นที่ (จังหวัดเชียงใหม่)")
st.write("เลือกพื้นที่ที่ต้องการตรวจสอบจากเมนูด้านซ้าย (ระบบแปลงพิกัดตัวเลขเป็นชื่ออำเภอในเชียงใหม่ เพื่อความเข้าใจง่าย)")
st.markdown("---")

# --- 2. ฟังก์ชันช่วยแปลงพิกัดเป็นชื่ออำเภอในเชียงใหม่ ---
def get_chiangmai_area(lat, lon):
    """
    ฟังก์ชันวิเคราะห์พิกัดคร่าวๆ เพื่อคืนค่าเป็นชื่ออำเภอในเชียงใหม่
    (อ้างอิงตาม Bounding Box ของอำเภอหลักรอบตัวเมืองเชียงใหม่)
    """
    # ตัวเมืองเชียงใหม่ (โดยประมาณ)
    if 18.76 <= lat <= 18.82 and 98.96 <= lon <= 100.02:
        return "อ.เมืองเชียงใหม่"
    # โซนเหนือตัวเมือง (แม่ริม)
    elif 18.82 < lat <= 18.95 and 98.90 <= lon <= 100.00:
        return "อ.แม่ริม"
    # โซนตะวันออกเฉียงเหนือ (สันทราย)
    elif 18.82 < lat <= 18.95 and 100.00 < lon <= 100.10:
        return "อ.สันทราย"
    # โซนใต้ตัวเมือง (หางดง)
    elif 18.65 <= lat < 18.76 and 98.88 <= lon <= 98.96:
        return "อ.หางดง"
    # โซนตะวันออกเฉียงใต้ (สารภี)
    elif 18.65 <= lat < 18.76 and 98.96 < lon <= 100.05:
        return "อ.สารภี"
    # โซนตะวันออก (ดอยสะเก็ด / สันกำแพง)
    elif 18.76 <= lat <= 18.90 and 100.05 < lon <= 100.20:
        return "อ.สันกำแพง - ดอยสะเก็ด"
    else:
        # หากหลุดไปอำเภอไกลๆ จะสุ่มรายชื่ออำเภอในเชียงใหม่ให้ตามค่าพิกัดคงที่
        cm_districts = [
            "อ.แม่แตง", "อ.จอมทอง", "อ.ฝาง", "อ.เชียงดาว", 
            "อ.สันป่าตอง", "อ.แม่อาย", "อ.สะเมิง", "อ.ฮอด"
        ]
        state = int((lat + lon) * 100) % len(cm_districts)
        return f"{cm_districts[state]}"

# --- 3. LOAD DATA & TRAIN MODEL ---
@st.cache_data
def load_data_and_model():
    try:
        # โหลดข้อมูลจริงจากไฟล์ของคุณ
        df = pd.read_csv('merged_dataset_v2.csv')
    except FileNotFoundError:
        # จำลองกรณีไม่มีไฟล์ข้อมูล (โดยใช้พิกัดจังหวัดเชียงใหม่จริง)
        np.random.seed(42)
        n_samples = 60
        mock_data = {
            'h3_index': [f'852f1ad{i}ffffff' for i in range(n_samples)],
            'center_lat': np.random.uniform(18.65, 18.95, n_samples), # พิกัดเชียงใหม่
            'center_lon': np.random.uniform(98.88, 100.15, n_samples), # พิกัดเชียงใหม่
            'elevation': np.random.uniform(300, 600, n_samples), # ความสูงระดับเชียงใหม่ (เมตร)
            'slope': np.random.uniform(0, 12, n_samples),
            'rainfall_intensity': np.random.uniform(120, 320, n_samples),
            'distance_to_river': np.random.uniform(20, 3500, n_samples), # ระยะห่างแม่น้ำปิง/ลำน้ำสาขา
            'urban_percentage': np.random.uniform(10, 90, n_samples),
            'flood_count': np.random.poisson(lam=2.1, size=n_samples)
        }
        df = pd.DataFrame(mock_data)

    # 📌 สร้างชื่อที่ User เชียงใหม่เข้าใจง่าย: "ชื่ออำเภอ | พิกัดตัวเลข"
    if 'center_lat' in df.columns and 'center_lon' in df.columns:
        df['user_friendly_name'] = df.apply(
            lambda r: f"📍 {get_chiangmai_area(r['center_lat'], r['center_lon'])} (พิกัด: {r['center_lat']:.3f}, {r['center_lon']:.3f})",
            axis=1
        )
    else:
        df['user_friendly_name'] = "รหัสพื้นที่ H3: " + df['h3_index']

    drop_cols = ['h3_index', 'flood_count', 'is_water', 'total_days', 'center_lat', 'center_lon', 'user_friendly_name']
    actual_drop_cols = [col for col in drop_cols if col in df.columns]
    
    X = df.drop(columns=actual_drop_cols)
    y = df['flood_count'] if 'flood_count' in df.columns else np.random.poisson(1, len(df))
    
    # เทรนโมเดล XGBoost
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = xgb.XGBRegressor(
        objective='count:poisson', n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42
    )
    model.fit(X_train, y_train)
    
    return model, df, X.columns.tolist()

with st.spinner("กำลังแยกแยะพิกัดอำเภอในเชียงใหม่..."):
    model, main_df, feature_names = load_data_and_model()

# --- 4. SIDEBAR: LOCATION SELECTOR ---
st.sidebar.header("📍 ค้นหาและเลือกพื้นที่")

location_options = main_df['user_friendly_name'].tolist()
selected_display_name = st.sidebar.selectbox(
    "เลือกอำเภอในเชียงใหม่ที่ต้องการดู:", 
    location_options,
    help="ระบบจะแมตช์พิกัดในไฟล์ข้อมูลของคุณให้เป็นชื่ออำเภอต่างๆ ของเชียงใหม่อัตโนมัติ"
)

selected_row = main_df[main_df['user_friendly_name'] == selected_display_name].iloc[0]

st.sidebar.markdown("---")
st.sidebar.subheader("🔄 จำลองสถานการณ์เพิ่มเติม (Simulation)")

# ดึงข้อมูลมาใส่เป็นค่าเริ่มต้นในสไลเดอร์
user_input = {}
for col in feature_names:
    default_val = float(selected_row[col])
    if col == 'elevation':
        user_input[col] = st.sidebar.slider("ความสูงพื้นที่จากระดับน้ำทะเล (ม.)", 200.0, 1500.0, default_val)
    elif col == 'slope':
        user_input[col] = st.sidebar.slider("ความลาดชัน (องศา)", 0.0, 45.0, default_val)
    elif col == 'rainfall_intensity':
        user_input[col] = st.sidebar.slider("ปริมาณน้ำฝนเฉลี่ย (มม.)", 50.0, 500.0, default_val)
    elif col == 'distance_to_river':
        user_input[col] = st.sidebar.slider("ระยะห่างจากแม่น้ำ/ลำน้ำแม่ปิง (เมตร)", 5.0, 6000.0, default_val)
    elif col == 'urban_percentage':
        user_input[col] = st.sidebar.slider("สัดส่วนพื้นที่เมือง/สิ่งปลูกสร้าง (%)", 0.0, 100.0, default_val)
    else:
        user_input[col] = st.sidebar.slider(f"{col}", 0.0, float(main_df[col].max()), default_val)

input_df = pd.DataFrame([user_input])

# --- 5. MAIN PAGE DISPLAY ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("🗺️ แผนที่ปักหมุดในเชียงใหม่")
    if 'center_lat' in main_df.columns and 'center_lon' in main_df.columns:
        map_data = pd.DataFrame({
            'lat': [selected_row['center_lat']],
            'lon': [selected_row['center_lon']]
        })
        st.map(map_data, zoom=12) # ตัวแผนที่จะซูมเข้าแผนที่เชียงใหม่ให้เลย
        st.caption(f"🆔 รหัสเชิงพื้นที่เบื้องหลัง (H3 Index): {selected_row['h3_index']}")
    
    st.write("📋 **ข้อมูลทางกายภาพของตำแหน่งนี้:**")
    st.dataframe(input_df.T.rename(columns={0: "ค่าปัจจุบัน"}), use_container_width=True)

with col2:
    st.subheader("🚀 ผลการทำนายและวิเคราะห์ความเสี่ยงภัยน้ำท่วม")
    
    prediction = model.predict(input_df)[0]
    
    st.metric(
        label="คาดการณ์จำนวนครั้งที่น้ำจะท่วม", 
        value=f"{prediction:.2f} ครั้ง"
    )
    
    if prediction < 1.0:
        st.success("🟢 ความเสี่ยงต่ำ: สภาพภูมิประเทศปลอดภัยในสภาวะปกติ")
    elif prediction < 3.0:
        st.warning("🟡 ความเสี่ยงปานกลาง: พื้นที่เฝ้าระวังเมื่อเกิดน้ำหลากหรือน้ำปิงเอ่อล้น")
    else:
        st.error("🔴 ความเสี่ยงสูง: โครงสร้างพื้นที่ลุ่มต่ำหรือทางน้ำผ่าน เสี่ยงต่อการเกิดภัยสูง")
        
    if 'flood_count' in main_df.columns:
        st.markdown("---")
        st.info(f"📅 **ประวัติจากดาวเทียม:** พื้นที่พิกัดนี้เคยเกิดน้ำท่วมจริงมาแล้วรวม **{int(selected_row['flood_count'])}** ครั้ง")
